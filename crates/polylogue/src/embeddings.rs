//! Embeddings, opt-in with a cost preview:
//! Voyage, or a local model through an OpenAI-compatible endpoint. Vectors
//! are keyed by embedding key and NFC text hash in `embeddings.db` and are
//! never destroyed by a reindex.
//!
//! Settings (`polylogue setting KEY VALUE`):
//! - `embed.provider`: `local` or `voyage`; unset means embeddings are off;
//! - `embed.endpoint`: default `http://127.0.0.1:11434/v1` (local) or
//!   `https://api.voyageai.com/v1`;
//! - `embed.model`: default `qwen3-embedding:0.6b` (local) or `voyage-4`;
//! - `embed.price_per_mtok`: dollars per million tokens, for the preview;
//!   unset for a paid provider means the cost is shown as unknown.
//!
//! Startup supplies the Voyage credential captured from `VOYAGE_API_KEY` and never runs without
//! `--yes` after a preview.

use crate::daemon::Response;
use crate::daemon::service::{error, ok, store_error};
use polylogue_store::Reader;
use polylogue_store::vectors::{Vector, Vectors};
use serde::Serialize;
use serde_json::{Value, json};
use std::time::Instant;
use unicode_normalization::UnicodeNormalization;

/// Texts per request.
const BATCH: usize = 32;
/// Characters per chunk: long texts are embedded in chunks, and a text's
/// best chunk ranks it.
const CHUNK_CHARS: usize = 6000;
/// Texts read from the index per page.
const PAGE: u64 = 2000;

/// Secrets captured once at the startup boundary; diagnostics never print them.
#[derive(Clone, PartialEq, Eq)]
pub struct Credentials {
    voyage: Option<String>,
}
impl Credentials {
    pub fn new(voyage: Option<String>) -> Self {
        Self { voyage }
    }
}
impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials")
            .field("voyage", &self.voyage.as_ref().map(|_| "[redacted]"))
            .finish()
    }
}
pub struct Context<'a> {
    pub settings: &'a polylogue_model::settings::Snapshot,
    pub credentials: &'a Credentials,
    pub clock: &'a query_language::time::Clock,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderKind {
    Voyage,
    Local,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Provider {
    pub kind: ProviderKind,
    pub endpoint: String,
    pub model: String,
    pub price_per_mtok: Option<f64>,
    credentials: Credentials,
}

impl Provider {
    /// From settings; `None` while embeddings are off.
    pub fn from_settings(
        settings: &polylogue_model::settings::Snapshot,
        credentials: &Credentials,
    ) -> Result<Option<Self>, String> {
        use polylogue_model::settings::{EmbeddingProvider, Value};
        let kind = match settings.get("embed.provider") {
            Some(Value::EmbeddingProvider(EmbeddingProvider::Off)) => return Ok(None),
            Some(Value::EmbeddingProvider(EmbeddingProvider::Local)) => ProviderKind::Local,
            Some(Value::EmbeddingProvider(EmbeddingProvider::Voyage)) => ProviderKind::Voyage,
            _ => return Err("the typed settings snapshot lacks embed.provider".into()),
        };
        let (endpoint, model) = match kind {
            ProviderKind::Local => ("http://127.0.0.1:11434/v1", "qwen3-embedding:0.6b"),
            ProviderKind::Voyage => ("https://api.voyageai.com/v1", "voyage-4"),
        };
        let endpoint = match settings.get("embed.endpoint") {
            Some(Value::Endpoint(value)) => value.clone(),
            None => endpoint.into(),
            _ => return Err("invalid typed embed.endpoint".into()),
        };
        let model = match settings.get("embed.model") {
            Some(Value::Text(value)) => value.clone(),
            None => model.into(),
            _ => return Err("invalid typed embed.model".into()),
        };
        let price_per_mtok = match settings.get("embed.price_per_mtok") {
            Some(Value::NonnegativeNumber(value)) => Some(*value),
            None if kind == ProviderKind::Local => Some(0.0),
            None => None,
            _ => return Err("invalid typed embed.price_per_mtok".into()),
        };
        Ok(Some(Self {
            endpoint,
            model,
            kind,
            price_per_mtok,
            credentials: credentials.clone(),
        }))
    }

    fn name(&self) -> &'static str {
        match self.kind {
            ProviderKind::Voyage => "voyage",
            ProviderKind::Local => "local",
        }
    }

    /// The embedding key without its dimension count, which the first
    /// answer tells.
    pub fn key_prefix(&self) -> String {
        format!("{}/{}/", self.name(), self.model)
    }

    fn key(&self, dims: usize) -> String {
        format!("{}{dims}/document/unit", self.key_prefix())
    }

    /// Embeds `inputs`, returning one unit vector each.
    pub fn embed(&self, inputs: &[String], query: bool) -> Result<Vec<Vec<f32>>, EmbedError> {
        let limit = polylogue_store::read::limit::ReadLimit::current();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| EmbedError::Transport(error.to_string()))?;
        let result = runtime.block_on(async {
            let Some(limit) = limit else {
                return self
                    .request(inputs, query, Some(std::time::Duration::from_secs(60)))
                    .await;
            };
            let request = self.request(inputs, query, None);
            tokio::pin!(request);
            let mut tick = tokio::time::interval(std::time::Duration::from_millis(5));
            loop {
                tokio::select! {
                    result = &mut request => {
                        limit.check()?;
                        return result;
                    }
                    _ = tick.tick() => { limit.check()?; }
                }
            }
        });
        // Name resolution can use an OS worker; it must not keep this reader
        // waiting after its request future has been cancelled.
        runtime.shutdown_background();
        result
    }

    async fn request(
        &self,
        inputs: &[String],
        query: bool,
        timeout: Option<std::time::Duration>,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut body = json!({ "model": self.model, "input": inputs });
        if self.kind == ProviderKind::Voyage {
            body["input_type"] = json!(if query { "query" } else { "document" });
        }
        let url = format!("{}/embeddings", self.endpoint.trim_end_matches('/'));
        let failed = |error: reqwest::Error| EmbedError::Transport(error.to_string());
        let mut client = reqwest::Client::builder();
        if let Some(timeout) = timeout {
            client = client.timeout(timeout);
        }
        let client = client.build().map_err(failed)?;
        let mut request = client.post(&url).json(&body);
        if self.kind == ProviderKind::Voyage {
            let key = self
                .credentials
                .voyage
                .as_deref()
                .filter(|key| !key.is_empty())
                .ok_or_else(|| {
                    EmbedError::Config("the startup Voyage credential is unavailable".into())
                })?;
            request = request.bearer_auth(key);
        }
        let response = request.send().await.map_err(failed)?;
        let code = response.status().as_u16();
        if (400..500).contains(&code) && code != 429 {
            return Err(EmbedError::Rejected(format!("HTTP {code}")));
        }
        if response.status().is_client_error() || response.status().is_server_error() {
            return Err(EmbedError::Transport(format!("HTTP {code}")));
        }
        let v: Value = response.json().await.map_err(failed)?;
        let data = v["data"]
            .as_array()
            .ok_or_else(|| EmbedError::Transport(format!("no `data` in the answer: {v}")))?;
        if data.len() != inputs.len() {
            return Err(EmbedError::Transport(format!(
                "{} vectors for {} inputs",
                data.len(),
                inputs.len()
            )));
        }
        let mut out = vec![Vec::new(); inputs.len()];
        for (i, d) in data.iter().enumerate() {
            let idx = d["index"].as_u64().map_or(i, |x| x as usize);
            let values: Vec<f32> = d["embedding"]
                .as_array()
                .ok_or_else(|| EmbedError::Transport("an item has no `embedding`".into()))?
                .iter()
                .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                .collect();
            if let Some(slot) = out.get_mut(idx) {
                *slot = unit(values);
            }
        }
        Ok(out)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("{0}")]
    Config(String),
    /// The provider refused this input for good (HTTP 4xx).
    #[error("rejected: {0}")]
    Rejected(String),
    #[error("{0}")]
    Transport(String),
    #[error(transparent)]
    Store(#[from] polylogue_store::StoreError),
}

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

/// The text a vector is computed from: NFC, cut into chunks.
fn chunks(text: &str) -> Vec<String> {
    let nfc: String = text.nfc().collect();
    let chars: Vec<char> = nfc.chars().collect();
    if chars.len() <= CHUNK_CHARS {
        return vec![nfc];
    }
    chars
        .chunks(CHUNK_CHARS)
        .map(|c| c.iter().collect())
        .collect()
}

/// What a build would embed, and what it would cost.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Preview {
    pub provider: String,
    pub model: String,
    pub texts: u64,
    pub chunks: u64,
    pub characters: u64,
    /// About four characters per token; the provider counts exactly.
    pub tokens_estimate: u64,
    /// `None`: the price is unknown (set `embed.price_per_mtok`).
    pub cost_usd_estimate: Option<f64>,
    pub already_embedded: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Built {
    pub key: Option<String>,
    pub texts: u64,
    pub vectors: u64,
    pub failed: u64,
    pub seconds: f64,
    pub texts_per_second: f64,
}

/// Every prose text not yet embedded for this provider's model, page by
/// page; `f` returns false to stop.
fn each_pending(
    reader: &Reader<'_>,
    done: &std::collections::HashSet<Vec<u8>>,
    sessions: Option<&[String]>,
    mut f: impl FnMut(polylogue_store::vectors::ProseText) -> Result<bool, EmbedError>,
) -> Result<u64, EmbedError> {
    let mut after = 0;
    let mut skipped = 0;
    loop {
        let page = reader.prose_texts(after, PAGE, sessions)?;
        let Some(last) = page.last() else {
            return Ok(skipped);
        };
        after = last.id;
        for t in page {
            if done.contains(&t.hash) {
                skipped += 1;
                continue;
            }
            if !f(t)? {
                return Ok(skipped);
            }
        }
    }
}

/// Hashes already embedded under any dimension count of this model.
fn done_for(
    vectors: &Vectors,
    provider: &Provider,
) -> Result<std::collections::HashSet<Vec<u8>>, EmbedError> {
    let mut out = std::collections::HashSet::new();
    for k in vectors.status()?.keys {
        if k.key.starts_with(&provider.key_prefix()) {
            out.extend(vectors.done(&k.key)?);
        }
    }
    Ok(out)
}

pub fn preview(
    reader: &Reader<'_>,
    provider: &Provider,
    sessions: Option<&[String]>,
    limit: Option<u64>,
) -> Result<Preview, EmbedError> {
    let vectors = reader.vectors()?;
    let done = done_for(&vectors, provider)?;
    let mut p = Preview {
        provider: provider.name().into(),
        model: provider.model.clone(),
        ..Preview::default()
    };
    let skipped = each_pending(reader, &done, sessions, |t| {
        p.texts += 1;
        p.characters += t.text.chars().count() as u64;
        p.chunks += chunks(&t.text).len() as u64;
        Ok(limit.is_none_or(|l| p.texts < l))
    })?;
    p.already_embedded = skipped;
    p.tokens_estimate = p.characters.div_ceil(4);
    p.cost_usd_estimate = provider
        .price_per_mtok
        .map(|price| price * p.tokens_estimate as f64 / 1_000_000.0);
    Ok(p)
}

pub fn build(
    reader: &Reader<'_>,
    provider: &Provider,
    sessions: Option<&[String]>,
    limit: Option<u64>,
) -> Result<Built, EmbedError> {
    let started = Instant::now();
    let mut vectors = reader.vectors_writer()?;
    let done = done_for(&vectors, provider)?;
    let mut out = Built::default();
    // (text hash, chunk number, chunk text)
    let mut batch: Vec<(Vec<u8>, u32, String)> = Vec::new();
    let flush = |batch: &mut Vec<(Vec<u8>, u32, String)>,
                 out: &mut Built,
                 vectors: &mut Vectors|
     -> Result<(), EmbedError> {
        if batch.is_empty() {
            return Ok(());
        }
        let inputs: Vec<String> = batch.iter().map(|b| b.2.clone()).collect();
        let result = match provider.embed(&inputs, false) {
            Ok(v) => Ok(v),
            // One bad input fails the whole request: find it alone.
            Err(EmbedError::Rejected(_)) if inputs.len() > 1 => Err(None),
            Err(e) => Err(Some(e)),
        };
        let mut stored = Vec::new();
        match result {
            Ok(values) => {
                for ((hash, chunk, _), v) in batch.iter().zip(values) {
                    stored.push(Vector {
                        text_hash: hash.clone(),
                        chunk: *chunk,
                        values: v,
                    });
                }
            }
            Err(Some(e)) => return Err(e),
            Err(None) => {
                for (hash, chunk, text) in batch.iter() {
                    match provider.embed(std::slice::from_ref(text), false) {
                        Ok(mut v) => stored.push(Vector {
                            text_hash: hash.clone(),
                            chunk: *chunk,
                            values: v.remove(0),
                        }),
                        Err(EmbedError::Rejected(why)) => {
                            let key = out.key.clone().unwrap_or_else(|| provider.key(0));
                            vectors.fail(&key, hash, &why)?;
                            out.failed += 1;
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        if let Some(first) = stored.first() {
            let key = provider.key(first.values.len());
            out.key.get_or_insert_with(|| key.clone());
            vectors.put(&key, &stored)?;
            out.vectors += stored.len() as u64;
        }
        batch.clear();
        Ok(())
    };
    each_pending(reader, &done, sessions, |t| {
        for (i, c) in chunks(&t.text).into_iter().enumerate() {
            batch.push((t.hash.clone(), i as u32, c));
            if batch.len() >= BATCH {
                flush(&mut batch, &mut out, &mut vectors)?;
            }
        }
        out.texts += 1;
        Ok(limit.is_none_or(|l| out.texts < l))
    })?;
    flush(&mut batch, &mut out, &mut vectors)?;
    out.seconds = started.elapsed().as_secs_f64();
    out.texts_per_second = if out.seconds > 0.0 {
        out.texts as f64 / out.seconds
    } else {
        0.0
    };
    Ok(out)
}

/// For `similar:`: the query text as a unit vector, embedded as a query.
pub fn query_vector(provider: &Provider, text: &str) -> Result<Vec<f32>, EmbedError> {
    let nfc: String = text.nfc().collect();
    Ok(provider.embed(&[nfc], true)?.remove(0))
}

/// The embedding key `similar:` ranks by: the provider's model with vectors,
/// if any. `similar:` is unavailable without one.
pub fn ranking_key(reader: &Reader<'_>, provider: Option<&Provider>) -> Option<String> {
    let p = provider?;
    reader
        .vectors()
        .and_then(|v| v.key_with_prefix(&p.key_prefix()))
        .ok()
        .flatten()
}

/// The IDs of every session a sessions query returns, through all its
/// pages.
fn query_sessions(
    reader: &Reader<'_>,
    query: &str,
    clock: &query_language::time::Clock,
) -> Result<Vec<String>, Response> {
    let mut ids = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let ask = crate::daemon::query::Ask {
            text: query,
            cursor: cursor.as_deref(),
            page_size: Some(1_000),
            why: false,
            session: None,
            now: clock.now.as_millisecond(),
            day: None,
            embeddings: false,
            near: None,
            fields: &[],
        };
        let page = crate::daemon::query::run(reader, ask).map_err(|f| Response::Error {
            kind: f.kind,
            message: f.message,
            detail: f.detail,
        })?;
        if page["unit"] != "sessions" {
            return Err(error("bad_request", "embed takes a sessions query"));
        }
        ids.extend(
            page["rows"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|r| r["id"].as_str().map(str::to_owned)),
        );
        match page["next"].as_str() {
            Some(next) => cursor = Some(next.to_owned()),
            None => return Ok(ids),
        }
    }
}

/// Embedding status, cost preview and build ([`crate::embeddings`]).
pub fn handle(
    reader: &polylogue_store::Reader<'_>,
    what: &str,
    query: Option<&str>,
    limit: Option<u64>,
    yes: bool,
    context: Context<'_>,
) -> Response {
    let provider = match Provider::from_settings(context.settings, context.credentials) {
        Ok(p) => p,
        Err(e) => return error("bad_request", e),
    };
    if what == "status" {
        let vectors = reader.vectors().and_then(|v| v.status());
        return match vectors {
            Ok(s) => ok(serde_json::json!({
                "provider": provider.as_ref().map(|p| serde_json::json!({
                    "endpoint": p.endpoint, "model": p.model,
                })),
                "vectors": s,
            })),
            Err(e) => store_error(&e),
        };
    }
    let Some(provider) = provider else {
        return error(
            "embeddings_off",
            "embeddings are off; `polylogue setting embed.provider local` (or `voyage`) turns them on",
        );
    };
    // A query limits the work to the sessions it finds.
    let sessions: Option<Vec<String>> = match query.filter(|q| !q.trim().is_empty()) {
        None => None,
        Some(q) => match query_sessions(reader, q, context.clock) {
            Ok(ids) => Some(ids),
            Err(response) => return response,
        },
    };
    let result = match what {
        "preview" => preview(reader, &provider, sessions.as_deref(), limit)
            .map(|p| serde_json::to_value(p).unwrap_or_default()),
        "build" => {
            if provider.kind == ProviderKind::Voyage && !yes {
                return error(
                    "confirm",
                    "Voyage is paid: `polylogue embed preview` shows the cost; build with --yes",
                );
            }
            build(reader, &provider, sessions.as_deref(), limit)
                .map(|b| serde_json::to_value(b).unwrap_or_default())
        }
        other => {
            return error(
                "bad_request",
                format!("embed {other}: status, preview or build"),
            );
        }
    };
    match result {
        Ok(v) => ok(v),
        Err(e) => error("embed", e),
    }
}

#[cfg(test)]
mod tests;
