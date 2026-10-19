//! Embeddings, opt-in with a cost preview (architecture 7):
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
//! Voyage reads its key from `VOYAGE_API_KEY` and never runs without
//! `--yes` after a preview.

use polylogue_store::vectors::{Vector, Vectors};
use polylogue_store::{Archive, Reader};
use serde::Serialize;
use serde_json::{Value, json};
use std::time::Instant;
use unicode_normalization::UnicodeNormalization;

/// Texts per request.
const BATCH: usize = 32;
/// Characters per chunk: long texts are embedded in pieces, and a text's
/// best chunk ranks it.
const CHUNK_CHARS: usize = 6000;
/// Texts read from the index per page.
const PAGE: u64 = 2000;
/// Texts `near:` ranks by.
pub const NEAREST: usize = 500;

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
}

impl Provider {
    /// From settings; `None` while embeddings are off.
    pub fn from_settings(reader: &Reader<'_>, machine: &str) -> Result<Option<Self>, String> {
        let get = |k: &str| reader.setting(k, machine).map_err(|e| e.to_string());
        let kind = match get("embed.provider")?.as_deref() {
            None | Some("off" | "") => return Ok(None),
            Some("local") => ProviderKind::Local,
            Some("voyage") => ProviderKind::Voyage,
            Some(other) => {
                return Err(format!(
                    "embed.provider is {other:?}; it is `local`, `voyage` or `off`"
                ));
            }
        };
        let (endpoint, model) = match kind {
            ProviderKind::Local => ("http://127.0.0.1:11434/v1", "qwen3-embedding:0.6b"),
            ProviderKind::Voyage => ("https://api.voyageai.com/v1", "voyage-4"),
        };
        let price = match get("embed.price_per_mtok")? {
            Some(p) => Some(
                p.parse::<f64>()
                    .map_err(|_| format!("embed.price_per_mtok is not a number: {p:?}"))?,
            ),
            None if kind == ProviderKind::Local => Some(0.0),
            None => None,
        };
        Ok(Some(Self {
            endpoint: get("embed.endpoint")?.unwrap_or_else(|| endpoint.to_owned()),
            model: get("embed.model")?.unwrap_or_else(|| model.to_owned()),
            kind,
            price_per_mtok: price,
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
        let mut body = json!({ "model": self.model, "input": inputs });
        if self.kind == ProviderKind::Voyage {
            body["input_type"] = json!(if query { "query" } else { "document" });
        }
        let url = format!("{}/embeddings", self.endpoint.trim_end_matches('/'));
        let mut req = ureq::post(&url).header("content-type", "application/json");
        let auth;
        if self.kind == ProviderKind::Voyage {
            let key = std::env::var("VOYAGE_API_KEY")
                .map_err(|_| EmbedError::Config("VOYAGE_API_KEY is not set".into()))?;
            auth = format!("Bearer {key}");
            req = req.header("authorization", &auth);
        }
        let resp = req.send_json(&body);
        let mut resp = match resp {
            Ok(r) => r,
            Err(ureq::Error::StatusCode(code)) if (400..500).contains(&code) && code != 429 => {
                return Err(EmbedError::Rejected(format!("HTTP {code}")));
            }
            Err(e) => return Err(EmbedError::Transport(e.to_string())),
        };
        let v: Value = resp
            .body_mut()
            .read_json()
            .map_err(|e| EmbedError::Transport(e.to_string()))?;
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
    archive: &Archive,
    reader: &Reader<'_>,
    provider: &Provider,
    sessions: Option<&[String]>,
    limit: Option<u64>,
) -> Result<Preview, EmbedError> {
    let vectors = Vectors::open_read(archive)?;
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
    archive: &Archive,
    reader: &Reader<'_>,
    provider: &Provider,
    sessions: Option<&[String]>,
    limit: Option<u64>,
) -> Result<Built, EmbedError> {
    let started = Instant::now();
    let mut vectors = Vectors::open(archive)?;
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

/// For `near:`: the index's nearest texts to `text`, as the JSON the
/// query binds (`[[text id, similarity], ...]`). Embeds the query text
/// once with the provider.
pub fn nearest_json(
    archive: &Archive,
    reader: &Reader<'_>,
    provider: &Provider,
    text: &str,
) -> Result<String, EmbedError> {
    let vectors = Vectors::open_read(archive)?;
    let Some(key) = vectors
        .status()?
        .keys
        .into_iter()
        .map(|k| k.key)
        .find(|k| k.starts_with(&provider.key_prefix()))
    else {
        return Ok("[]".into());
    };
    let nfc: String = text.nfc().collect();
    let query = provider.embed(&[nfc], true)?.remove(0);
    let hits = vectors.nearest(&key, &query, NEAREST)?;
    let hashes: Vec<Vec<u8>> = hits.iter().map(|(h, _)| h.clone()).collect();
    let ids = reader.text_ids(&hashes)?;
    let pairs: Vec<Value> = hits
        .iter()
        .zip(ids)
        .filter_map(|((_, score), id)| id.map(|id| json!([id, score])))
        .collect();
    Ok(Value::Array(pairs).to_string())
}

/// Whether `near:` can rank: a provider is set and vectors exist for it.
pub fn available(archive: &Archive, provider: Option<&Provider>) -> bool {
    let Some(p) = provider else { return false };
    Vectors::open_read(archive)
        .and_then(|v| v.status())
        .is_ok_and(|s| s.keys.iter().any(|k| k.key.starts_with(&p.key_prefix())))
}

#[cfg(test)]
mod tests;
