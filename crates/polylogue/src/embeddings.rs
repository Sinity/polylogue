//! Embeddings, opt-in with a cost preview:
//! Voyage, or a local model through an OpenAI-compatible endpoint. Vectors
//! are keyed by embedding key and the BLAKE3 hash of the stored text bytes
//! (the vector itself is computed from the NFC text) in `embeddings.db`, so a
//! re-parse that yields the same text keeps them, and they are never
//! destroyed by a reindex.
//!
//! Settings (`polylogue set KEY VALUE`):
//! - `embed.provider`: `local` or `voyage`; unset means embeddings are off;
//! - `embed.calls`: `on` lets Polylogue send text to the provider (building
//!   vectors, embedding new `similar:` texts); default `off`, when stored
//!   vectors still rank and nothing is sent;
//! - `embed.endpoint`: default `http://127.0.0.1:11434/v1` (local) or
//!   `https://api.voyageai.com/v1`; for Voyage it must stay on Voyage's
//!   origin, because the Voyage key is sent nowhere else; for the local
//!   provider it must be a loopback address, so text never leaves the machine;
//! - `embed.model`: default `qwen3-embedding:0.6b` (local) or `voyage-4`;
//! - `embed.price_per_mtok`: dollars per million tokens, for the preview;
//!   unset for a paid provider means the cost is shown as unknown.
//!
//! Startup supplies the Voyage credential captured from `VOYAGE_API_KEY` and never runs without
//! `--yes` after a preview.

use crate::daemon::Failure;
use crate::daemon::service::error;
use command_surface::Answer as Answered;
use polylogue_model::JsonExt as _;
use polylogue_model::money::Amount;
use polylogue_store::Reader;
use polylogue_store::vectors::{Vector, Vectors};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use unicode_normalization::UnicodeNormalization;

/// Texts per request.
const BATCH: usize = 32;
use polylogue_store::vectors::CHUNK_CHARS;
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
    pub clock: &'a polylogue_model::WallClock,
    pub settings: &'a polylogue_model::settings::Snapshot,
    pub credentials: &'a Credentials,
    pub operation: &'a std::sync::Arc<polylogue_reading::operation::Context>,
}

/// The only origin the Voyage key is ever sent to.
const VOYAGE_ORIGIN: &str = "https://api.voyageai.com";

/// Whether `endpoint` is on Voyage's own origin (scheme, host and port).
fn is_voyage_origin(endpoint: &str) -> bool {
    match (
        reqwest::Url::parse(endpoint),
        reqwest::Url::parse(VOYAGE_ORIGIN),
    ) {
        (Ok(endpoint), Ok(voyage)) => endpoint.origin() == voyage.origin(),
        _ => false,
    }
}

/// Whether `endpoint` names this machine (a loopback address or `localhost`).
fn is_loopback(endpoint: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(endpoint) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    match host.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V6(ip)) => {
            ip.is_loopback() || ip.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
        Ok(ip) => ip.is_loopback(),
        Err(_) => host.eq_ignore_ascii_case("localhost"),
    }
}

/// The HTTP client settings each provider needs. The local provider never
/// goes through a proxy: its text must not leave the machine, and a proxy set
/// in the environment would otherwise carry loopback requests too.
fn client_for(kind: &ProviderKind, builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
    match kind {
        ProviderKind::Local => builder.no_proxy(),
        ProviderKind::Voyage => builder,
    }
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
    pub price_per_mtok: Option<polylogue_model::Rate>,
    /// Model calls are on: text may be sent to the provider. Off, only
    /// stored vectors rank.
    pub calls: bool,
    credentials: Credentials,
}

/// Why a text without a stored vector cannot be embedded.
pub const CALLS_OFF: &str = "semantic search for new text is off (model calls are off)";

impl Provider {
    /// From settings; `None` while embeddings are off.
    pub fn from_settings(
        settings: &polylogue_model::settings::Snapshot,
        credentials: &Credentials,
    ) -> Result<Option<Self>, String> {
        use polylogue_model::settings::{EmbeddingProvider, Value};
        let kind = match settings
            .get("embed.provider")
            .map_err(|error| error.to_string())?
        {
            Some(Value::EmbeddingProvider(EmbeddingProvider::Off)) => return Ok(None),
            Some(Value::EmbeddingProvider(EmbeddingProvider::Local)) => ProviderKind::Local,
            Some(Value::EmbeddingProvider(EmbeddingProvider::Voyage)) => ProviderKind::Voyage,
            _ => return Err("the typed settings snapshot lacks embed.provider".into()),
        };
        let (endpoint, model) = match kind {
            ProviderKind::Local => (
                "http://127.0.0.1:11434/v1".to_owned(),
                "qwen3-embedding:0.6b",
            ),
            ProviderKind::Voyage => (format!("{VOYAGE_ORIGIN}/v1"), "voyage-4"),
        };
        let endpoint = match settings
            .get("embed.endpoint")
            .map_err(|error| error.to_string())?
        {
            Some(Value::Endpoint(value)) => value.clone(),
            None => endpoint.as_str().into(),
            _ => return Err("invalid typed embed.endpoint".into()),
        };
        if kind == ProviderKind::Local && !is_loopback(&endpoint) {
            return Err(format!(
                "embed.endpoint {endpoint} is not on this machine; the local provider \
                 only sends text to a loopback address"
            ));
        }
        if kind == ProviderKind::Voyage && !is_voyage_origin(&endpoint) {
            return Err(format!(
                "embed.endpoint {endpoint} is not Voyage's origin ({VOYAGE_ORIGIN}); \
                 the Voyage key is only sent to Voyage"
            ));
        }
        let model = match settings
            .get("embed.model")
            .map_err(|error| error.to_string())?
        {
            Some(Value::Text(value)) => value.clone(),
            None => model.into(),
            _ => return Err("invalid typed embed.model".into()),
        };
        let price_per_mtok = match settings
            .get("embed.price_per_mtok")
            .map_err(|error| error.to_string())?
        {
            Some(Value::NonnegativeNumber(value)) => Some(*value),
            None if kind == ProviderKind::Local => Some(polylogue_model::Rate::ZERO),
            None => None,
            _ => return Err("invalid typed embed.price_per_mtok".into()),
        };
        let calls = match settings
            .get("embed.calls")
            .map_err(|error| error.to_string())?
        {
            Some(Value::Boolean(value)) => *value,
            None => false,
            _ => return Err("invalid typed embed.calls".into()),
        };
        Ok(Some(Self {
            endpoint,
            model,
            kind,
            price_per_mtok,
            calls,
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

    /// The embedding key of a query vector with `dims` values.
    pub fn query_key(&self, dims: usize) -> String {
        format!("{}{dims}/query/unit", self.key_prefix())
    }

    /// Embeds `inputs`, returning one unit vector each. Nothing is sent
    /// while model calls are off.
    pub fn embed(&self, inputs: &[String], query: bool) -> Result<Vec<Vec<f32>>, EmbedError> {
        if !self.calls {
            return Err(EmbedError::CallsOff);
        }
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
        let mut client = client_for(&self.kind, reqwest::Client::builder());
        if let Some(timeout) = timeout {
            client = client.timeout(timeout);
        }
        let client = client.build().map_err(failed)?;
        let mut request = client.post(&url).json(&body);
        if self.kind == ProviderKind::Voyage {
            // The key goes to Voyage and nowhere else, whatever endpoint the
            // provider was built with.
            if !is_voyage_origin(&self.endpoint) {
                return Err(EmbedError::Config(format!(
                    "refusing to send the Voyage key to {}; it only goes to {VOYAGE_ORIGIN}",
                    self.endpoint
                )));
            }
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
        if self.kind == ProviderKind::Local && !is_loopback(&self.endpoint) {
            return Err(EmbedError::Config(format!(
                "refusing to send text to {}; the local provider only uses loopback addresses",
                self.endpoint
            )));
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
        batch_vectors(&v, inputs.len())
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
    #[error("{CALLS_OFF}")]
    CallsOff,
}

/// Validate the complete indexed response before any vector reaches storage.
fn batch_vectors(response: &Value, inputs: usize) -> Result<Vec<Vec<f32>>, EmbedError> {
    let invalid = |why: &str| EmbedError::Transport(format!("invalid embedding response: {why}"));
    let data = response
        .array("data")
        .ok_or_else(|| invalid("missing data array"))?;
    if data.len() != inputs {
        return Err(invalid("vector count differs from input count"));
    }
    let mut out = vec![None; inputs];
    let mut dimensions = None;
    for item in data {
        let index = item
            .u64("index")
            .ok_or_else(|| invalid("missing or invalid input index"))?;
        let index = usize::try_from(index).map_err(|_| invalid("input index is out of range"))?;
        let slot = out
            .get_mut(index)
            .ok_or_else(|| invalid("input index is out of range"))?;
        if slot.is_some() {
            return Err(invalid("duplicate input index"));
        }
        let values = item
            .array("embedding")
            .ok_or_else(|| invalid("missing embedding array"))?;
        if dimensions.is_some_and(|dimensions| dimensions != values.len()) {
            return Err(invalid("inconsistent vector dimensions"));
        }
        let mut vector = Vec::with_capacity(values.len());
        for value in values {
            let component = value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid("a component is not a finite number"))?
                as f32;
            if !component.is_finite() {
                return Err(invalid("a component is outside finite f32 range"));
            }
            vector.push(component);
        }
        let vector = unit(vector)?;
        dimensions = Some(vector.len());
        *slot = Some(vector);
    }
    out.into_iter()
        .map(|vector| vector.ok_or_else(|| invalid("missing input index")))
        .collect()
}

fn unit(mut values: Vec<f32>) -> Result<Vec<f32>, EmbedError> {
    // Squaring finite f32 values in f64 preserves tiny and large vectors.
    let norm = values
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum::<f64>()
        .sqrt();
    if values.is_empty() || !norm.is_finite() || norm == 0.0 {
        return Err(EmbedError::Transport(
            "invalid embedding response: vector has no finite nonzero norm".into(),
        ));
    }
    for value in &mut values {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(values)
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

/// What an embedding preview or build answers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Answer {
    Preview(Preview),
    Built(Built),
}

/// A money amount as `money::Amount` writes it; read back to its nano-USD.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct WrittenAmount {
    nusd: i64,
    #[serde(rename = "usd")]
    _usd: serde_json::Number,
}

fn written_amount<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Amount>, D::Error> {
    Ok(Option::<WrittenAmount>::deserialize(d)?.map(|amount| Amount(amount.nusd)))
}

/// What a build would embed, and what it would cost.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Preview {
    pub provider: String,
    pub model: String,
    pub texts: u64,
    pub chunks: u64,
    pub characters: u64,
    /// About four characters per token; the provider counts exactly.
    pub tokens_estimate: u64,
    /// `None`: the price is unknown (set `embed.price_per_mtok`).
    #[serde(deserialize_with = "written_amount")]
    #[schemars(with = "Option<WrittenAmount>")]
    pub cost_estimate: Option<Amount>,
    pub already_embedded: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Built {
    pub key: Option<String>,
    pub texts: u64,
    pub vectors: u64,
    pub failed: u64,
    pub seconds: f64,
    pub texts_per_second: f64,
}

/// Every selected prose page, stopping on a read or provider failure.
fn each_prose_page(
    reader: &Reader<'_>,
    sessions: Option<&[String]>,
    mut f: impl FnMut(Vec<polylogue_store::vectors::ProseText>) -> Result<(), EmbedError>,
) -> Result<(), EmbedError> {
    let mut after = 0;
    loop {
        let page = reader.prose_texts(after, PAGE, sessions)?;
        let Some(last) = page.last() else {
            return Ok(());
        };
        after = last.id;
        f(page)?;
    }
}

fn rejected_for(
    vectors: &Vectors,
    provider: &Provider,
    key: Option<&str>,
) -> Result<HashSet<Vec<u8>>, EmbedError> {
    // A rejection before any vector was returned has no known dimension.
    let mut rejected = vectors.rejected(&provider.key(0))?;
    if let Some(key) = key {
        rejected.extend(vectors.rejected(key)?);
    }
    Ok(rejected)
}

fn missing_for_page(
    vectors: &Vectors,
    provider: &Provider,
    key: Option<&str>,
    rejected: &HashSet<Vec<u8>>,
    page: &[polylogue_store::vectors::ProseText],
) -> Result<HashMap<Vec<u8>, HashSet<u32>>, EmbedError> {
    // Existing text rows have no normalized length metadata. Count before
    // allocating strings, and read purchased identities once for this page.
    let texts = page
        .iter()
        .filter(|text| !rejected.contains(&text.hash))
        .map(|text| {
            (
                text.hash.clone(),
                text.text.nfc().count().div_ceil(CHUNK_CHARS).max(1),
            )
        })
        .collect::<Vec<_>>();
    let key = key.map(str::to_owned).unwrap_or_else(|| provider.key(0));
    Ok(vectors.missing_chunks(&key, &texts)?)
}

fn pending_chunks(text: &str, missing: &HashSet<u32>) -> Result<Vec<(u32, String)>, EmbedError> {
    if missing.is_empty() {
        return Ok(Vec::new());
    }
    chunks(text)
        .into_iter()
        .enumerate()
        .filter_map(|(index, text)| {
            let index = match u32::try_from(index) {
                Ok(index) => index,
                Err(_) => {
                    return Some(Err(EmbedError::Transport(
                        "embedding chunk identity is out of range".into(),
                    )));
                }
            };
            missing.contains(&index).then_some(Ok((index, text)))
        })
        .collect()
}

pub fn preview(
    reader: &Reader<'_>,
    provider: &Provider,
    sessions: Option<&[String]>,
) -> Result<Preview, EmbedError> {
    let vectors = reader.vectors()?;
    let key = vectors.key_with_prefix(&provider.key_prefix())?;
    let rejected = rejected_for(&vectors, provider, key.as_deref())?;
    let mut p = Preview {
        provider: provider.name().into(),
        model: provider.model.clone(),
        ..Preview::default()
    };
    each_prose_page(reader, sessions, |page| {
        let missing = missing_for_page(&vectors, provider, key.as_deref(), &rejected, &page)?;
        for text in page {
            let pending = match missing.get(&text.hash) {
                Some(missing) => pending_chunks(&text.text, missing)?,
                None => Vec::new(),
            };
            if pending.is_empty() {
                p.already_embedded += 1;
                continue;
            }
            p.texts += 1;
            p.characters += pending
                .iter()
                .map(|(_, text)| text.chars().count() as u64)
                .sum::<u64>();
            p.chunks += pending.len() as u64;
        }
        Ok(())
    })?;
    p.tokens_estimate = p.characters.div_ceil(4);
    p.cost_estimate = provider
        .price_per_mtok
        .and_then(|price| purchase_estimate(p.tokens_estimate, price));
    Ok(p)
}

/// The price of `tokens` at `price`: tokens times pico-USD per token, rounded
/// once to nano-USD.
fn purchase_estimate(
    tokens: u64,
    price: polylogue_model::Rate,
) -> Option<polylogue_model::money::Amount> {
    let pico = i128::from(tokens).checked_mul(price.pico_usd_per_token())?;
    let nusd = polylogue_model::money::rounded_quotient(pico, 1_000)?;
    i64::try_from(nusd).ok().map(polylogue_model::money::Amount)
}

pub fn build(
    clock: &polylogue_model::WallClock,
    reader: &Reader<'_>,
    provider: &Provider,
    sessions: Option<&[String]>,
) -> Result<Built, EmbedError> {
    let started = Instant::now();
    let mut vectors = reader.vectors_writer()?;
    let mut key = vectors.key_with_prefix(&provider.key_prefix())?;
    let rejected = rejected_for(&vectors, provider, key.as_deref())?;
    let mut out = Built::default();
    // (text hash, chunk number, chunk text)
    let mut batch: Vec<(Vec<u8>, u32, String)> = Vec::new();
    let flush = |batch: &mut Vec<(Vec<u8>, u32, String)>,
                 out: &mut Built,
                 vectors: &mut Vectors,
                 selected_key: &mut Option<String>|
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
                            let key = selected_key.clone().unwrap_or_else(|| provider.key(0));
                            vectors.fail(&key, hash, &why, clock)?;
                            out.failed += 1;
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        if let Some(first) = stored.first() {
            let key = provider.key(first.values.len());
            if stored
                .iter()
                .any(|vector| vector.values.len() != first.values.len())
            {
                return Err(EmbedError::Transport(
                    "provider returned inconsistent vector dimensions across requests".into(),
                ));
            }
            if selected_key
                .as_ref()
                .is_some_and(|selected| selected != &key)
            {
                return Err(EmbedError::Transport(
                    "provider vector dimensions differ from the selected embedding key".into(),
                ));
            }
            selected_key.get_or_insert_with(|| key.clone());
            out.key.get_or_insert_with(|| key.clone());
            vectors.put(&key, &stored, clock)?;
            out.vectors += stored.len() as u64;
        }
        batch.clear();
        Ok(())
    };
    each_prose_page(reader, sessions, |page| {
        let missing = missing_for_page(&vectors, provider, key.as_deref(), &rejected, &page)?;
        for text in page {
            let pending = match missing.get(&text.hash) {
                Some(missing) => pending_chunks(&text.text, missing)?,
                None => Vec::new(),
            };
            if pending.is_empty() {
                continue;
            }
            for (chunk, input) in pending {
                batch.push((text.hash.clone(), chunk, input));
                if batch.len() >= BATCH {
                    flush(&mut batch, &mut out, &mut vectors, &mut key)?;
                }
            }
            out.texts += 1;
        }
        Ok(())
    })?;
    flush(&mut batch, &mut out, &mut vectors, &mut key)?;
    out.seconds = started.elapsed().as_secs_f64();
    out.texts_per_second = if out.seconds > 0.0 {
        out.texts as f64 / out.seconds
    } else {
        0.0
    };
    Ok(out)
}

/// A `similar:` text as it is embedded and stored: NFC.
pub fn query_text(text: &str) -> String {
    text.nfc().collect()
}

/// For `similar:`: the query text (NFC) as a unit vector, embedded as a query.
pub fn query_vector(provider: &Provider, text: &str) -> Result<Vec<f32>, EmbedError> {
    Ok(provider.embed(&[text.to_owned()], true)?.remove(0))
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

/// Physical session IDs covered by every returned session row, through all
/// pages. Each logical row includes its known structural members; retained
/// rows keep their captured physical membership.
fn query_sessions(
    reader: &Reader<'_>,
    query: &str,
    context: &std::sync::Arc<polylogue_reading::operation::Context>,
) -> Result<Vec<String>, Failure> {
    let mut ids = Vec::new();
    let mut roots = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let ask = crate::daemon::query::Ask {
            text: query,
            cursor: cursor.as_deref(),
            page_size: Some(1_000),
            why: false,
            session: None,
            context: std::sync::Arc::clone(context),
            embeddings: false,
            ranking_identity: None,
            near: None,
            fields: &[],
        };
        let answer = crate::daemon::query::run(reader, ask)?;
        let crate::protocol::query::Answer::Rows(page) = &answer.value else {
            return Err(Failure::bad_request("embed takes session rows"));
        };
        if page.unit != "sessions" || page.shape != polylogue_query::Shape::Rows {
            return Err(Failure::bad_request("embed takes session rows"));
        }
        if answer.outcome == crate::daemon::query::Outcome::Incomplete {
            return Err(Failure::not_yet_available(
                "embedding selection requires complete query results",
            ));
        }
        let references = page
            .references
            .as_ref()
            .filter(|references| references.len() == page.rows.len())
            .ok_or_else(|| {
                Failure::bad_request("embedding selection requires session identities")
            })?;
        if answer.logical_sessions.len() != references.len() {
            return Err(Failure::new(
                polylogue_model::Code::Internal,
                "embedding selection lost session coordinates",
            ));
        }
        for (reference, logical) in references.iter().zip(&answer.logical_sessions) {
            let polylogue_model::Reference::Session(id) = reference else {
                return Err(Failure::bad_request("embed takes session rows"));
            };
            if *logical {
                roots.push(id.to_string());
            } else {
                ids.push(id.to_string());
            }
        }
        match &page.next {
            Some(next) => cursor = Some(next.clone()),
            None => {
                ids.extend(reader.logical_session_members(&roots)?);
                ids.sort_unstable();
                ids.dedup();
                return Ok(ids);
            }
        }
    }
}

/// Embedding cost preview and build ([`crate::embeddings`]).
pub fn handle(
    reader: &polylogue_store::Reader<'_>,
    query: Option<&str>,
    yes: bool,
    context: Context<'_>,
) -> Answered<Answer> {
    let provider = match Provider::from_settings(context.settings, context.credentials) {
        Ok(p) => p,
        Err(e) => return error(polylogue_model::Code::BadRequest, e),
    };
    let Some(provider) = provider else {
        return crate::daemon::Failure::refusal(polylogue_model::Reason::EmbeddingsOff {}).answer();
    };
    // A query limits the work to the sessions it finds.
    let sessions: Option<Vec<String>> = match query.filter(|q| !q.trim().is_empty()) {
        None => None,
        Some(q) => match query_sessions(reader, q, context.operation) {
            Ok(ids) => Some(ids),
            Err(failure) => return failure.answer(),
        },
    };
    if yes && !provider.calls {
        return crate::daemon::Failure::refusal(polylogue_model::Reason::EmbeddingCallsOff {})
            .answer();
    }
    let result = if yes {
        build(context.clock, reader, &provider, sessions.as_deref()).map(Answer::Built)
    } else {
        preview(reader, &provider, sessions.as_deref()).map(Answer::Preview)
    };
    match result {
        Ok(answer) => Answered::ok(answer),
        Err(e) => error(polylogue_model::Code::Embed, e),
    }
}

#[cfg(test)]
mod tests;
