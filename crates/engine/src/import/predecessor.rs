//! `polylogue import --format NAME PATH` for the Python predecessor's own
//! stores ([`polylogue_store::predecessor`]), each named explicitly:
//!
//! - `predecessor-user-db`: its `user.db`. Every row of every table is kept
//!   verbatim as one imported input (`user.db!/rows.jsonl`); its assertions
//!   become assertions with their identities, authors, times and revision
//!   history, all with one `pass` as their origin.
//! - `predecessor-vectors`: one of its vector stores. A vector named by the
//!   address of the request that bought it is stored for every text here
//!   whose embedding request has that address. A vector that matches no
//!   text, and every vector of a store keyed by message (whose embedded text
//!   cannot be proven), stays in its source file and is reported.
//!
//! The source catalog (`predecessor-archive`) is staged as package members
//! by `legacy`. Importing the same store again changes nothing.

use super::older_archive::{ClassCount, RefusedRow, keep, refusal, restored, tally};
use super::*;
use polylogue_store::predecessor::user_db::{self, UserDb};
use polylogue_store::predecessor::vectors::{Layout, VectorStore};
use polylogue_store::writer::Restored;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use unicode_normalization::UnicodeNormalization;

/// The predecessor's source catalog: a folder with `source.db` and `blob/`.
pub const ARCHIVE: &str = "predecessor-archive";
/// The predecessor's `user.db` of authored records.
pub const USER_DB: &str = "predecessor-user-db";
/// One of the predecessor's vector stores.
pub const VECTORS: &str = "predecessor-vectors";

/// The format names that name a predecessor store, not a parser.
pub const STORES: [&str; 3] = [ARCHIVE, USER_DB, VECTORS];

/// Vectors written in one transaction.
const VECTORS_PER_GROUP: usize = 4096;
/// Texts read from the index per page while matching addresses.
const TEXTS_PER_PAGE: u64 = 4096;

/// What became of the rows of one predecessor store.
#[derive(Debug, Default, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct PredecessorReport {
    pub path: PathBuf,
    pub format: String,
    /// Per class of row: `found` is always the sum of the other four.
    pub classes: BTreeMap<String, ClassCount>,
    pub refused: Vec<RefusedRow>,
    /// The imported input keeping every row of a `user.db` verbatim, by its ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows_input: Option<String>,
    /// Vectors the store holds.
    #[serde(default)]
    pub vectors: u64,
    /// Vectors stored here under at least one text key; the outcome of
    /// each key is counted in `classes` under `text key`.
    #[serde(default)]
    pub vectors_matched: u64,
    /// Vectors left in their source file, by reason; with `vectors_matched`
    /// they sum to `vectors`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unmatched: BTreeMap<String, u64>,
    /// The address of every vector of an addressed store that matched no
    /// text here, in hex.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unmatched_addresses: Vec<String>,
}

/// The predecessor's address of an embedding request: SHA-256 over its
/// domain and the model and NFC text, each prefixed by its length as eight
/// big-endian bytes. Only requests of its default shape (input type
/// `document`, 1024 dimensions, no options) were addressed this way.
pub fn address(model: &str, nfc_text: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"polylogue.embedding-input.v1\0");
    for segment in [model.as_bytes(), nfc_text.as_bytes()] {
        hasher.update((segment.len() as u64).to_be_bytes());
        hasher.update(segment);
    }
    hasher.finalize().into()
}

/// The embedding key here of a predecessor vector: the predecessor embedded
/// only through Voyage, as documents, keeping the provider's unit vectors.
fn embedding_key(model: &str, dims: usize) -> Option<String> {
    model
        .starts_with("voyage")
        .then(|| format!("voyage/{model}/{dims}/document/unit"))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

impl PredecessorReport {
    fn tally(&mut self, class: &str, n: u64, outcome: &Restored, row: &str) {
        tally(&mut self.classes, &mut self.refused, class, n, outcome, row);
    }

    fn keep(&mut self, class: &str, n: u64) {
        keep(&mut self.classes, class, n);
    }

    fn unmatched(&mut self, reason: &str, n: u64) {
        self.vectors += n;
        *self.unmatched.entry(reason.to_owned()).or_default() += n;
    }
}

impl Engine {
    /// Imports the predecessor store at `path` named by `format`, one of
    /// [`USER_DB`] and [`VECTORS`].
    pub(super) fn import_predecessor_store(
        &mut self,
        format: &str,
        path: &Path,
        authorship: Option<&polylogue_model::authorship::Authorship>,
    ) -> Result<PredecessorReport, EngineError> {
        let scratch = self
            .archive
            .archive_db()
            .parent()
            .map(|root| root.join("older-archive-copies"))
            .ok_or_else(|| std::io::Error::other("the archive has no folder"))?;
        let mut report = PredecessorReport {
            path: path.to_path_buf(),
            format: format.to_owned(),
            ..Default::default()
        };
        match format {
            USER_DB => self.import_predecessor_user_db(path, &scratch, authorship, &mut report)?,
            VECTORS => self.import_predecessor_vectors(path, &scratch, &mut report)?,
            _ => {
                return Err(std::io::Error::other(format!(
                    "{format} is not a predecessor store this route imports"
                ))
                .into());
            }
        }
        Ok(report)
    }

    fn import_predecessor_user_db(
        &mut self,
        path: &Path,
        scratch: &Path,
        authorship: Option<&polylogue_model::authorship::Authorship>,
        report: &mut PredecessorReport,
    ) -> Result<(), EngineError> {
        let db = UserDb::open(path, scratch)?;
        let (rows_bytes, _) = self.archive.prepare_write(|out| db.write_rows(out))?;
        let rows_id = rows_bytes.hash().to_string();
        report.rows_input = Some(rows_id.clone());
        let folder = path.parent().unwrap_or(Path::new(""));
        let container = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "user.db".into());
        self.store_kept_member(folder, &container, "rows.jsonl", rows_bytes, authorship)?;

        let rows = db.assertions()?;
        let machine = self.config.machine.clone();
        let writer = self.writer.sender()?;
        let pass_id = format!("predecessor-user-db:{rows_id}");
        let pass = polylogue_model::RevisionReference::exact(pass_id.clone(), pass_id.clone());
        if !rows.is_empty() {
            let outcome = match user_db::pass(&pass_id, &rows, &machine) {
                Ok(fact) => match writer.restore_assertion(fact) {
                    Ok(outcome) => restored(&outcome),
                    Err(error) => refusal(error)?,
                },
                Err(reason) => Restored::Refused(reason),
            };
            if let Restored::Refused(reason) = &outcome {
                return Err(std::io::Error::other(format!(
                    "the pass grouping the predecessor's assertions was refused: {reason}"
                ))
                .into());
            }
            report.tally("pass", 1, &outcome, &pass_id);
        }
        for (id, mapped) in user_db::facts(&rows, &machine, &pass, &db.result_set_members()?) {
            let outcome = match mapped {
                Ok(fact) => match writer.restore_assertion(fact) {
                    Ok(outcome) => restored(&outcome),
                    Err(error) => refusal(error)?,
                },
                Err(reason) => Restored::Refused(reason),
            };
            report.tally("assertions", 1, &outcome, &id);
        }
        // Rows with no counterpart here are kept verbatim in the rows input:
        // annotation schemas (a rubric here needs a unit query in this query
        // language), the predecessor's query plans, result sets and receipts.
        for (table, n) in db.row_counts()? {
            if table != "assertions" && n > 0 {
                report.keep(&table, n);
            }
        }
        Ok(())
    }

    fn import_predecessor_vectors(
        &mut self,
        path: &Path,
        scratch: &Path,
        report: &mut PredecessorReport,
    ) -> Result<(), EngineError> {
        let store = VectorStore::open(path, scratch)?;
        if store.layout() == Layout::ByMessage {
            for n in store.by_message_counts()?.into_values() {
                report.unmatched(
                    "keyed by the predecessor's message identity; the embedded text cannot be proven",
                    n,
                );
            }
            return Ok(());
        }
        let identities = store.addressed()?;
        let Some(dims) = identities.first().map(|identity| identity.dims) else {
            return Ok(());
        };
        if identities.iter().any(|identity| identity.dims != dims) {
            return Err(std::io::Error::other(
                "the vector store mixes dimensions; each must be imported by its own dimension",
            )
            .into());
        }
        let mut keys = BTreeMap::new();
        for identity in &identities {
            if !keys.contains_key(&identity.model) {
                let key = embedding_key(&identity.model, dims).ok_or_else(|| {
                    std::io::Error::other(format!(
                        "model {} has no known provider here",
                        identity.model
                    ))
                })?;
                keys.insert(identity.model.clone(), key);
            }
        }
        let by_address: HashMap<[u8; 32], &polylogue_store::predecessor::vectors::Addressed> =
            identities
                .iter()
                .map(|identity| (identity.address, identity))
                .collect();

        // Every prose text here, by the address its embedding request would
        // have had under each model the store holds.
        let models: Vec<&String> = keys.keys().collect();
        let mut texts: HashMap<[u8; 32], Vec<(Vec<u8>, bool)>> = HashMap::new();
        {
            let reader = self.reads.reader()?;
            let mut after = 0;
            loop {
                let page = reader.prose_texts(after, TEXTS_PER_PAGE, None)?;
                let Some(last) = page.last() else { break };
                after = last.id;
                for text in page {
                    let nfc: String = text.text.nfc().collect();
                    let chunked = nfc.chars().count() > polylogue_store::vectors::CHUNK_CHARS;
                    for model in &models {
                        let address = address(model, &nfc);
                        if by_address.contains_key(&address) {
                            texts
                                .entry(address)
                                .or_default()
                                .push((text.hash.clone(), chunked));
                        }
                    }
                }
            }
        }

        let reader = self.reads.reader()?;
        let mut vectors = reader.vectors_writer()?;
        let mut batch: Vec<polylogue_store::older_archive::StoredVector> = Vec::new();
        let mut seen = BTreeSet::new();
        let mut flush = |batch: &mut Vec<polylogue_store::older_archive::StoredVector>,
                         report: &mut PredecessorReport|
         -> Result<(), polylogue_store::StoreError> {
            if batch.is_empty() {
                return Ok(());
            }
            let outcomes = vectors.restore(batch)?;
            for (vector, outcome) in batch.iter().zip(&outcomes) {
                report.tally("text key", 1, outcome, &hex(&vector.text_hash));
            }
            batch.clear();
            Ok(())
        };
        store.for_each(dims, |id, values| {
            let Some(address) = parse_address(id) else {
                return Err(polylogue_store::StoreError::Invalid(format!(
                    "vector {id} is not named by a 32-byte address"
                )));
            };
            let Some(identity) = by_address.get(&address) else {
                return Err(polylogue_store::StoreError::Invalid(format!(
                    "vector {id} has no metadata row"
                )));
            };
            seen.insert(address);
            let Some(embedded_at) = identity.embedded_at else {
                report.unmatched("its purchase time is not recorded", 1);
                report.unmatched_addresses.push(id.to_owned());
                return Ok(());
            };
            let matches = texts.get(&address).map(Vec::as_slice).unwrap_or_default();
            let whole: Vec<&Vec<u8>> = matches
                .iter()
                .filter(|(_, chunked)| !chunked)
                .map(|(hash, _)| hash)
                .collect();
            if whole.is_empty() {
                report.unmatched(
                    if matches.is_empty() {
                        "no text here has this request's address"
                    } else {
                        "its text is longer than one chunk here, so it is embedded in parts"
                    },
                    1,
                );
                report.unmatched_addresses.push(id.to_owned());
                return Ok(());
            }
            report.vectors += 1;
            report.vectors_matched += 1;
            for hash in whole {
                batch.push(polylogue_store::older_archive::StoredVector {
                    text_hash: hash.clone(),
                    model: keys[&identity.model].clone(),
                    chunk: 0,
                    dims: dims as i64,
                    vector: values.to_vec(),
                    embedded_at,
                });
            }
            if batch.len() >= VECTORS_PER_GROUP {
                flush(&mut batch, report)?;
            }
            Ok(())
        })?;
        flush(&mut batch, report)?;
        for identity in &identities {
            if !seen.contains(&identity.address) {
                report.unmatched("named in the store's metadata with no stored values", 1);
                report.unmatched_addresses.push(hex(&identity.address));
            }
        }
        Ok(())
    }
}

fn parse_address(id: &str) -> Option<[u8; 32]> {
    if id.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(id.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod tests;
