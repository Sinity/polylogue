//! `polylogue import --origin python-vectors PATH`: one of Python
//! Polylogue's vector stores ([`polylogue_store::python_polylogue`]). A
//! vector named by the address of the request that bought it is stored for
//! every text here whose embedding request has that address. A vector that
//! matches no text, and every vector of a store keyed by message (whose
//! embedded text cannot be checked), stays in its source file and is
//! reported. Importing the same store again changes nothing.

use super::*;
use polylogue_model::hex;
pub use polylogue_store::python_polylogue::vectors::request_address as address;
use polylogue_store::python_polylogue::vectors::{Layout, VectorStore};
use polylogue_store::vectors::{Restored, StoredVector};
use std::collections::{BTreeSet, HashMap};
use unicode_normalization::UnicodeNormalization;

/// One of Python Polylogue's vector stores.
pub const VECTORS: &str = "python-vectors";

/// Vectors written in one transaction.
const VECTORS_PER_GROUP: usize = 4096;
/// Texts read from the index per page while matching addresses.
const TEXTS_PER_PAGE: u64 = 4096;

/// What became of the rows of one class. `found` is always the sum of the
/// other three.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct ClassCount {
    /// Rows the store holds.
    pub found: u64,
    /// Written by this import.
    pub stored: u64,
    /// Already here unchanged, from an earlier import.
    pub present: u64,
    /// Not carried; each is listed in the report's `refused` with its reason.
    pub refused: u64,
}

/// One row the import did not carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct RefusedRow {
    pub class: String,
    /// The row's identity in the store.
    pub row: String,
    pub reason: String,
}

/// What became of the vectors of one Python Polylogue vector store.
#[derive(Debug, Default, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct PythonImportReport {
    pub path: PathBuf,
    pub format: String,
    /// Per class of row: `found` is the sum of its reported outcomes.
    pub classes: BTreeMap<String, ClassCount>,
    pub refused: Vec<RefusedRow>,
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

/// The embedding key here of a Python Polylogue vector: Python Polylogue embedded
/// only through Voyage, as documents, keeping the provider's unit vectors.
fn embedding_key(model: &str, dims: usize) -> Option<String> {
    model
        .starts_with("voyage")
        .then(|| format!("voyage/{model}/{dims}/document/unit"))
}

impl PythonImportReport {
    fn tally(&mut self, class: &str, outcome: &Restored, row: &str) {
        let count = self.classes.entry(class.to_owned()).or_default();
        count.found += 1;
        match outcome {
            Restored::Stored => count.stored += 1,
            Restored::Present => count.present += 1,
            Restored::Refused(reason) => {
                count.refused += 1;
                self.refused.push(RefusedRow {
                    class: class.to_owned(),
                    row: row.to_owned(),
                    reason: reason.clone(),
                });
            }
        }
    }

    fn unmatched(&mut self, reason: &str, n: u64) {
        self.vectors += n;
        *self.unmatched.entry(reason.to_owned()).or_default() += n;
    }
}

impl Engine {
    /// Imports the vector store at `path`.
    pub(super) fn import_python_vectors(
        &mut self,
        path: &Path,
    ) -> Result<PythonImportReport, EngineError> {
        let scratch = self
            .archive
            .archive_db()
            .parent()
            .map(|root| root.join("vector-store-copies"))
            .ok_or_else(|| std::io::Error::other("the archive has no folder"))?;
        let mut report = PythonImportReport {
            path: path.to_path_buf(),
            format: VECTORS.to_owned(),
            ..Default::default()
        };
        self.read_vectors(path, &scratch, &mut report)?;
        Ok(report)
    }

    fn read_vectors(
        &mut self,
        path: &Path,
        scratch: &Path,
        report: &mut PythonImportReport,
    ) -> Result<(), EngineError> {
        let store = VectorStore::open(path, scratch)?;
        if store.layout() == Layout::ByMessage {
            for n in store.by_message_counts()?.into_values() {
                report.unmatched(
                    "keyed by Python Polylogue's message identity; the embedded text cannot be checked",
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
        let by_address: HashMap<[u8; 32], &polylogue_store::python_polylogue::vectors::Addressed> =
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
            let prepared = identities
                .iter()
                .all(|identity| identity.normalized_hash.is_some());
            let lookups: Vec<_> = if prepared {
                identities.iter().map(Some).collect()
            } else {
                vec![None]
            };
            for lookup in lookups {
                let mut after = 0;
                loop {
                    let page = match lookup {
                        Some(identity) => reader.prose_texts_by_normalized_hash(
                            &identity
                                .normalized_hash
                                .expect("validated prepared request"),
                            after,
                            TEXTS_PER_PAGE,
                        )?,
                        None => reader.prose_texts(after, TEXTS_PER_PAGE, None)?,
                    };
                    let Some(last) = page.last() else { break };
                    after = last.id;
                    for text in page {
                        let nfc: String = text.text.nfc().collect();
                        let chunked = nfc.chars().count() > polylogue_store::vectors::CHUNK_CHARS;
                        for model in &models {
                            let address = address(model, &nfc);
                            if lookup.is_some_and(|identity| identity.address != address) {
                                continue;
                            }
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
        }

        let reader = self.reads.reader()?;
        let mut vectors = reader.vectors_writer()?;
        let mut batch: Vec<StoredVector> = Vec::new();
        let mut seen = BTreeSet::new();
        let mut flush = |batch: &mut Vec<StoredVector>,
                         report: &mut PythonImportReport|
         -> Result<(), polylogue_store::StoreError> {
            if batch.is_empty() {
                return Ok(());
            }
            let outcomes = vectors.restore(batch)?;
            for (vector, outcome) in batch.iter().zip(&outcomes) {
                report.tally("text key", outcome, &hex(&vector.text_hash));
            }
            batch.clear();
            Ok(())
        };
        store.for_each(dims, |id, values| {
            let Some(address) = parse_address(id) else {
                return Err(polylogue_store::StoreError::stored_data(format!(
                    "vector {id} is not named by a 32-byte address"
                )));
            };
            let Some(identity) = by_address.get(&address) else {
                return Err(polylogue_store::StoreError::stored_data(format!(
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
                batch.push(StoredVector {
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
