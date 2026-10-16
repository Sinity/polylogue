//! A bounded reference projection over retained prompt snapshots. Presentation
//! admission never certifies conversation coverage or poisons independent reads.
use polylogue_store::{Error, ReadHandle};
use serde_json::{Value, json};
use std::time::{Duration, Instant};
const REFERENCE_BYTES: usize = 8192;
const SNAPSHOT_BYTES: usize = 4096;
fn encoded(value: &Value) -> usize {
    serde_json::to_vec(value)
        .map(|v| v.len())
        .unwrap_or(usize::MAX)
}
pub fn pending(
    reader: &ReadHandle,
    until: Instant,
    selection: Option<&str>,
) -> Result<Value, Error> {
    let mut sources = reader.codex_prompt_sources()?;
    let catalog_complete = sources.len() <= 512;
    sources.truncate(512);
    let admitted_snapshots = catalog_complete.then_some(sources.len());
    let mut references = Vec::new();
    let mut snapshots = Vec::new();
    let mut reference_bytes = 0;
    let mut snapshot_bytes = 0;
    let mut raw_bytes = 0;
    let mut observed = 0u64;
    let mut matching = 0u64;
    let mut examined = 0;
    let mut malformed = 0;
    let mut incomplete_tails = 0;
    let mut unavailable = 0;
    let mut scan_complete = catalog_complete;
    let scan_until = until.min(Instant::now() + Duration::from_secs(5));
    for source in sources {
        if Instant::now() >= scan_until || raw_bytes + source.input.length > 4 * 1024 * 1024 {
            scan_complete = false;
            break;
        }
        raw_bytes += source.input.length;
        examined += 1;
        let provenance = reader.codex_prompt_provenance(&source.input.id)?;
        let bytes = match reader.read_source_range(
            &source.input.id,
            source.input.chain,
            source.binding,
            0,
            source.input.length,
        ) {
            Ok(bytes) => bytes,
            Err(Error::ExcisionCheckRequired) => return Err(Error::ExcisionCheckRequired),
            Err(error) => {
                unavailable += 1;
                scan_complete = false;
                let value = json!({"input":source.input.id,"availability":"unavailable","reason":error.to_string(),"conversation_coverage":null});
                let size = encoded(&value);
                if snapshot_bytes + size <= SNAPSHOT_BYTES {
                    snapshot_bytes += size;
                    snapshots.push(value);
                }
                continue;
            }
        };
        let projection = match polylogue_sources::codex::prompt_history::project(&bytes) {
            Ok(p) => p,
            Err(reason) => {
                unavailable += 1;
                scan_complete = false;
                let value = json!({"input":source.input.id,"availability":"unavailable","reason":reason,"conversation_coverage":null});
                let size = encoded(&value);
                if snapshot_bytes + size <= SNAPSHOT_BYTES {
                    snapshot_bytes += size;
                    snapshots.push(value);
                }
                continue;
            }
        };
        malformed += projection.malformed_records;
        incomplete_tails += u64::from(projection.incomplete_tail);
        let snapshot = json!({"input":source.input.id,"chain":source.input.chain,"binding":source.binding,"length":source.input.length,"availability":"accessible","malformed_records":projection.malformed_records,"incomplete_tail":projection.incomplete_tail});
        let size = encoded(&snapshot);
        if snapshot_bytes + size <= SNAPSHOT_BYTES {
            snapshot_bytes += size;
            snapshots.push(snapshot);
        }
        for reference in projection.references {
            observed += 1;
            let id = format!("codex:{}", reference.native_session);
            if selection.is_some_and(|selected| selected != id) {
                continue;
            }
            matching += 1;
            if reference_bytes >= REFERENCE_BYTES {
                continue;
            }
            let identity = polylogue_model::SessionId::from_stored(id.clone())
                .map_err(|e| Error::Invariant(e.to_string()))?;
            let state = if reader.selected_session(&identity, 65536)?.is_some() {
                "conversation_evidence"
            } else {
                "pending"
            };
            let value = json!({"session":id,"state":state,"conversation_coverage":null,"assistant_response":null,"source":{"input":source.input.id,"format":"codex-prompt-history","chain":source.input.chain,"binding":source.binding,"provenance":provenance,"start":reference.start,"end":reference.end,"raw_hash":blake3::hash(&bytes[reference.start as usize..reference.end as usize]).to_hex().to_string()},"reason":"prompt reference alone does not prove acquired conversation"});
            let size = encoded(&value);
            if reference_bytes + size <= REFERENCE_BYTES {
                reference_bytes += size;
                references.push(value);
            }
        }
    }
    let shown = references.len();
    let snapshot_details = snapshots.len();
    Ok(
        json!({"recipe":"codex-prompt-references-v1","conversation_coverage":null,"references":references,"snapshots":snapshots,
        "selection":selection,"scan_complete":scan_complete,"reference_observations":scan_complete.then_some(observed),"observed_reference_prefix":observed,
        "matching_reference_observations":scan_complete.then_some(matching),"references_shown":shown,"references_omitted":scan_complete.then_some(matching-shown as u64),
        "retained_prompt_snapshots":admitted_snapshots,"examined_snapshots":examined,"snapshot_details_shown":snapshot_details,
        "snapshot_details_omitted":admitted_snapshots.map(|count|count-snapshot_details),"malformed_record_observations":malformed,"incomplete_tail_observations":incomplete_tails,"unavailable_snapshot_observations":unavailable,
        "presentation_complete":scan_complete && shown as u64==matching && snapshot_details==examined,
        "detail_route":"Use an exact Session read for matching pending-reference details. Retained input cat reads admitted snapshot bytes; omitted metadata is not proof of absent conversation."}),
    )
}
