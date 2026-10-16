//! A reference projection over retained prompt snapshots. It deliberately has
//! no canonical Session/Message publication or parser settlement certificate.
use polylogue_store::{Error, ReadHandle};
use serde_json::{Value, json};
use std::time::Instant;
pub fn pending(reader: &ReadHandle, until: Instant) -> Result<Value, Error> {
    let mut references = Vec::new();
    let mut snapshots = Vec::new();
    let mut total = 0;
    for source in reader.codex_prompt_sources()? {
        if Instant::now() >= until {
            return Err(Error::Cancelled);
        }
        total += source.input.length;
        if total > 4 * 1024 * 1024 {
            return Err(Error::ParserAdmission {
                resource: "prompt reference retained bytes",
                required: total,
            });
        }
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
                snapshots.push(json!({"input":source.input.id,"chain":source.input.chain,"binding":source.binding,"length":source.input.length,"availability":"unavailable","reason":error.to_string(),"conversation_coverage":null}));
                continue;
            }
        };
        let projection = polylogue_sources::codex::prompt_history::project(&bytes)
            .map_err(|s| Error::Invariant(s.into()))?;
        snapshots.push(json!({"input":source.input.id,"chain":source.input.chain,"binding":source.binding,"length":source.input.length,"availability":"accessible","malformed_records":projection.malformed_records,"incomplete_tail":projection.incomplete_tail}));
        for reference in projection.references {
            if references.len() >= 4096 {
                return Err(Error::ParserAdmission {
                    resource: "pending prompt references",
                    required: references.len() as u64 + 1,
                });
            }
            let id = format!("codex:{}", reference.native_session);
            let identity = polylogue_model::SessionId::from_stored(id.clone())
                .map_err(|e| Error::Invariant(e.to_string()))?;
            let state = if reader.selected_session(&identity, 65536)?.is_some() {
                "conversation_evidence"
            } else {
                "pending"
            };
            // A discovered conversation remains independently incomplete. A
            // prompt reference itself is never another canonical Session row.
            references.push(json!({"session":id,"state":state,"conversation_coverage":null,"assistant_response":null,"source":{"input":source.input.id,"format":"codex-prompt-history","chain":source.input.chain,"binding":source.binding,"provenance":provenance,"start":reference.start,"end":reference.end,"raw_hash":blake3::hash(&bytes[reference.start as usize..reference.end as usize]).to_hex().to_string()},"reason":"prompt reference alone does not prove acquired conversation"}));
        }
    }
    let value = json!({"recipe":"codex-prompt-references-v1","conversation_coverage":null,"references":references,"snapshots":snapshots});
    let required = serde_json::to_vec(&value)
        .map_err(|e| Error::Invariant(e.to_string()))?
        .len();
    if required > 16384 {
        return Err(Error::ParserAdmission {
            resource: "prompt reference response metadata",
            required: required as u64,
        });
    }
    Ok(value)
}
