//! The finite preserved execution transition. No normal spool reader decodes
//! prior execution options and no original frame or body is rewritten.
use crate::{Error, Result};
use serde_json::Value;
use std::io::Write;
use std::path::Path;

/// Extract query records through the spool's existing checked body/frame reader.
/// The authoritative queue is opened read-only; source seals and bytes survive.
pub fn extract(store: &Path, producer: &str, output: &mut dyn Write) -> Result<u64> {
    let spool = crate::spool::Spool::preserved(store, producer)?;
    let mut after = None;
    let mut count = 0;
    loop {
        let entries = spool.pending_after(after, 128)?;
        if entries.is_empty() {
            break;
        }
        for entry in entries {
            after = Some(entry.order);
            let prepared = &entry;
            if !prepared.is_named() {
                continue;
            }
            let records: Vec<Value> = if prepared.material().is_some() {
                let sinex_protocol::Ingress::Material { material, .. } =
                    spool.decode_frame(&entry.identity)?
                else {
                    return Err(Error::invalid("named material has no frame"));
                };
                if !matches!(
                    material.parser.as_str(),
                    "polylogue-products" | "polylogue-request-restore"
                ) {
                    continue;
                }
                serde_json::from_reader(spool.body(&entry.identity)?)?
            } else {
                let sinex_protocol::Ingress::Events { events, .. } =
                    spool.decode_frame(&entry.identity)?
                else {
                    continue;
                };
                events
                    .into_iter()
                    .filter(|event| {
                        matches!(
                            event.kind.name.as_str(),
                            "polylogue.query_run" | "polylogue.query_run_outcome"
                        )
                    })
                    .map(|event| serde_json::json!({"product":event.payload["record"]}))
                    .collect()
            };
            let records: Vec<_> = records
                .into_iter()
                .filter(|write| {
                    matches!(
                        write["product"]["product"].as_str(),
                        Some("query_run" | "query_run_outcome")
                    )
                })
                .collect();
            if !records.is_empty() {
                count += records.len() as u64;
                serde_json::to_writer(&mut *output, &records)?;
                output.write_all(b"\n")?;
            }
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests;
