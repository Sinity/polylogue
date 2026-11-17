//! The finite preserved execution transition. No normal spool reader decodes
//! prior execution options and no original frame or body is rewritten.
use crate::{Error, Result};
use serde_json::Value;
use std::io::Write;
use std::path::Path;

/// Extract changed query records through the spool's existing checked body/frame reader.
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
            let records: Vec<Value> = if prepared.material().is_some() {
                let sinex_protocol::Ingress::Material { material, .. } =
                    spool.decode_frame(&entry.identity)?
                else {
                    return Err(Error::invalid("named material has no frame"));
                };
                if material.parser != "polylogue-products" {
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
                            "polylogue.query"
                                | "polylogue.query_run"
                                | "polylogue.query_run_outcome"
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
                        Some("query" | "query_run" | "query_run_outcome")
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

/// Queue the converted products through the normal producer. A new source
/// version cannot reuse the immutable frame bytes of its predecessor.
pub fn prepare_current(products: &Path, sinex: &crate::Sinex) -> Result<u64> {
    use polylogue_store::primary::{Product, ProductWrite};
    use std::io::BufRead;
    let input = std::io::BufReader::new(std::fs::File::open(products)?);
    let mut count = 0;
    for line in input.lines() {
        let product: Product = serde_json::from_str(&line?)?;
        let (name, machine) = match &product {
            Product::Query { object } => (format!("query:{}", object.id), None),
            Product::QueryRun { run } => {
                (format!("query-run:{}", run.id), Some(run.machine.clone()))
            }
            Product::QueryRunOutcome { outcome } => {
                (format!("query-run-outcome:{}", outcome.run), None)
            }
            _ => {
                return Err(Error::invalid(
                    "converted products must contain only query records",
                ));
            }
        };
        sinex.prepare_named_products(
            &name,
            &[ProductWrite {
                product,
                location: None,
                parents: vec![],
                machine,
                output_index: 0,
            }],
        )?;
        count += 1;
    }
    Ok(count)
}

/// Submit queued current products using the existing sender and its refusal
/// handling. Confirmation and cache reconstruction remain the normal consumer.
pub fn restore_current(products: &Path, sinex: &crate::Sinex) -> Result<u64> {
    let count = prepare_current(products, sinex)?;
    // The normal producer first admits observation material, then the events
    // bound to that material. Drive both stages before reporting submission.
    sinex.drain_pending()?;
    sinex.drain_pending()?;
    Ok(count)
}

#[cfg(test)]
mod tests;
