//! The original parser owns values; native state selects their local contribution.
use super::*;
use crate::primary::SourceRecord;
use crate::primary::products::allowed;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Snapshot {
    reading: crate::primary::Reading,
    records: Vec<(SourceRecord, bool)>,
}

pub(super) fn validate(index: &Connection, step: &ParseStep) -> Result<()> {
    let snapshot = step.source_eligibility.as_ref().ok_or_else(|| {
        StoreError::Invalid("a primary reading has no original eligibility snapshot".into())
    })?;
    for (record, keep) in &snapshot.records {
        if allowed(index, &snapshot.reading, record)? != *keep {
            return Err(StoreError::PrimaryPending {
                identity: snapshot.reading.input.id.clone(),
            });
        }
    }
    Ok(())
}

fn retain<T>(values: &mut Vec<T>, mask: &[bool]) {
    let mut index = 0;
    values.retain(|_| {
        let keep = mask[index];
        index += 1;
        keep
    });
}
fn reindex<T>(values: &mut std::collections::BTreeMap<usize, T>, mask: &[bool]) {
    let mut positions = vec![None; mask.len()];
    let mut next = 0;
    for (index, keep) in mask.iter().copied().enumerate() {
        if keep {
            positions[index] = Some(next);
            next += 1;
        }
    }
    *values = std::mem::take(values)
        .into_iter()
        .filter_map(|(index, value)| {
            positions
                .get(index)
                .copied()
                .flatten()
                .map(|index| (index, value))
        })
        .collect();
}

/// Keep the parser's prepared values and stable event identities; only native
/// eligibility decides which source contributions reach the ordinary writer.
pub(super) fn retain_source_records(
    index: &Connection,
    input: &InputKey,
    chain: i64,
    length: u64,
    step: &mut crate::ParseStep,
) -> Result<()> {
    let products = crate::primary::products::parse(input, chain, Some(length), step)?;
    let mut masks = std::collections::BTreeMap::<String, Vec<bool>>::new();
    let reading = crate::primary::Reading::from_step(input, chain, Some(length), step);
    let mut records = Vec::with_capacity(products.len());
    step.preserve_export_body()?;
    for product in products {
        let reading = product.reading().expect("parser fact has source");
        let record = SourceRecord::of(&product)?.expect("parser fact has identity");
        let keep = allowed(index, reading, &record)?;
        // The message owns its local tool blocks. Native parser diagnostics
        // do not own the original parser's local checkpoint or completeness.
        if matches!(record.family.as_str(), "tool_call" | "parse_progress") {
            continue;
        }
        masks.entry(record.family.clone()).or_default().push(keep);
        records.push((record, keep));
    }
    step.source_eligibility = Some(Snapshot { reading, records });
    let all = masks.values().all(|mask| mask.iter().all(|keep| *keep));
    if all {
        return Ok(());
    }
    // Intentional native suppression changes visible contributions, not the
    // bytes the original parser processed. Its missing values cannot supply
    // this receiver's resume context.
    step.checkpoint = None;
    let event_ids = step.event_identities(&input.id, chain)?;
    let output = &mut step.output;
    let mask = |family: &str| masks.get(family).map(Vec::as_slice).unwrap_or(&[]);
    retain(&mut output.sessions, mask("session"));
    reindex(&mut output.session_receipts, mask("session"));
    let messages = mask("message");
    retain(&mut output.messages, messages);
    retain(&mut step.prepared.messages, messages);
    reindex(&mut output.message_order, messages);
    reindex(&mut output.message_projections, messages);
    reindex(&mut output.message_receipts, messages);
    reindex(&mut output.native_hook_fields, messages);
    reindex(&mut output.declared_injections, messages);
    reindex(&mut output.declared_attachment_bindings, messages);
    let documents = mask("document");
    retain(&mut output.documents, documents);
    retain(&mut step.prepared.documents, documents);
    reindex(&mut output.document_receipts, documents);
    let fragments = mask("fragment");
    retain(&mut output.fragments, fragments);
    step.prepared.fragments.retain(fragments)?;
    reindex(&mut output.fragment_receipts, fragments);
    let events = mask("event");
    retain(&mut output.events, events);
    retain(&mut step.prepared.events, events);
    output.event_ids = event_ids.into_iter().enumerate().collect();
    reindex(&mut output.event_ids, events);
    reindex(&mut output.event_receipts, events);
    reindex(&mut output.imported_events, events);
    let links = mask("link");
    retain(&mut output.links, links);
    reindex(&mut output.link_receipts, links);
    let copies = mask("prefix_copy");
    retain(&mut output.prefix_copies, copies);
    reindex(&mut output.prefix_copy_receipts, copies);
    let members = mask("prefix_member");
    retain(&mut output.prefix_members, members);
    reindex(&mut output.prefix_member_receipts, members);
    retain(&mut output.background, mask("background"));
    retain(&mut output.repo_lookups, mask("repository"));
    step.prepared.validate(output)
}
