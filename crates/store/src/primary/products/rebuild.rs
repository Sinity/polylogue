//! One inverse of canonical products serves replay and private pending custody.
use super::*;
use polylogue_model::Output;
use std::collections::BTreeMap;

#[derive(Default)]
pub(crate) struct AssembledReading {
    pub output: Output,
    pub counts: RecordCounts,
    pub receipt: Option<crate::wire_shapes::ReadingReceipt>,
    pub reset: bool,
    pub comparison: Option<crate::tool_databases::Comparison>,
    pub checkpoint: Option<polylogue_model::batch::ParserCheckpoint>,
    pub stopped_attempt: Option<polylogue_model::batch::StoppedAttempt>,
    pub semantic_event_ids: Option<BTreeMap<usize, String>>,
    controls: usize,
}
impl AssembledReading {
    pub(crate) fn push(
        &mut self,
        product: Product,
        admitted: Option<u64>,
        receipt: Option<crate::wire_shapes::ReadingReceipt>,
    ) -> crate::Result<()> {
        match product {
            Product::Session { facts, .. } => {
                save_receipt(
                    &mut self.output.session_receipts,
                    self.output.sessions.len(),
                    admitted,
                );
                self.output.sessions.push(facts);
            }
            Product::Message {
                facts,
                order,
                projection,
                native_fields,
                declared_injections,
                declared_attachment_bindings,
                ..
            } => {
                let ordinal = self.output.messages.len();
                if let Some(native_fields) = native_fields {
                    self.output
                        .native_hook_fields
                        .insert(ordinal, native_fields);
                }
                if let Some(declared_injections) = declared_injections {
                    self.output
                        .declared_injections
                        .insert(ordinal, declared_injections);
                }
                if let Some(declared_attachment_bindings) = declared_attachment_bindings {
                    self.output
                        .declared_attachment_bindings
                        .insert(ordinal, declared_attachment_bindings);
                }
                if let Some(projection) = projection {
                    self.output
                        .message_projections
                        .insert(self.output.messages.len(), projection);
                }
                if let Some(order) = order {
                    self.output
                        .message_order
                        .insert(self.output.messages.len(), order);
                }
                save_receipt(
                    &mut self.output.message_receipts,
                    self.output.messages.len(),
                    admitted,
                );
                self.output.messages.push(*facts);
            }
            Product::Document { facts, .. } => {
                save_receipt(
                    &mut self.output.document_receipts,
                    self.output.documents.len(),
                    admitted,
                );
                self.output.documents.push(facts);
            }
            Product::Fragment { facts, .. } => {
                save_receipt(
                    &mut self.output.fragment_receipts,
                    self.output.fragments.len(),
                    admitted,
                );
                self.output.fragments.push(*facts);
            }
            Product::Event {
                id,
                facts,
                imported,
                ..
            } => {
                save_receipt(
                    &mut self.output.event_receipts,
                    self.output.events.len(),
                    admitted,
                );
                self.output.event_ids.insert(self.output.events.len(), id);
                if let Some(imported) = imported {
                    self.output
                        .imported_events
                        .insert(self.output.events.len(), imported);
                }
                self.output.events.push(facts);
            }
            Product::Link { facts, .. } => {
                save_receipt(
                    &mut self.output.link_receipts,
                    self.output.links.len(),
                    admitted,
                );
                self.output.links.push(facts);
            }
            Product::Background { facts, .. } => self.output.background.push(facts),
            Product::Repository { facts, .. } => self.output.repo_lookups.push(facts),
            Product::ParseProgress {
                counts: record_counts,
                reset,
                comparison: source_comparison,
                checkpoint,
                stopped_attempt,
                semantic_event_ids,
                ..
            } => {
                self.semantic_event_ids = semantic_event_ids;
                self.counts = record_counts;
                self.reset = reset;
                self.comparison = source_comparison;
                self.checkpoint = checkpoint;
                self.stopped_attempt = stopped_attempt;
                self.receipt = receipt;
                self.controls += 1;
            }
            _ => unreachable!(),
        }
        Ok(())
    }
}
fn save_receipt(map: &mut BTreeMap<usize, u64>, index: usize, admitted: Option<u64>) {
    if let Some(admitted) = admitted {
        map.insert(index, admitted);
    }
}
impl CapturedReading {
    /// Reconstructs one complete original callback from actual retained products.
    /// Receiver receipts are absent; a pending body cannot supply them.
    pub fn from_product_group<'a>(
        key: &ExportReadingKey,
        products: impl IntoIterator<Item = &'a Product>,
    ) -> crate::Result<Self> {
        key.validate()?;
        let mut reading: Option<Reading> = None;
        let mut body = AssembledReading::default();
        for product in products {
            let next = product.reading().ok_or_else(|| {
                crate::StoreError::Invalid(
                    "pending export body contains an unlocated product".into(),
                )
            })?;
            if next.delivery.is_none()
                || !key.matches_reading(next)
                || reading.as_ref().is_some_and(|prior| prior != next)
            {
                return Err(crate::StoreError::Invalid(
                    "pending export body does not contain one bound callback group".into(),
                ));
            }
            reading.get_or_insert_with(|| next.clone());
            body.push(product.clone(), None, None)?;
        }
        let reading = reading
            .ok_or_else(|| crate::StoreError::Invalid("pending export callback is empty".into()))?;
        if body.controls != 1 {
            return Err(crate::StoreError::Invalid(
                "pending export callback needs exactly one retained control".into(),
            ));
        }
        body.output.event_ids = body.semantic_event_ids.ok_or_else(|| {
            crate::StoreError::Invalid(
                "pending export control omitted original semantic event identities".into(),
            )
        })?;
        let captured = Self {
            export_key: Some(key.clone()),
            delivery: reading.delivery,
            output: body.output,
            counts: body.counts,
            start: reading.start,
            parsed_upto: reading.parsed_upto,
            progress: reading.progress,
            order_from: reading.order_from,
            reading_version: reading.reading_version,
            repeats_whole_conversations: reading.repeats_whole_conversations,
            reset: body.reset,
            comparison: body.comparison,
            checkpoint: body.checkpoint,
            stopped_attempt: body.stopped_attempt,
        };
        captured.clone().into_step().validate_export()?;
        Ok(captured)
    }
}
