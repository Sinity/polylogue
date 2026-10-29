//! One-record projection and transfer over a caller-owned pinned selection.
use super::super::{Response, error};
use polylogue_model::bundle::{Counts, Record};
use polylogue_model::read::*;
use std::io::Write;

/// Canonical rows are kept separate from layout-only omissions. A mixed query
/// can report unsupported targets without exporting their hidden source bodies.
pub(super) fn project(
    record: Record,
    options: &ReadOptions,
    evidence: &ProjectionEvidence,
    mixed: bool,
    footer: &mut ReadFooter,
) -> Result<Option<ReadRecord>, Response> {
    let reference = record.reference().unwrap_or_default().to_owned();
    match project_record(record, options, evidence) {
        Ok(Some(record)) => {
            let parts = polylogue_render::layout_record(&record, options, evidence)
                .map_err(|e| error("read", e))?;
            Ok(Some(ReadRecord::Canonical { record: Box::new(record), parts }))
        }
        Ok(None) => Ok(None),
        Err(ReadError::UnsupportedTarget { view, target, accepted_views })
            if mixed && target.starts_with("document:") =>
        {
            if let Some(coverage) = footer.coverage.iter_mut().find(|coverage| {
                coverage.target == TargetKind::Document && coverage.view == view
            }) {
                coverage.omitted += 1;
            } else {
                footer.coverage.push(Coverage {
                    target: TargetKind::Document,
                    view,
                    omitted: 1,
                    accepted_views,
                });
            }
            Ok(Some(ReadRecord::Presentation {
                parts: vec![ReadPart::Item(ReadItem {
                    source: Source::record(reference),
                    kind: ItemKind::Omission {
                        count: ValueState::known(1),
                        unit: "document".into(),
                        why: format!("{view} does not read documents; use transcript or raw"),
                        continue_call: None,
                    },
                })],
            }))
        }
        Err(e) => Err(error("read", e)),
    }
}

/// Every emit checks the pinned archive scope, including the final completion.
/// The callback remains valid after a Reader has released a prepared spool.
pub(super) struct Stream<'a> {
    output: &'a mut dyn Write,
    check: &'a dyn Fn() -> Result<(), Response>,
    counts: Counts,
    shown: u64,
}
impl<'a> Stream<'a> {
    pub(super) fn begin(
        output: &'a mut dyn Write,
        check: &'a dyn Fn() -> Result<(), Response>,
        begin: &ReadBegin,
    ) -> Result<Self, Response> {
        let mut stream = Self { output, check, counts: Counts::default(), shown: 0 };
        stream.emit("begin", begin)?;
        Ok(stream)
    }

    fn emit(&mut self, name: &str, payload: &impl serde::Serialize) -> Result<(), Response> {
        (self.check)()?;
        crate::protocol::stream::event(self.output, name, payload)
            .map_err(|e| error("output", e))
    }

    pub(super) fn record(&mut self, record: &ReadRecord) -> Result<(), Response> {
        if matches!(record.record(), Some(Record::Manifest(_))) {
            return Err(error("read", "a read cannot emit an embedded manifest"));
        }
        self.emit("record", record)?;
        if let Some(record) = record.record() {
            match record {
                Record::Session(_) => self.counts.sessions += 1,
                Record::Message(_) => self.counts.messages += 1,
                Record::Document(_) => self.counts.documents += 1,
                Record::Event(_) => self.counts.events += 1,
                Record::Link(_) => self.counts.links += 1,
                Record::Authored(_) => self.counts.authored += 1,
                Record::Variant(_) => self.counts.variants += 1,
                Record::VariantPart(_) => self.counts.variant_parts += 1,
                Record::Blob(_) => self.counts.blobs += 1,
                Record::Manifest(_) => return Err(error("read", "a read cannot emit an embedded manifest")),
            }
        }
        self.shown += record.parts().iter().filter(|part| matches!(part, ReadPart::Item(_))).count() as u64;
        Ok(())
    }

    pub(super) fn blob(&mut self, chunk: &ReadBlobChunk) -> Result<(), Response> {
        self.emit("blob", chunk)
    }

    pub(super) fn finish(mut self, mut footer: ReadFooter) -> Result<(), Response> {
        footer.shown = self.shown;
        footer.count = ValueState::known(self.shown);
        let complete = ReadComplete { footer, counts: self.counts };
        self.emit("complete", &complete)
    }
}
