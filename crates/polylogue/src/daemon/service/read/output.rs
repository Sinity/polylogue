//! One-record projection and transfer over a caller-owned pinned selection.
use super::super::{Response, error};
use polylogue_model::bundle::{Counts, Record};
use polylogue_model::read::*;
use std::io::Write;

/// Canonical rows are kept separate from layout-only omissions. A mixed query
/// can report unsupported targets without exporting their hidden source bodies.
pub(super) fn project(
    selected: super::records::SelectedRecord,
    options: &ReadOptions,
    evidence: &ProjectionEvidence,
    mixed: bool,
    footer: &mut ReadFooter,
) -> Result<Option<ReadRecord>, Response> {
    let record = match selected {
        super::records::SelectedRecord::Canonical(record) => *record,
        super::records::SelectedRecord::LinkIssue { session, issue } => {
            return Ok(Some(ReadRecord::Presentation {
                parts: vec![ReadPart::Item(ReadItem {
                    source: Source::record(format!("session:{session}")),
                    kind: ItemKind::LinkIssue { issue: *issue },
                })],
            }));
        }
    };
    let reference = record.reference().unwrap_or_default().to_owned();
    let target_kind = match &record {
        Record::Document(_) => Some(TargetKind::Document),
        Record::TextFragment(_) => Some(TargetKind::TextFragment),
        _ => None,
    };
    match project_record(record, options, evidence) {
        Ok(Some(record)) => Ok(Some(ReadRecord::Canonical {
            record: Box::new(record),
            parts: Vec::new(),
        })),
        Ok(None) => Ok(None),
        Err(error) => projection_error(reference, target_kind, error, mixed, footer),
    }
}

/// A page can report unsupported document presentation before hydrating bodies.
/// The same omission and error policy serves full canonical projection.
pub(super) fn document_omission(
    reference: &str,
    options: &ReadOptions,
    mixed: bool,
    footer: &mut ReadFooter,
) -> Result<Option<ReadRecord>, Response> {
    if options.view != View::Dialogue {
        return Ok(None);
    }
    projection_error(
        reference.into(),
        Some(TargetKind::Document),
        ReadError::UnsupportedTarget {
            view: options.view,
            target: reference.into(),
            accepted_views: vec![View::Transcript, View::Raw],
        },
        mixed,
        footer,
    )
}

fn projection_error(
    reference: String,
    target_kind: Option<TargetKind>,
    error: ReadError,
    mixed: bool,
    footer: &mut ReadFooter,
) -> Result<Option<ReadRecord>, Response> {
    match error {
        ReadError::UnsupportedTarget {
            view,
            target,
            accepted_views,
        } if mixed && target_kind.is_some() => {
            let target_kind = target_kind.expect("checked target family");
            if let Some(coverage) = footer
                .coverage
                .iter_mut()
                .find(|coverage| coverage.target == target_kind && coverage.view == view)
            {
                coverage.omitted += 1;
            } else {
                footer.coverage.push(Coverage {
                    target: target_kind,
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
                        unit: match target_kind {
                            TargetKind::Document => "document",
                            TargetKind::TextFragment => "text fragment",
                            _ => unreachable!("mixed canonical family"),
                        }
                        .into(),
                        why: ReadError::UnsupportedTarget {
                            view,
                            target,
                            accepted_views: vec![],
                        }
                        .to_string(),
                        continue_call: None,
                    },
                })],
            }))
        }
        error => Err(read_error(error)),
    }
}

pub(super) fn emit_record(
    reader: &polylogue_store::Reader<'_>,
    record: &ReadRecord,
    at: SelectionLocator,
    layout: &super::evidence::Layout<'_>,
    emit: &mut super::Emit<'_>,
) -> Result<bool, Response> {
    if record.record().is_some() && !emit(record, at, None)? {
        return Ok(false);
    }
    super::evidence::visit_parts(reader, record, layout, |item, ordinal| {
        emit(
            &ReadRecord::Presentation {
                parts: vec![ReadPart::Item(item)],
            },
            at,
            Some(ordinal),
        )
    })
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
        let mut stream = Self {
            output,
            check,
            counts: Counts::default(),
            shown: 0,
        };
        stream.emit("begin", begin)?;
        Ok(stream)
    }

    fn emit(&mut self, name: &str, payload: &impl serde::Serialize) -> Result<(), Response> {
        (self.check)()?;
        crate::protocol::stream::event(self.output, name, payload).map_err(|e| error("io", e))
    }

    pub(super) fn record(&mut self, record: &ReadRecord) -> Result<(), Response> {
        if matches!(record.record(), Some(Record::Manifest(_))) {
            return Err(error(
                "bad_request",
                "a read cannot emit an embedded manifest",
            ));
        }
        self.emit("record", record)?;
        if let Some(record) = record.record() {
            match record {
                Record::Session(_) => self.counts.sessions += 1,
                Record::Message(_) => self.counts.messages += 1,
                Record::Document(_) => self.counts.documents += 1,
                Record::Event(_) => self.counts.events += 1,
                Record::Link(_) => self.counts.links += 1,
                Record::Historical(record) => match &record.evidence {
                    polylogue_model::historical::HistoricalRecord::Accepted(_) => {
                        self.counts.authored += 1
                    }
                    polylogue_model::historical::HistoricalRecord::Assertion(_) => {
                        self.counts.assertions += 1
                    }
                    polylogue_model::historical::HistoricalRecord::Run(_) => self.counts.runs += 1,
                },
                Record::Authored(_) => self.counts.authored += 1,
                Record::Assertion(_) => self.counts.assertions += 1,
                Record::Variant(_) => self.counts.variants += 1,
                Record::VariantPart(_) => self.counts.variant_parts += 1,
                Record::TextFragment(_) => self.counts.text_fragments += 1,
                Record::Blob(_) => self.counts.blobs += 1,
                Record::Manifest(_) => {
                    return Err(error(
                        "bad_request",
                        "a read cannot emit an embedded manifest",
                    ));
                }
            }
        }
        self.shown += record
            .parts()
            .iter()
            .filter(|part| matches!(part, ReadPart::Item(_)))
            .count() as u64;
        Ok(())
    }

    pub(super) fn blob(&mut self, chunk: &ReadBlobChunk) -> Result<(), Response> {
        self.emit("blob", chunk)
    }

    pub(super) fn finish(mut self, mut footer: ReadFooter) -> Result<(), Response> {
        footer.shown = self.shown;
        footer.count = ValueState::known(self.shown);
        let complete = ReadComplete {
            footer,
            counts: self.counts,
        };
        self.emit("complete", &complete)
    }
}

pub(super) fn read_error(error: ReadError) -> Response {
    let (kind, detail) = match &error {
        ReadError::UnsupportedTarget {
            view,
            target,
            accepted_views,
        } => (
            "bad_request",
            Some(serde_json::json!({"view":view,"target":target,"accepted_views":accepted_views})),
        ),
        ReadError::RequiredExpansion {
            view,
            target,
            option,
        } => (
            "bad_request",
            Some(
                serde_json::json!({"view":view,"target":target,"option":option,"required":"full"}),
            ),
        ),
        ReadError::MissingEvidence {
            view,
            evidence,
            reference,
        } => (
            "not_yet_available",
            Some(serde_json::json!({"view":view,"evidence":evidence,"ref":reference})),
        ),
        _ => ("bad_request", None),
    };
    Response::Error {
        kind: kind.into(),
        message: error.to_string(),
        detail,
    }
}
