//! Reading the producer sidecars written before parse progress became one value.
//! Serialization and the public schema keep only the current progress field.
use super::{CapturedReading, ExportReadingKey, RecordCounts};
use polylogue_model::batch::StepProgress;
use serde::{Deserialize, Deserializer};

#[derive(Deserialize)]
struct Wire {
    export_key: Option<ExportReadingKey>,
    delivery: Option<polylogue_model::batch::ExportDelivery>,
    output: polylogue_model::Output,
    counts: RecordCounts,
    start: u64,
    parsed_upto: u64,
    #[serde(flatten)]
    progress: Progress,
    order_from: Option<u64>,
    reading_version: String,
    repeats_whole_conversations: bool,
    reset: bool,
    comparison: Option<crate::tool_databases::Comparison>,
    checkpoint: Option<polylogue_model::batch::ParserCheckpoint>,
}

#[derive(Deserialize)]
struct Progress {
    progress: Option<StepProgress>,
    at_end: Option<bool>,
    no_parser_yet: Option<bool>,
}

impl<'de> Deserialize<'de> for CapturedReading {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        let progress = match wire.progress.progress {
            Some(progress) => progress,
            None if wire.progress.at_end.is_some() || wire.progress.no_parser_yet.is_some() => {
                let at_end = wire.progress.at_end.unwrap_or(false);
                if wire.progress.no_parser_yet.unwrap_or(false) {
                    StepProgress::NoParserYet { at_end }
                } else {
                    StepProgress::parsed(at_end)
                }
            }
            None => return Err(serde::de::Error::missing_field("progress")),
        };
        Ok(Self {
            export_key: wire.export_key,
            delivery: wire.delivery,
            output: wire.output,
            counts: wire.counts,
            start: wire.start,
            parsed_upto: wire.parsed_upto,
            progress,
            order_from: wire.order_from,
            reading_version: wire.reading_version,
            repeats_whole_conversations: wire.repeats_whole_conversations,
            reset: wire.reset,
            comparison: wire.comparison,
            checkpoint: wire.checkpoint,
        })
    }
}
