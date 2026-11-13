//! Small parser fixtures use in-memory ordering; engine tests exercise disk spill.
use polylogue_sources::{Parse, Step};

pub fn parse(parser: &mut Parse, bytes: &[u8]) -> Result<Step, String> {
    parse_input(
        parser,
        &polylogue_model::BlobHash::of(bytes).to_string(),
        bytes,
    )
}

pub fn parse_input(parser: &mut Parse, input: &str, bytes: &[u8]) -> Result<Step, String> {
    let Parse::Export(export) = parser else {
        return parser
            .step(None, bytes, 0, &Default::default())
            .map(|(step, _)| step)
            .map_err(|e| e.to_string());
    };
    let mut sink = Items::default();
    if let Err(error) = export.scan(input, &mut std::io::Cursor::new(bytes), &mut sink) {
        let polylogue_sources::export::ScanError::Malformed(error) = error else {
            return Err(error.to_string());
        };
        return Ok(polylogue_sources::export::refused(
            error.to_string(),
            bytes.len() as u64,
        ));
    }
    let mut items: Vec<_> = sink.rows.into_iter().flatten().collect();
    items.sort_by(|a, b| a.1.cmp(&b.1));
    let mut owners = polylogue_sources::export::MemoryState::default();
    let mut output = polylogue_sources::Output::default();
    let mut counts = polylogue_model::RecordCounts::default();
    let first = items.first().map(|(_, _, bytes)| bytes.as_slice());
    let mut steps: Vec<_> = export
        .begin(bytes.len() as u64, first)?
        .into_iter()
        .collect();
    for (range, _, raw) in items {
        steps.push(polylogue_sources::export::parse_item(
            &mut **export,
            &raw,
            range,
            &mut owners,
        )?);
    }
    for step in steps {
        counts.merge(&step.counts);
        append_output(&mut output, step.output);
    }
    Ok(Step {
        output,
        counts,
        consumed: bytes.len() as u64,
        start: 0,
        repeats_whole_conversations: false,
    })
}

// Exhaustive destructuring keeps new Output fields visible at this aggregation
// boundary. Sparse keys belong to their vector, never the global record count.
fn append_output(output: &mut polylogue_sources::Output, next: polylogue_sources::Output) {
    let message_base = output.messages.len();
    let event_base = output.events.len();
    let polylogue_sources::Output {
        mut sessions,
        mut messages,
        native_hook_fields,
        declared_injections,
        declared_attachment_bindings,
        mut documents,
        mut fragments,
        mut events,
        imported_events,
        event_ids,
        message_order,
        message_projections,
        mut links,
        mut prefix_copies,
        mut prefix_members,
        mut background,
        mut blobs,
        mut repo_lookups,
    } = next;
    macro_rules! append {
        ($($field:ident),*) => { $(output.$field.append(&mut $field);)* };
    }
    macro_rules! shift {
        ($field:ident, $base:ident) => {
            output.$field.extend(
                $field
                    .into_iter()
                    .map(|(index, value)| (index + $base, value)),
            );
        };
    }
    append!(
        sessions,
        messages,
        documents,
        fragments,
        events,
        links,
        prefix_copies,
        prefix_members,
        background,
        blobs,
        repo_lookups
    );
    shift!(native_hook_fields, message_base);
    shift!(declared_injections, message_base);
    shift!(declared_attachment_bindings, message_base);
    shift!(message_order, message_base);
    shift!(message_projections, message_base);
    shift!(imported_events, event_base);
    shift!(event_ids, event_base);
}

type Row = (std::ops::Range<u64>, Vec<u8>, Vec<u8>);
type Checkpoint = (Vec<Option<Row>>, std::collections::BTreeMap<String, usize>);
#[derive(Default)]
struct Items {
    rows: Vec<Option<Row>>,
    keys: std::collections::BTreeMap<String, usize>,
    checkpoint: Option<Checkpoint>,
}
impl polylogue_sources::export::ItemSink for Items {
    fn push(
        &mut self,
        range: std::ops::Range<u64>,
        key: Vec<u8>,
        source_key: Option<&str>,
        raw: &[u8],
    ) -> Result<(), String> {
        if let Some(source_key) = source_key
            && let Some(previous) = self.keys.insert(source_key.to_owned(), self.rows.len())
        {
            self.rows[previous] = None;
        }
        self.rows.push(Some((range, key, raw.to_vec())));
        Ok(())
    }
    fn begin(&mut self) -> Result<(), String> {
        if self.checkpoint.is_some() {
            return Err("nested fixture scan group".into());
        }
        self.checkpoint = Some((self.rows.clone(), self.keys.clone()));
        Ok(())
    }
    fn commit(&mut self) -> Result<(), String> {
        self.checkpoint.take().ok_or("no fixture scan group")?;
        Ok(())
    }
    fn rollback(&mut self) -> Result<(), String> {
        let (rows, keys) = self.checkpoint.take().ok_or("no fixture scan group")?;
        self.rows = rows;
        self.keys = keys;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_output_preserves_native_field_claim_coordinates() {
        let parser =
            polylogue_sources::claude_code::open(std::path::Path::new("projects/p/s1.jsonl"))
                .unwrap();
        let mut combined = polylogue_sources::Output::default();
        for (index, id) in ["first", "second"].into_iter().enumerate() {
            let record = serde_json::json!({"type":"system","subtype":"stop_hook_summary",
                "sessionId":"s1","uuid":id,"hookAdditionalContext":["context"]});
            let mut bytes = serde_json::to_vec(&record).unwrap();
            bytes.push(b'\n');
            let (step, _) = parser
                .step(None, &bytes, index as u64 * 1000, &Default::default())
                .unwrap();
            let output = step.output;
            append_output(&mut combined, output);
        }
        assert_eq!(combined.messages.len(), 2);
        assert_eq!(combined.native_hook_fields[&1][0].start, 1000);
        assert_eq!(
            combined.messages[1].message.id.to_string(),
            "claude-code:second"
        );
    }
}
