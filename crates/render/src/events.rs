//! Observation payloads are structured facts, never synthetic chat turns.
use crate::{Format, Rendered, View};
use polylogue_model::Event;
use serde_json::{Value, json};
use std::io::{self, Write};

pub fn event(mut event: Event, format: Format) -> io::Result<Value> {
    if format == Format::Json {
        polylogue_model::masking::mask_json(&mut event.data);
        return Ok(json!({"event":event}));
    }
    crate::formatted(&[Rendered::Event(&event)], format, View::Full)
}
pub fn text(event: &Event) -> String {
    let mut data = event.data.clone();
    polylogue_model::masking::mask_json(&mut data);
    data.sort_all_objects();
    serde_json::to_string_pretty(&data).expect("JSON values serialize")
}
pub(crate) fn terminal(event: &Event, out: &mut dyn Write) -> io::Result<()> {
    writeln!(
        out,
        "· {} {}\n{}",
        crate::mask(&event.kind),
        event.at.map(|t| t.to_string()).unwrap_or_default(),
        text(event)
    )
}
pub(crate) fn markdown(event: &Event, out: &mut dyn Write) -> io::Result<()> {
    writeln!(
        out,
        "*· {} {}*\n\n{}\n",
        crate::mask(&event.kind),
        event.at.map(|t| t.to_string()).unwrap_or_default(),
        text(event)
    )
}
pub(crate) fn html(event: &Event, out: &mut dyn Write) -> io::Result<()> {
    write!(
        out,
        "<aside class=\"event\"><strong>{}</strong><pre>{}</pre></aside>",
        crate::html::escape(&crate::mask(&event.kind)),
        crate::html::escape(&text(event))
    )
}
