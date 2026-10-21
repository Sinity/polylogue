//! Small parser fixtures use in-memory ordering; engine tests exercise disk spill.
use polylogue_sources::{Parse, Step};

pub fn parse(parser: &mut Parse, bytes: &[u8]) -> Result<Step, String> {
    let Parse::Export(export) = parser else {
        return parser
            .step(None, bytes, 0)
            .map(|(step, _)| step)
            .map_err(|e| e.to_string());
    };
    let mut items = Vec::new();
    let mut keys = std::collections::BTreeMap::new();
    if let Err(error) = export.scan(&mut &bytes[..], &mut |range, key, source_key, raw| {
        let row = (range, key, raw.to_owned());
        if let Some(source_key) = source_key {
            if let Some(&position) = keys.get(source_key) {
                items[position] = row;
            } else {
                keys.insert(source_key.to_owned(), items.len());
                items.push(row);
            }
        } else {
            items.push(row);
        }
        Ok(())
    }) {
        let polylogue_sources::export::ScanError::Malformed(error) = error else {
            return Err(error.to_string());
        };
        return Ok(polylogue_sources::export::refused(
            error.to_string(),
            bytes.len() as u64,
        ));
    }
    items.sort_by(|a, b| a.1.cmp(&b.1));
    let mut owners = polylogue_sources::chatgpt::Owners::default();
    let mut output = polylogue_sources::Output::default();
    let mut accounting = polylogue_model::Accounting::default();
    let first = items
        .first()
        .map(|(_, _, raw)| serde_json::from_str(raw.get()))
        .transpose()
        .map_err(|e| e.to_string())?;
    let mut steps: Vec<_> = export
        .begin(bytes.len() as u64, first.as_ref())?
        .into_iter()
        .collect();
    for (range, _, raw) in items {
        let value = serde_json::from_str(raw.get()).map_err(|e| e.to_string())?;
        steps.push(polylogue_sources::export::parse_item(
            &mut **export,
            &value,
            range,
            &mut owners,
        )?);
    }
    for mut step in steps {
        accounting.merge(&step.accounting);
        macro_rules! append { ($($field:ident),*) => { $(output.$field.append(&mut step.output.$field);)* } }
        append!(
            sessions,
            messages,
            events,
            links,
            background,
            blobs,
            repo_lookups
        );
    }
    Ok(Step {
        output,
        accounting,
        consumed: bytes.len() as u64,
        start: 0,
        snapshot: None,
    })
}
