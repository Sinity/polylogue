//! Small parser fixtures use in-memory ordering; engine tests exercise disk spill.
use polylogue_sources::{Parse, Step};

pub fn parse(parser: &mut Parse, bytes: &[u8]) -> Result<Step, String> {
    let Parse::Array(array) = parser else {
        return parser
            .step(None, bytes, 0)
            .map(|(step, _)| step)
            .map_err(|e| e.to_string());
    };
    let mut items = Vec::new();
    if let Err(error) = array.scan(&mut &bytes[..], &mut |range, key, raw| {
        items.push((range, key, raw.to_owned()));
        Ok(())
    }) {
        let polylogue_sources::array::ScanError::Malformed(error) = error else {
            return Err(error.to_string());
        };
        return Ok(polylogue_sources::array::refused(
            error.to_string(),
            bytes.len() as u64,
        ));
    }
    items.sort_by(|a, b| match (a.1, b.1) {
        (Some(a), Some(b)) => a.total_cmp(&b),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        _ => std::cmp::Ordering::Equal,
    });
    let mut owners = polylogue_sources::chatgpt::Owners::default();
    let mut output = polylogue_sources::Output::default();
    let mut accounting = polylogue_model::Accounting::default();
    let mut steps: Vec<_> = array.begin(bytes.len() as u64)?.into_iter().collect();
    for (range, _, raw) in items {
        let value = serde_json::from_str(raw.get()).map_err(|e| e.to_string())?;
        steps.push(polylogue_sources::array::parse_item(
            &mut **array,
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
