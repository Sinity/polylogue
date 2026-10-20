//! Small parser fixtures use in-memory ordering; engine tests exercise disk spill.
use polylogue_sources::{Parse, Step};

pub fn parse(parser: &Parse, bytes: &[u8]) -> Result<Step, String> {
    let Parse::Array(array) = parser else {
        return parser
            .step(None, bytes, 0)
            .map(|(step, _)| step)
            .map_err(|e| e.to_string());
    };
    let mut items = Vec::new();
    if let Err(error) = polylogue_sources::array::scan(bytes, |range, raw| {
        items.push((range, raw.to_owned()));
        Ok(())
    }) {
        return Ok(polylogue_sources::array::refused(
            error.to_string(),
            bytes.len() as u64,
        ));
    }
    items.sort_by(
        |a, b| match (array.order_key(&a.1), array.order_key(&b.1)) {
            (Some(a), Some(b)) => a.total_cmp(&b),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            _ => std::cmp::Ordering::Equal,
        },
    );
    let mut owners = polylogue_sources::chatgpt::Owners::default();
    let mut output = polylogue_sources::Output::default();
    let mut accounting = polylogue_model::Accounting::default();
    for (range, raw) in items {
        let value = serde_json::from_str(raw.get()).map_err(|e| e.to_string())?;
        let mut step = polylogue_sources::array::parse_item(&**array, &value, range, &mut owners)?;
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
