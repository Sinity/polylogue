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
            .step(None, bytes, 0)
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
    let mut accounting = polylogue_model::Accounting::default();
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
    for mut step in steps {
        accounting.merge(&step.accounting);
        macro_rules! append { ($($field:ident),*) => { $(output.$field.append(&mut step.output.$field);)* } }
        append!(
            sessions,
            messages,
            documents,
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
