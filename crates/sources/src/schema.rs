//! Offline structure and counts. Only caller-declared field names leave the observer.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, BufRead, Read};

mod diff;
mod observe;
pub use diff::{Change, ChangeKind, Diff, compare};

pub const RULE_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub input: String,
    pub revision: Option<i64>,
    pub records: u64,
    pub unreadable_records: u64,
    /// A fixed reason code, never an error containing source bytes.
    pub unreadable: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub types: BTreeMap<String, u64>,
    pub occurrences: u64,
    pub parents: u64,
    pub present: u64,
    pub missing: u64,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Schema {
    pub rule_version: u32,
    pub origin: String,
    pub named_fields: BTreeSet<String>,
    /// Explicit input selection is a sample, with no claim of origin coverage.
    pub sample: bool,
    pub selection: Vec<Selection>,
    pub records: u64,
    pub unreadable_inputs: u64,
    pub unreadable_records: u64,
    pub fields: BTreeMap<String, Field>,
}

#[derive(Debug, Clone, Copy)]
pub enum Framing {
    Lines,
    Json,
    Array,
    CaptureLog,
}

#[derive(Default)]
pub struct Observer {
    names: BTreeSet<String>,
    root: Node,
    records: u64,
    unreadable: u64,
}

#[derive(Default)]
struct Node {
    types: BTreeMap<String, u64>,
    occurrences: u64,
    present: u64,
    children: BTreeMap<String, Node>,
}

impl Node {
    fn kind(&mut self, kind: &str) {
        *self.types.entry(kind.into()).or_default() += 1;
        self.occurrences += 1;
    }

    fn merge(&mut self, other: Self) {
        self.occurrences += other.occurrences;
        self.present += other.present;
        for (kind, count) in other.types {
            *self.types.entry(kind).or_default() += count;
        }
        for (key, child) in other.children {
            self.children.entry(key).or_default().merge(child);
        }
    }

    fn flatten(self, path: String, parents: u64, fields: &mut BTreeMap<String, Field>) {
        let objects = self.types.get("object").copied().unwrap_or(0);
        let arrays = self.types.get("array").copied().unwrap_or(0);
        fields.insert(
            path.clone(),
            Field {
                types: self.types,
                occurrences: self.occurrences,
                parents,
                present: self.present,
                missing: parents - self.present,
                required: parents > 0 && self.present == parents,
            },
        );
        for (key, child) in self.children {
            let parents = if key == "[]" { arrays } else { objects };
            child.flatten(format!("{path}/{key}"), parents, fields);
        }
    }
}

impl Observer {
    pub fn new(names: BTreeSet<String>) -> Self {
        Self {
            names,
            ..Self::default()
        }
    }

    /// A failed JSON record contributes no partial structure.
    pub fn json(&mut self, reader: impl Read) -> Result<(), serde_json::Error> {
        let node = observe::read(reader, &self.names)?;
        self.root.merge(node);
        self.records += 1;
        Ok(())
    }

    pub fn read(&mut self, reader: impl BufRead, framing: Framing) -> io::Result<()> {
        match framing {
            Framing::Lines => self.lines(reader),
            Framing::Json => self.json(reader).map_err(io::Error::other),
            Framing::Array => {
                // Malformed exports must not contribute a valid-looking prefix.
                let mut whole = Self::new(self.names.clone());
                crate::export::scan(reader, |_, raw| {
                    whole
                        .json(raw.get().as_bytes())
                        .map_err(|_| "invalid_record".into())
                })
                .map_err(|_| io::Error::other("invalid_export"))?;
                self.merge(whole).map_err(io::Error::other)?;
                Ok(())
            }
            Framing::CaptureLog => self.captures(reader),
        }
    }

    fn lines(&mut self, mut reader: impl BufRead) -> io::Result<()> {
        let mut line = Vec::new();
        loop {
            line.clear();
            if reader.read_until(b'\n', &mut line)? == 0 {
                break;
            }
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            // Offline revisions have a known end, including a complete last line.
            if let Err(error) = self.json(line.as_slice()) {
                if error.is_io() {
                    return Err(io::Error::other("input_read"));
                }
                self.unreadable += 1;
            }
        }
        Ok(())
    }

    fn captures(&mut self, mut reader: impl BufRead) -> io::Result<()> {
        use polylogue_model::capture_log::{HEADER_LEN, Header, MAGIC};
        let mut magic = Vec::new();
        reader
            .by_ref()
            .take(MAGIC.len() as u64)
            .read_to_end(&mut magic)?;
        if magic != MAGIC {
            return self
                .json(io::Cursor::new(magic).chain(reader))
                .map_err(io::Error::other);
        }
        let mut whole = Self::new(self.names.clone());
        while !reader.fill_buf()?.is_empty() {
            let mut header = [0; HEADER_LEN];
            reader.read_exact(&mut header)?;
            let header = Header::decode(&header).map_err(io::Error::other)?;
            let mut bytes = Vec::new();
            reader
                .by_ref()
                .take(header.length)
                .read_to_end(&mut bytes)?;
            if bytes.len() as u64 != header.length || !header.verifies(&bytes) {
                return Err(io::Error::other("invalid_capture"));
            }
            whole.json(bytes.as_slice()).map_err(io::Error::other)?;
        }
        self.merge(whole).map_err(io::Error::other)?;
        Ok(())
    }

    pub fn counts(&self) -> (u64, u64) {
        (self.records, self.unreadable)
    }

    pub fn merge(&mut self, other: Self) -> Result<(), &'static str> {
        if self.names != other.names {
            return Err("observer field-name selections differ");
        }
        self.root.merge(other.root);
        self.records += other.records;
        self.unreadable += other.unreadable;
        Ok(())
    }

    pub fn finish(self, origin: String, mut selection: Vec<Selection>) -> Schema {
        selection.sort_by(|a, b| (&a.input, a.revision).cmp(&(&b.input, b.revision)));
        let mut fields = BTreeMap::new();
        if self.records > 0 {
            self.root.flatten(String::new(), self.records, &mut fields);
        }
        Schema {
            rule_version: RULE_VERSION,
            origin,
            named_fields: self.names,
            sample: true,
            unreadable_inputs: selection.iter().filter(|s| s.unreadable.is_some()).count() as u64,
            selection,
            records: self.records,
            unreadable_records: self.unreadable,
            fields,
        }
    }
}

/// Escaped field names cannot collide with the wildcard or array markers.
fn field_name(name: &str) -> String {
    match name {
        "*" => "~2".into(),
        "[]" => "~3".into(),
        _ => name.replace('~', "~0").replace('/', "~1"),
    }
}

#[cfg(test)]
mod tests;
