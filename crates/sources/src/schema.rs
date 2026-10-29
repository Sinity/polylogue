//! Stored structure and counts, with dynamic map keys grouped together.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, BufRead, Read};

mod diff;
mod observe;
pub use diff::{Change, ChangeKind, Diff, Sample, TableChange, compare};

pub const RULE_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub input: String,
    pub revision: Option<i64>,
    pub records: u64,
    pub unreadable_records: u64,
    /// A fixed reason code, never an error containing source bytes.
    pub unreadable: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub types: BTreeMap<String, u64>,
    pub occurrences: u64,
    pub parents: u64,
    pub present: u64,
    pub missing: u64,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Schema {
    pub rule_version: u32,
    pub origin: String,
    /// Names forced into paths even when they look like dynamic keys.
    pub named_fields: BTreeSet<String>,
    /// Explicit input selection is a sample, with no claim of origin coverage.
    pub sample: bool,
    pub selection: Vec<Selection>,
    pub records: u64,
    pub unreadable_inputs: u64,
    pub unreadable_records: u64,
    pub fields: BTreeMap<String, Field>,
    /// Native row-dump table identities; scalar cell values and SQL are discarded.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tables: BTreeMap<String, Table>,
}

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct Column {
    pub cid: i64,
    pub name: String,
    pub declared_type: String,
    pub not_null: bool,
    pub primary_key: i64,
    pub hidden: i64,
}

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct TableMetadata {
    pub columns: Vec<Column>,
    pub without_rowid: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Table {
    pub records: u64,
    pub deleted_records: u64,
    pub headers: u64,
    pub metadata_headers: u64,
    /// Distinct producer declarations, separate from observed JSON types.
    pub metadata: BTreeSet<TableMetadata>,
    pub unobserved_columns: bool,
    /// Declared native columns with no captured row-field observation.
    pub unobserved_declared_columns: BTreeSet<String>,
    pub fields: BTreeMap<String, Field>,
}

fn unobserved_declared_columns(
    metadata: &BTreeSet<TableMetadata>,
    fields: &BTreeMap<String, Field>,
) -> BTreeSet<String> {
    metadata
        .iter()
        .flat_map(|metadata| metadata.columns.iter())
        .filter(|column| !fields.contains_key(&format!("/{}", field_name(&column.name))))
        .map(|column| column.name.clone())
        .collect()
}

#[derive(Default)]
struct ObservedTable {
    deleted_records: u64,
    root: Tree,
    records: u64,
    headers: u64,
    metadata_headers: u64,
    metadata: BTreeSet<TableMetadata>,
}

#[derive(Debug, Clone, Copy)]
pub enum Framing {
    Lines,
    Json,
    Array,
    CaptureLog,
    RowDump,
}

#[derive(Default)]
pub struct Observer {
    names: BTreeSet<String>,
    root: Tree,
    records: u64,
    unreadable: u64,
    tables: BTreeMap<String, ObservedTable>,
}

#[derive(Default)]
struct Tree {
    nodes: Vec<Node>,
    free: Vec<usize>,
}

#[derive(Default)]
struct Node {
    types: BTreeMap<String, u64>,
    occurrences: u64,
    present: u64,
    children: BTreeMap<String, usize>,
    // At most 51 names establish whether all unforced members must be grouped.
    members: BTreeSet<String>,
    grouped: bool,
    // Grouped presence is the number of objects with any unforced member.
    unforced_present: u64,
    // Parser-assigned numbers count presence once per parent at a shared path.
    last_unforced: u64,
    last_parent: u64,
}

impl Tree {
    fn root(&mut self) -> usize {
        if self.nodes.is_empty() {
            self.nodes.push(Node::default());
        }
        0
    }

    fn allocate(&mut self) -> usize {
        if let Some(index) = self.free.pop() {
            index
        } else {
            self.nodes.push(Node::default());
            self.nodes.len() - 1
        }
    }

    fn child(&mut self, parent: usize, key: String) -> usize {
        if let Some(index) = self.nodes[parent].children.get(&key) {
            return *index;
        }
        let index = self.allocate();
        self.nodes[parent].children.insert(key, index);
        index
    }

    fn kind(&mut self, index: usize, kind: &str) {
        let node = &mut self.nodes[index];
        *node.types.entry(kind.into()).or_default() += 1;
        node.occurrences += 1;
    }

    fn member(
        &mut self,
        parent: usize,
        name: &str,
        object: u64,
        names: &BTreeSet<String>,
        fixed: bool,
    ) -> usize {
        let forced = fixed || names.contains(name);
        let key = field_name(name);
        let node = &mut self.nodes[parent];
        let was_grouped = node.grouped;
        if !node.grouped && !fixed {
            node.members.insert(key.clone());
            if node.members.len() > 50 {
                node.grouped = true;
            }
        }
        if !forced && node.last_unforced != object {
            node.unforced_present += 1;
            node.last_unforced = object;
        }
        if !was_grouped && node.grouped {
            self.normalize(parent, names);
        }
        let grouped = !forced && (self.nodes[parent].grouped || identifier(name));
        let index = self.child(parent, if grouped { "*".into() } else { key });
        let node = &mut self.nodes[index];
        if node.last_parent != object {
            node.present += 1;
            node.last_parent = object;
        }
        index
    }

    fn group(&mut self, parent: usize, names: &BTreeSet<String>) {
        if !self.nodes[parent].grouped {
            return;
        }
        self.nodes[parent].members.clear();
        let keys: Vec<_> = self.nodes[parent]
            .children
            .keys()
            .filter(|key| {
                key.as_str() != "[]" && key.as_str() != "*" && !names.contains(&unescape(key))
            })
            .cloned()
            .collect();
        if keys.is_empty() {
            return;
        }
        let target = self.child(parent, "*".into());
        for key in keys {
            let source = self.nodes[parent]
                .children
                .remove(&key)
                .expect("observed member");
            self.merge_nodes(target, source);
        }
        self.nodes[target].present = self.nodes[parent].unforced_present;
        self.nodes[target].last_parent = self.nodes[parent].last_unforced;
    }

    fn merge_nodes(&mut self, target: usize, source: usize) {
        let mut pending = vec![(target, source)];
        while let Some((target, source)) = pending.pop() {
            let node = std::mem::take(&mut self.nodes[source]);
            self.free.push(source);
            self.add_counts(target, &node);
            for (key, source) in node.children {
                let child = self.child(target, key);
                pending.push((child, source));
            }
        }
    }

    fn add_counts(&mut self, target: usize, source: &Node) {
        let node = &mut self.nodes[target];
        node.occurrences += source.occurrences;
        node.present += source.present;
        node.unforced_present += source.unforced_present;
        node.last_parent = node.last_parent.max(source.last_parent);
        node.last_unforced = node.last_unforced.max(source.last_unforced);
        for (kind, count) in &source.types {
            *node.types.entry(kind.clone()).or_default() += count;
        }
        node.grouped |= source.grouped;
        if !node.grouped {
            for key in &source.members {
                node.members.insert(key.clone());
                if node.members.len() > 50 {
                    node.grouped = true;
                    break;
                }
            }
        }
    }

    fn merge(&mut self, mut other: Self, names: &BTreeSet<String>) {
        if other.nodes.is_empty() {
            return;
        }
        let root = self.root();
        let mut pending = vec![(root, Some(0))];
        while let Some((target, source)) = pending.pop() {
            let Some(source) = source else {
                self.group_presence(target);
                continue;
            };
            let node = std::mem::take(&mut other.nodes[source]);
            let was_grouped = self.nodes[target].grouped;
            self.add_counts(target, &node);
            // Only a newly grouped path needs to fold its existing subtrees.
            if !was_grouped && self.nodes[target].grouped {
                self.normalize(target, names);
            }
            pending.push((target, None));
            for (key, source) in node.children {
                let key = if self.nodes[target].grouped
                    && key != "[]"
                    && !names.contains(&unescape(&key))
                {
                    "*".into()
                } else {
                    key
                };
                let child = self.child(target, key);
                pending.push((child, Some(source)));
            }
        }
    }

    fn normalize(&mut self, root: usize, names: &BTreeSet<String>) {
        let mut pending = vec![root];
        while let Some(index) = pending.pop() {
            self.group(index, names);
            self.group_presence(index);
            pending.extend(self.nodes[index].children.values().copied());
        }
    }

    fn group_presence(&mut self, index: usize) {
        if self.nodes[index].grouped
            && let Some(child) = self.nodes[index].children.get("*").copied()
        {
            self.nodes[child].present = self.nodes[index].unforced_present;
            self.nodes[child].last_parent = self.nodes[index].last_unforced;
        }
    }

    fn flatten(mut self, records: u64) -> BTreeMap<String, Field> {
        let mut fields = BTreeMap::new();
        let mut pending = vec![(0, String::new(), records)];
        while let Some((index, path, parents)) = pending.pop() {
            let node = std::mem::take(&mut self.nodes[index]);
            let objects = node.types.get("object").copied().unwrap_or(0);
            let arrays = node.types.get("array").copied().unwrap_or(0);
            for (key, child) in node.children {
                let parents = if key == "[]" { arrays } else { objects };
                pending.push((child, format!("{path}/{key}"), parents));
            }
            fields.insert(
                path,
                Field {
                    types: node.types,
                    occurrences: node.occurrences,
                    parents,
                    present: node.present,
                    missing: parents - node.present,
                    required: parents > 0 && node.present == parents,
                },
            );
        }
        fields
    }
}

fn unescape(key: &str) -> String {
    key.replace("~3", "[]")
        .replace("~2", "*")
        .replace("~1", "/")
        .replace("~0", "~")
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
        self.root.merge(node, &self.names);
        self.records += 1;
        Ok(())
    }

    pub fn read(&mut self, reader: impl BufRead, framing: Framing) -> io::Result<()> {
        match framing {
            Framing::Lines => self.lines(reader, false),
            Framing::RowDump => self.lines(reader, true),
            Framing::Json => self.json(reader).map_err(io::Error::other),
            Framing::Array => {
                let whole = observe::array(reader, &self.names).map_err(io::Error::other)?;
                self.merge(whole).map_err(io::Error::other)?;
                Ok(())
            }
            Framing::CaptureLog => self.captures(reader),
        }
    }

    fn lines(&mut self, mut reader: impl BufRead, row_dump: bool) -> io::Result<()> {
        while !reader.fill_buf()?.is_empty() {
            // Draining a failed record starts the next read at its own newline.
            let mut line = observe::Line {
                reader: &mut reader,
                ended: false,
                nonwhite: false,
            };
            let result = if row_dump {
                observe::row_dump(&mut line, &self.names).map(|record| {
                    for (name, metadata) in record.tables {
                        let table = self.tables.entry(name).or_default();
                        table.headers += 1;
                        if let Some(metadata) = metadata {
                            table.metadata_headers += 1;
                            table.metadata.insert(metadata);
                        }
                    }
                    if record.gone {
                        self.tables
                            .entry(record.table.clone().expect("validated table"))
                            .or_default()
                            .deleted_records += 1;
                    }
                    if let (Some(name), Some(row)) = (record.table, record.row) {
                        let table = self.tables.entry(name).or_default();
                        table.root.merge(row, &self.names);
                        table.records += 1;
                        self.records += 1;
                    }
                })
            } else {
                self.json(&mut line)
            };
            io::copy(&mut line, &mut io::sink())?;
            if let Err(error) = result {
                if error.is_io() {
                    return Err(io::Error::other("input_read"));
                }
                if line.nonwhite {
                    self.unreadable += 1;
                }
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
            let mut frame = CaptureReader {
                reader: reader.by_ref().take(header.length),
                verifier: header.verifier(),
            };
            whole.json(&mut frame).map_err(io::Error::other)?;
            if !frame.verifier.verifies() {
                return Err(io::Error::other("invalid_capture"));
            }
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
        self.root.merge(other.root, &self.names);
        for (name, other) in other.tables {
            let table = self.tables.entry(name).or_default();
            table.root.merge(other.root, &self.names);
            table.records += other.records;
            table.deleted_records += other.deleted_records;
            table.headers += other.headers;
            table.metadata_headers += other.metadata_headers;
            table.metadata.extend(other.metadata);
        }
        self.records += other.records;
        self.unreadable += other.unreadable;
        Ok(())
    }

    pub fn finish(self, origin: String, mut selection: Vec<Selection>) -> Schema {
        selection.sort_by(|a, b| (&a.input, a.revision).cmp(&(&b.input, b.revision)));
        let fields = if !self.root.nodes.is_empty() {
            self.root.flatten(self.records)
        } else {
            BTreeMap::new()
        };
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
            tables: self
                .tables
                .into_iter()
                .map(|(name, table)| {
                    let fields = if table.records > 0 {
                        table.root.flatten(table.records)
                    } else {
                        BTreeMap::new()
                    };
                    (
                        name,
                        Table {
                            records: table.records,
                            deleted_records: table.deleted_records,
                            headers: table.headers,
                            metadata_headers: table.metadata_headers,
                            unobserved_columns: table.records == 0,
                            unobserved_declared_columns: unobserved_declared_columns(
                                &table.metadata,
                                &fields,
                            ),
                            metadata: table.metadata,
                            fields,
                        },
                    )
                })
                .collect(),
        }
    }
}

struct CaptureReader<'a, R> {
    reader: io::Take<R>,
    verifier: polylogue_model::capture_log::Verifier<'a>,
}

impl<R: Read> Read for CaptureReader<'_, R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let length = self.reader.read(bytes)?;
        self.verifier.update(&bytes[..length]);
        Ok(length)
    }
}

/// Identifier-shaped members describe map entries rather than fixed fields.
fn identifier(name: &str) -> bool {
    let hex = |text: &str| text.bytes().all(|c| c.is_ascii_hexdigit());
    let uuid = name.len() == 36
        && [8, 13, 18, 23].iter().all(|i| name.as_bytes()[*i] == b'-')
        && name.split('-').all(hex);
    let hash = name.len() >= 16 && hex(name);
    let numeral = name.len() >= 6 && name.bytes().all(|c| c.is_ascii_digit());
    let date = name.len() >= 10
        && name.as_bytes()[..10].iter().enumerate().all(|(i, c)| {
            if i == 4 || i == 7 {
                *c == b'-'
            } else {
                c.is_ascii_digit()
            }
        })
        && (name.len() == 10 || matches!(name.as_bytes()[10], b'T' | b' '));
    let path = name.starts_with(['/', '\\'])
        || name.starts_with("~/")
        || name.starts_with("./")
        || name.starts_with("../")
        || name.contains('\\')
        || name.contains("://")
        || name.contains('/');
    uuid || hash || numeral || date || path
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
