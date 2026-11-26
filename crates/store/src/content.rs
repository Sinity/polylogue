//! Content stored once in the index (docs/archive.md, content). Texts,
//! message readings and event readings keep their bytes in groups: each
//! `content_group` row is one zstd frame of about `GROUP_BYTES` of pieces
//! laid one after another in the order they were written, and a row names
//! its piece by group, start and length in the group's decoded bytes. The
//! shown rows name the piece of the reading they show; nothing is copied.
//! Groups are made off the writer (the parse workers compress them) and
//! never change once stored, and a group's key is never given to another
//! group, so a reference read once stays valid while the group exists.
use crate::row::Rows as _;
use crate::{Result, StoreError};
use rusqlite::Connection;
use std::cell::RefCell;
use std::ops::Range;

/// Uncompressed bytes a group collects before it is closed. A piece longer
/// than this is a group of its own.
pub(crate) const GROUP_BYTES: usize = 64 * 1024;
/// Workers compress; the writer and readers only copy and decode.
const LEVEL: i32 = 3;
/// The decoded group a thread last read is kept for the next read when it is
/// no larger than this, so a group holding one very large piece is not held.
const KEPT_GROUP_BYTES: usize = 4 * GROUP_BYTES;

/// One compressed group and the hash that names it: two equal groups are one
/// row.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Group {
    pub(crate) hash: [u8; 32],
    pub(crate) body: Vec<u8>,
}

impl Group {
    fn compress(bytes: &[u8]) -> Result<Self> {
        let body = COMPRESSOR.with(|compressor| {
            let mut compressor = compressor.borrow_mut();
            let compressor = match &mut *compressor {
                Some(compressor) => compressor,
                empty => empty.insert(zstd::bulk::Compressor::new(LEVEL)?),
            };
            compressor.compress(bytes)
        })?;
        Ok(Self {
            hash: *blake3::hash(&body).as_bytes(),
            body,
        })
    }
}

/// Where a piece lies in the groups prepared with it: the group's place in
/// that list, and the piece's start and length in its decoded bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Placed {
    pub(crate) group: usize,
    pub(crate) start: u64,
    pub(crate) length: u64,
}

/// Groups prepared together (one parse step's), in the order they were made.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Groups(Vec<Group>);

impl From<Vec<Group>> for Groups {
    fn from(groups: Vec<Group>) -> Self {
        Self(groups)
    }
}

impl From<Groups> for Vec<Group> {
    fn from(groups: Groups) -> Self {
        groups.0
    }
}

impl Groups {
    pub(crate) fn get(&self, group: usize) -> Result<&Group> {
        self.0
            .get(group)
            .ok_or_else(|| StoreError::Invalid("a prepared piece names no group".into()))
    }

    /// The bytes of a prepared piece.
    pub(crate) fn read(&self, placed: Placed) -> Result<Vec<u8>> {
        let group = self.get(placed.group)?;
        slice(&group.body, to_i64(placed.start)?, to_i64(placed.length)?)
    }
}

fn to_i64(n: u64) -> Result<i64> {
    i64::try_from(n).map_err(|_| StoreError::Invalid("a piece lies beyond SQLite's range".into()))
}

/// Lays pieces one after another into groups of about [`GROUP_BYTES`].
#[derive(Default)]
pub(crate) struct Packer {
    groups: Vec<Group>,
    pending: Vec<u8>,
    /// Pieces in `pending`, counted apart from its bytes: an empty piece
    /// still needs its group.
    members: usize,
}

impl Packer {
    /// A packer whose groups follow `groups`.
    pub(crate) fn continuing(groups: Groups) -> Self {
        Self {
            groups: groups.0,
            pending: Vec::new(),
            members: 0,
        }
    }

    /// Adds a piece; where it will lie once the groups are finished.
    pub(crate) fn add(&mut self, piece: &[u8]) -> Result<Placed> {
        if self.members > 0 && self.pending.len() + piece.len() > GROUP_BYTES {
            self.close()?;
        }
        let placed = Placed {
            group: self.groups.len(),
            start: self.pending.len() as u64,
            length: piece.len() as u64,
        };
        self.pending.extend_from_slice(piece);
        self.members += 1;
        Ok(placed)
    }

    fn close(&mut self) -> Result<()> {
        self.groups.push(Group::compress(&self.pending)?);
        self.pending.clear();
        self.members = 0;
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<Groups> {
        if self.members > 0 {
            self.close()?;
        }
        Ok(Groups(self.groups))
    }
}

/// The row of a group, stored once: a group already stored with the same
/// bytes is shared.
pub(crate) fn store(conn: &Connection, group: &Group) -> Result<i64> {
    let stored: Option<i64> = conn.value(
        "SELECT id FROM content_group WHERE hash=?1",
        [group.hash.as_slice()],
    )?;
    if let Some(id) = stored {
        return Ok(id);
    }
    conn.prepare_cached("INSERT INTO content_group(hash,body) VALUES (?1,?2)")?
        .execute(rusqlite::params![group.hash.as_slice(), group.body])?;
    Ok(conn.last_insert_rowid())
}

/// The rows of one list of prepared groups, each stored when a row first
/// names it.
pub(crate) struct Stored<'g> {
    groups: &'g Groups,
    rows: Vec<Option<i64>>,
}

impl<'g> Stored<'g> {
    pub(crate) fn new(groups: &'g Groups) -> Self {
        Self {
            groups,
            rows: vec![None; groups.0.len()],
        }
    }

    /// The stored row of a prepared piece's group, and where the piece lies.
    pub(crate) fn place(&mut self, conn: &Connection, placed: Placed) -> Result<Stretch> {
        let group = self.groups.get(placed.group)?;
        let row = match self.rows[placed.group] {
            Some(row) => row,
            None => {
                let row = store(conn, group)?;
                self.rows[placed.group] = Some(row);
                row
            }
        };
        Ok(Stretch {
            group: row,
            start: to_i64(placed.start)?,
            length: to_i64(placed.length)?,
        })
    }
}

/// A stored piece: its group's row, start and length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Stretch {
    pub(crate) group: i64,
    pub(crate) start: i64,
    pub(crate) length: i64,
}

/// A stored piece that may be followed by a second one, its data: a block's
/// body and its structured value, or an event's fields and its data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Piece {
    pub(crate) group: i64,
    pub(crate) start: i64,
    pub(crate) length: i64,
    pub(crate) data_length: Option<i64>,
}

impl Piece {
    /// The piece in columns `at..at + 4` of `row`: group, start, length and
    /// data length.
    pub(crate) fn of(row: &rusqlite::Row<'_>, at: usize) -> rusqlite::Result<Self> {
        Ok(Self {
            group: row.get(at)?,
            start: row.get(at + 1)?,
            length: row.get(at + 2)?,
            data_length: row.get(at + 3)?,
        })
    }

    /// Where its data lies, when it has data.
    pub(crate) fn data(&self) -> Option<Stretch> {
        self.data_length.map(|length| Stretch {
            group: self.group,
            start: self.start + self.length,
            length,
        })
    }

    pub(crate) fn body(&self) -> Stretch {
        Stretch {
            group: self.group,
            start: self.start,
            length: self.length,
        }
    }
}

/// SQL for the JSON text of the body of the block or event row `{r}`: a
/// block with its text moved out, or an event's fields.
pub(crate) const BODY_SQL: &str =
    "(SELECT pl_text_at({r}_g.body,{r}.content_start,{r}.content_length)
 FROM content_group {r}_g WHERE {r}_g.id={r}.content_group)";
/// SQL for the JSON text of the data of the block or event row `{r}`: a
/// tool result's structured value or an event's data; NULL when it has none.
pub(crate) const DATA_SQL: &str =
    "(SELECT pl_text_at({r}_g.body,{r}.content_start+{r}.content_length,{r}.data_length)
 FROM content_group {r}_g WHERE {r}_g.id={r}.content_group AND {r}.data_length IS NOT NULL)";
/// [`BODY_SQL`] as bytes.
pub(crate) const BODY_BYTES_SQL: &str =
    "(SELECT pl_content({r}_g.body,{r}.content_start,{r}.content_length)
 FROM content_group {r}_g WHERE {r}_g.id={r}.content_group)";
/// [`DATA_SQL`] as bytes.
pub(crate) const DATA_BYTES_SQL: &str =
    "(SELECT pl_content({r}_g.body,{r}.content_start+{r}.content_length,{r}.data_length)
 FROM content_group {r}_g WHERE {r}_g.id={r}.content_group AND {r}.data_length IS NOT NULL)";

/// One of the SQL expressions above for the row named `row`.
pub(crate) fn sql_for(expression: &str, row: &str) -> String {
    expression.replace("{r}", row)
}

/// Removes a group once no text, message reading or event reading names it.
/// The shown rows name only pieces of the readings present, so a group no
/// reading names is named by nothing.
pub(crate) fn release(conn: &Connection, group: i64) -> Result<()> {
    conn.prepare_cached(
        "DELETE FROM content_group WHERE id=?1
         AND NOT EXISTS(SELECT 1 FROM text WHERE group_id=?1)
         AND NOT EXISTS(SELECT 1 FROM message_parse WHERE content_group=?1)
         AND NOT EXISTS(SELECT 1 FROM event_parse WHERE content_group=?1)",
    )?
    .execute([group])?;
    Ok(())
}

/// Stores a piece in a group of its own, for rows written without a reading
/// (test archives).
#[cfg(any(test, feature = "fixture"))]
pub(crate) fn store_piece(conn: &Connection, body: &[u8], data: Option<&[u8]>) -> Result<Piece> {
    let mut bytes = body.to_vec();
    bytes.extend_from_slice(data.unwrap_or_default());
    let group = store(conn, &Group::compress(&bytes)?)?;
    Ok(Piece {
        group,
        start: 0,
        length: body.len() as i64,
        data_length: data.map(|data| data.len() as i64),
    })
}

/// Points row `pk` of `table` (`block` or `event`) at a new piece holding
/// `body` (else the row's own body) and `data`, as a reading holding them
/// would have (test archives).
#[cfg(test)]
pub(crate) fn rewrite_piece(
    conn: &Connection,
    table: &str,
    pk: i64,
    body: Option<&[u8]>,
    data: Option<&[u8]>,
) -> Result<()> {
    let stored = conn.query_row(
        &format!(
            "SELECT content_group,content_start,content_length,data_length FROM {table} WHERE pk=?1"
        ),
        [pk],
        |r| Piece::of(r, 0),
    )?;
    let own = Reader::default().bytes(conn, stored.body())?.to_vec();
    let piece = store_piece(conn, body.unwrap_or(&own), data)?;
    conn.execute(
        &format!(
            "UPDATE {table} SET content_group=?2,content_start=?3,content_length=?4,data_length=?5 WHERE pk=?1"
        ),
        rusqlite::params![pk, piece.group, piece.start, piece.length, piece.data_length],
    )?;
    Ok(())
}

/// Points the rows of `table` (`block` or `event`) matching `condition` at a
/// piece whose body does not decode, keeping their data's length (test
/// archives: what a reader never decodes cannot fail it).
#[cfg(test)]
pub(crate) fn poison_bodies(
    conn: &Connection,
    table: &str,
    condition: &str,
    params: impl rusqlite::Params,
) -> Result<usize> {
    let rows = matching(conn, table, condition, params)?;
    for &pk in &rows {
        let stored = conn.query_row(
            &format!("SELECT content_group,content_start,content_length,data_length FROM {table} WHERE pk=?1"),
            [pk],
            |r| Piece::of(r, 0),
        )?;
        let data = match stored.data() {
            Some(data) => Some(Reader::default().bytes(conn, data)?.to_vec()),
            None => None,
        };
        rewrite_piece(conn, table, pk, Some(b"\xff"), data.as_deref())?;
    }
    Ok(rows.len())
}

/// Points the rows of `table` matching `condition` at a piece whose data does
/// not decode, keeping their body (test archives).
#[cfg(test)]
pub(crate) fn poison_data(
    conn: &Connection,
    table: &str,
    condition: &str,
    params: impl rusqlite::Params,
) -> Result<usize> {
    let rows = matching(conn, table, condition, params)?;
    for &pk in &rows {
        rewrite_piece(conn, table, pk, None, Some(b"\xff"))?;
    }
    Ok(rows.len())
}

#[cfg(test)]
fn matching(
    conn: &Connection,
    table: &str,
    condition: &str,
    params: impl rusqlite::Params,
) -> Result<Vec<i64>> {
    conn.values(&format!("SELECT pk FROM {table} WHERE {condition}"), params)
}

/// Reads stored pieces by their group's key on one connection, decoding a
/// group once for consecutive pieces of it. Groups never change and keys are
/// never reused, so the decoded group stays right across transactions.
#[derive(Default)]
pub(crate) struct Reader {
    last: Option<(i64, Vec<u8>)>,
}

impl Reader {
    pub(crate) fn bytes(&mut self, conn: &Connection, stretch: Stretch) -> Result<&[u8]> {
        if self
            .last
            .as_ref()
            .is_none_or(|(id, _)| *id != stretch.group)
        {
            let body: Vec<u8> = conn
                .value(
                    "SELECT body FROM content_group WHERE id=?1",
                    [stretch.group],
                )?
                .ok_or_else(|| StoreError::stored(0, "a stored piece names no group"))?;
            self.last = Some((stretch.group, decode(&body)?));
        }
        let decoded = &self.last.as_ref().expect("just read").1;
        decoded
            .get(range(stretch.start, stretch.length)?)
            .ok_or_else(|| StoreError::stored(0, "a stored piece lies outside its group"))
    }

    /// A stored piece read as JSON.
    pub(crate) fn json<T: serde::de::DeserializeOwned>(
        &mut self,
        conn: &Connection,
        stretch: Stretch,
    ) -> Result<T> {
        crate::encoding::read_json(self.bytes(conn, stretch)?)
    }
}

thread_local! {
    /// Each worker reuses its compression context: creating one per group
    /// costs more than compressing a small group.
    static COMPRESSOR: RefCell<Option<zstd::bulk::Compressor<'static>>> = const { RefCell::new(None) };
    /// The compressed and decoded bytes of the group this thread last read
    /// by its body. Groups are compared by their bytes, so the entry is right
    /// for any connection and any snapshot.
    static LAST_GROUP: RefCell<(Vec<u8>, Vec<u8>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// The bytes at `start..start + length` of the group `body`. Consecutive
/// reads from one group decode it once.
pub(crate) fn slice(body: &[u8], start: i64, length: i64) -> Result<Vec<u8>> {
    let range = range(start, length)?;
    LAST_GROUP.with(|last| {
        let mut last = last.borrow_mut();
        if last.0 != body {
            let decoded = decode(body)?;
            if decoded.len() > KEPT_GROUP_BYTES {
                return within(&decoded, range).map(<[u8]>::to_vec);
            }
            *last = (body.to_vec(), decoded);
        }
        within(&last.1, range).map(<[u8]>::to_vec)
    })
}

/// The text at `start..start + length` of the group `body`.
pub(crate) fn text(body: &[u8], start: i64, length: i64) -> Result<String> {
    String::from_utf8(slice(body, start, length)?)
        .map_err(|_| StoreError::stored(0, "a stored text is not valid UTF-8"))
}

pub(crate) fn decode(body: &[u8]) -> Result<Vec<u8>> {
    crate::encoding::unpack_bytes(body)
        .map_err(|error| StoreError::stored(0, format!("a content group does not decode: {error}")))
}

pub(crate) fn range(start: i64, length: i64) -> Result<Range<usize>> {
    usize::try_from(start)
        .ok()
        .zip(usize::try_from(length).ok())
        .and_then(|(start, length)| Some(start..start.checked_add(length)?))
        .ok_or_else(|| StoreError::stored(0, "a stored piece has an invalid place"))
}

fn within(group: &[u8], range: Range<usize>) -> Result<&[u8]> {
    group
        .get(range)
        .ok_or_else(|| StoreError::stored(0, "a stored piece lies outside its group"))
}

/// What a value left in the stored bytes reads as in the index
/// (docs/archive.md, content): a string naming the record that holds it, as
/// a byte range of the reading's chain, and the value's JSON pointer in that
/// record. Only a reader showing the value reads the record
/// ([`crate::read::Reader::resolve_stored_value`]).
pub(crate) fn stored_marker(field: &polylogue_model::StoredField) -> String {
    format!("{STORED}{} {} {}", field.start, field.end, field.pointer)
}

const STORED: &str = "\u{0}stored ";

/// Where a value [`stored_marker`] names is: the record's byte range in the
/// chain and the value's pointer in it.
pub(crate) fn stored_at(text: &str) -> Option<(u64, u64, &str)> {
    let rest = text.strip_prefix(STORED)?;
    let (start, rest) = rest.split_once(' ')?;
    let (end, pointer) = rest.split_once(' ')?;
    Some((start.parse().ok()?, end.parse().ok()?, pointer))
}

/// Replaces the value at `field.field` in `value` with its marker.
pub(crate) fn leave_stored(
    value: &mut serde_json::Value,
    field: &polylogue_model::StoredField,
) -> Result<()> {
    let slot = value.pointer_mut(&field.field).ok_or_else(|| {
        StoreError::Invalid(format!("a stored field names no value at {}", field.field))
    })?;
    *slot = serde_json::Value::String(stored_marker(field));
    Ok(())
}

/// Parts laid out as one piece: the part count and each part's length as
/// varints, then the parts one after another.
pub(crate) fn join_parts(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.len() + 3).sum::<usize>() + 3);
    crate::sorted_runs::put_varint(&mut out, parts.len() as u64);
    for part in parts {
        crate::sorted_runs::put_varint(&mut out, part.len() as u64);
    }
    for part in parts {
        out.extend_from_slice(part);
    }
    out
}

/// Where each part of a piece made by [`join_parts`] lies in it.
pub(crate) fn parts(piece: &[u8]) -> Result<Vec<Range<usize>>> {
    let bad = || StoreError::stored(0, "a stored piece's parts are malformed");
    let mut at = 0usize;
    let mut varint = || -> Result<usize> {
        let (mut value, mut shift) = (0u64, 0);
        loop {
            let byte = *piece.get(at).ok_or_else(bad)?;
            at += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte < 0x80 {
                return usize::try_from(value).map_err(|_| bad());
            }
            shift += 7;
            if shift > 63 {
                return Err(bad());
            }
        }
    };
    let count = varint()?;
    let lengths = (0..count).map(|_| varint()).collect::<Result<Vec<_>>>()?;
    let mut ranges = Vec::with_capacity(count);
    for length in lengths {
        let end = at.checked_add(length).ok_or_else(bad)?;
        ranges.push(at..end);
        at = end;
    }
    if at != piece.len() {
        return Err(bad());
    }
    Ok(ranges)
}

#[cfg(test)]
mod tests;
