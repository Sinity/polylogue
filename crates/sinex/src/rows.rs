//! A row batch as Sinex holds it: JSON lines, one `{"row": {…}, "table": …}`
//! per row with every column under its own name. Integers and reals are
//! numbers, text is a string, a blob is `{"hex": …}` and null is null. Keys
//! are sorted, so the same rows always give the same bytes.
use crate::{Error, Result};
use polylogue_store::row_batches::{BatchReader, Row, RowBatch, Value};
use serde::Serialize;
use serde_json::{Map, Value as Json, json};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, Write};

/// One batch encoded into a private temporary file.
pub(crate) struct Encoded {
    pub file: std::fs::File,
    pub length: u64,
    pub hash: hashed_blobs::Hash,
    pub rows: u64,
    /// The input blobs the rows cite, in row order.
    pub blobs: Vec<hashed_blobs::Hash>,
}

#[derive(Serialize)]
struct Line<'a> {
    row: BTreeMap<&'a str, Json>,
    table: &'a str,
}

pub(crate) fn encode(batch_reader: &BatchReader, batch: &RowBatch) -> Result<Encoded> {
    let mut file = tempfile::tempfile()?;
    let mut out = Hashing {
        inner: BufWriter::new(&mut file),
        digest: blake3::Hasher::new(),
        length: 0,
    };
    let mut rows = 0;
    let mut blobs = Vec::new();
    batch_reader.rows(batch, |row| {
        if row.table == "blob"
            && let Some(Value::Blob(hash)) = column(&row, "hash")
            && let Ok(hash) = <[u8; 32]>::try_from(hash.as_slice())
        {
            blobs.push(hashed_blobs::Hash(hash));
        }
        let line = Line {
            row: row
                .columns
                .iter()
                .zip(row.values)
                .map(|(column, value)| (column.as_str(), to_json(value)))
                .collect(),
            table: row.table,
        };
        serde_json::to_writer(&mut out, &line)?;
        out.write_all(b"\n")?;
        rows += 1;
        Ok(())
    })?;
    out.flush()?;
    let (length, hash) = (
        out.length,
        hashed_blobs::Hash(*out.digest.finalize().as_bytes()),
    );
    drop(out);
    file.rewind()?;
    Ok(Encoded {
        file,
        length,
        hash,
        rows,
        blobs,
    })
}

/// Decodes a batch, passing each row to `visit`, and returns the row count.
pub(crate) fn decode(
    bytes: impl Read,
    mut visit: impl FnMut(Row<'_>) -> Result<()>,
) -> Result<u64> {
    let mut reader = BufReader::new(bytes);
    let mut line = Vec::new();
    let mut rows = 0;
    let mut columns = Vec::new();
    let mut values = Vec::new();
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            return Ok(rows);
        }
        if line.last() == Some(&b'\n') {
            line.pop();
        }
        if line.is_empty() {
            continue;
        }
        let mut object: Map<String, Json> = serde_json::from_slice(&line)?;
        let Some(Json::String(table)) = object.remove("table") else {
            return Err(Error::Invalid("a row names no table".into()));
        };
        let Some(Json::Object(row)) = object.remove("row") else {
            return Err(Error::Invalid(format!("a {table} row is not an object")));
        };
        columns.clear();
        values.clear();
        for (column, value) in row {
            values.push(from_json(&value)?);
            columns.push(column);
        }
        visit(Row {
            table: &table,
            columns: &columns,
            values: &values,
        })?;
        rows += 1;
    }
}

fn column<'a>(row: &'a Row<'_>, name: &str) -> Option<&'a Value> {
    row.columns
        .iter()
        .position(|column| column == name)
        .map(|index| &row.values[index])
}

fn to_json(value: &Value) -> Json {
    match value {
        Value::Null => Json::Null,
        Value::Integer(n) => json!(n),
        Value::Real(x) => json!(x),
        Value::Text(text) => Json::String(text.clone()),
        Value::Blob(bytes) => json!({ "hex": hex(bytes) }),
    }
}

fn from_json(value: &Json) -> Result<Value> {
    Ok(match value {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Integer(i64::from(*b)),
        Json::Number(n) if n.is_f64() => Value::Real(
            n.as_f64()
                .ok_or_else(|| Error::Invalid("a real column is not finite".into()))?,
        ),
        // Integer tokens never pass through f64, which would round values
        // outside SQLite's range instead of refusing them.
        Json::Number(n) => Value::Integer(
            n.as_i64()
                .ok_or_else(|| Error::Invalid(format!("{n} is outside SQLite's integer range")))?,
        ),
        Json::String(text) => Value::Text(text.clone()),
        Json::Object(object) if object.len() == 1 => match object.get("hex") {
            Some(Json::String(hex)) => Value::Blob(unhex(hex)?),
            _ => return Err(Error::Invalid(format!("unknown column value {value}"))),
        },
        _ => return Err(Error::Invalid(format!("unknown column value {value}"))),
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str) -> Result<Vec<u8>> {
    if !text.is_ascii() || !text.len().is_multiple_of(2) {
        return Err(Error::Invalid(format!("bad hex {text}")));
    }
    (0..text.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&text[i..i + 2], 16)
                .map_err(|_| Error::Invalid(format!("bad hex {text}")))
        })
        .collect()
}

/// Hashes and counts what it writes, so an encoded batch is never read back
/// to be hashed.
struct Hashing<W> {
    inner: W,
    digest: blake3::Hasher,
    length: u64,
}

impl<W: Write> Write for Hashing<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(bytes)?;
        self.digest.update(&bytes[..n]);
        self.length += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests;
