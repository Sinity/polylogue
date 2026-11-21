//! Old compressed contents remain owned by one model transaction until commit.
//! A large retraction spills rather than retaining an archive-sized body map.
use super::{Group, Text, TextBatch};
use crate::Result;
use rusqlite::Connection;
use std::io::{Read, Seek, Write};

pub(crate) struct RemovedTexts {
    file: std::fs::File,
    entries: u64,
}
impl RemovedTexts {
    pub(in crate::text_rows) fn new() -> Result<Self> {
        Ok(Self {
            file: tempfile::tempfile()?,
            entries: 0,
        })
    }
    pub(in crate::text_rows) fn push(&mut self, conn: &Connection, id: i64) -> Result<()> {
        let (body, start, length): (Vec<u8>, i64, i64) = conn.prepare_cached(
            "SELECT g.body,t.start,t.length FROM text t JOIN text_group g ON g.id=t.group_id WHERE t.id=?1"
        )?.query_row([id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        for value in [id, start, length] {
            self.file.write_all(&value.to_le_bytes())?;
        }
        self.file.write_all(&(body.len() as u64).to_le_bytes())?;
        self.file.write_all(&body)?;
        self.entries += 1;
        Ok(())
    }
    pub(super) fn send(mut self, mut send: impl FnMut(TextBatch) -> Result<()>) -> Result<()> {
        self.file.rewind()?;
        for _ in 0..self.entries {
            crate::cancellation::check_current()?;
            let mut header = [0u8; 32];
            self.file.read_exact(&mut header)?;
            let number = |offset: usize| {
                i64::from_le_bytes(header[offset..offset + 8].try_into().expect("fixed header"))
            };
            let length = u64::from_le_bytes(header[24..32].try_into().expect("fixed header"));
            let length = usize::try_from(length).map_err(crate::StoreError::invalid)?;
            let mut body = vec![0; length];
            self.file.read_exact(&mut body)?;
            send(TextBatch {
                texts: vec![Text {
                    id: number(0),
                    bits: 0,
                    present: 0,
                    start: number(8),
                    length: number(16),
                    body: None,
                }],
                groups: vec![Group {
                    body,
                    members: vec![0],
                }],
            })?;
        }
        Ok(())
    }
}
