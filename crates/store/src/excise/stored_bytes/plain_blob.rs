use super::*;
use std::fs::File;
use std::io::{Seek, SeekFrom};
use std::os::unix::fs::FileExt;

/// One verified plain blob, isolated from the archive until replacement.
pub(super) struct PlainBlob(File);

impl PlainBlob {
    pub fn load(t: &Transactions<'_, '_>, input: &str, hash: BlobHash) -> Result<Self> {
        let encoding: String = t.archive.query_row(
            "SELECT encoding FROM blob WHERE hash=?1",
            [&hash.0[..]],
            |r| r.get(0),
        )?;
        if encoding != "plain" {
            let format =
                t.archive
                    .query_row("SELECT format FROM input WHERE id=?1", [input], |r| {
                        r.get(0)
                    })?;
            return Err(StoreError::ExcisionFormat {
                input: input.into(),
                format,
                detail: format!("blob encoding {encoding} needs its own record-excision decoder"),
            });
        }
        let mut file = tempfile::tempfile()?;
        crate::blobs::copy_plain(t.archive, t.store, hash, &mut file)?;
        Ok(Self(file))
    }

    pub fn plan(
        &self,
        ranges: &[(usize, usize)],
    ) -> std::result::Result<super::super::record_ranges::Erasure, super::super::record_ranges::Error>
    {
        super::super::record_ranges::plan_file(&self.0, ranges)
    }

    pub fn replace(
        mut self,
        t: &Transactions<'_, '_>,
        hash: &BlobHash,
        ranges: &[(usize, usize)],
    ) -> Result<Option<Gone>> {
        let length = self.0.metadata()?.len();
        let mut buffer = [0; 64 * 1024];
        for &(start, end) in ranges {
            if start > end || end as u64 > length {
                return Err(StoreError::Invalid(
                    "replacement range is outside its blob".into(),
                ));
            }
            let mut at = start;
            while at < end {
                let count = (end - at).min(buffer.len());
                self.0.read_exact_at(&mut buffer[..count], at as u64)?;
                for byte in &mut buffer[..count] {
                    if *byte != b'\n' {
                        *byte = b' ';
                    }
                }
                self.0.write_all_at(&buffer[..count], at as u64)?;
                at += count;
            }
        }
        self.0.seek(SeekFrom::Start(0))?;
        let piece = Piece::prepare_reader(t.store, self.0)?;
        record_replacement(t, hash, piece)
    }
}
