use super::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
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
        erasures: &super::erasures::Erasures,
        hash: &[u8],
    ) -> std::result::Result<(usize, usize), super::super::record_ranges::Error> {
        erasures.plan_blob(hash, &self.0)
    }

    pub fn find(&self, needle: &[u8], visit: impl FnMut(usize, usize) -> Result<()>) -> Result<()> {
        let mut reader = self.0.try_clone()?;
        reader.seek(SeekFrom::Start(0))?;
        literal_ranges(reader, needle, 64 * 1024, visit)
    }

    pub fn replace(
        mut self,
        t: &Transactions<'_, '_>,
        hash: &BlobHash,
        visit_ranges: impl FnOnce(&mut dyn FnMut(usize, usize) -> Result<()>) -> Result<()>,
    ) -> Result<Option<Gone>> {
        let length = self.0.metadata()?.len();
        let mut buffer = [0; 64 * 1024];
        visit_ranges(&mut |start, end| {
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
            Ok(())
        })?;
        self.0.seek(SeekFrom::Start(0))?;
        let piece = Piece::prepare_reader(t.store, self.0)?;
        record_replacement(t, hash, piece)
    }
}

/// Retain only the suffix that could start a match in the next chunk.
fn literal_ranges(
    mut reader: impl Read,
    needle: &[u8],
    chunk: usize,
    mut visit: impl FnMut(usize, usize) -> Result<()>,
) -> Result<()> {
    if needle.is_empty() {
        return Ok(());
    }
    let finder = memchr::memmem::Finder::new(needle);
    let mut bytes = Vec::new();
    let mut base = 0;
    let mut next = 0usize;
    loop {
        let carry = bytes.len();
        bytes.resize(carry + chunk, 0);
        let n = loop {
            match reader.read(&mut bytes[carry..]) {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                result => break result?,
            }
        };
        bytes.truncate(carry + n);
        if n == 0 {
            break;
        }
        let skip = next.saturating_sub(base).min(bytes.len());
        for at in finder.find_iter(&bytes[skip..]) {
            let start = base + skip + at;
            next = start + needle.len();
            visit(start, next)?;
        }
        let keep = bytes.len().min(needle.len() - 1);
        let discard = bytes.len() - keep;
        bytes.copy_within(discard.., 0);
        bytes.truncate(keep);
        base += discard;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
