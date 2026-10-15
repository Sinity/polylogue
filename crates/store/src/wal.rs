//! Conservative cold-start gate, not a replacement WAL recovery engine.
//! Caller owns the archive lock; no SQLite connection has opened yet.
use super::{Error, Result, owned_metadata, sibling};
use std::{
    fs::File,
    io::Read,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub(crate) fn required(file: &Path, reason: &'static str, frame: Option<u64>) -> Error {
    Error::RecoveryRequired {
        file: file.into(),
        reason,
        frame,
    }
}
pub(crate) fn check_cancel(cancelled: &AtomicBool) -> Result<()> {
    if cancelled.load(Ordering::Acquire) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
pub(crate) fn validate(root: &Path, cancelled: &AtomicBool) -> Result<()> {
    for name in ["archive.db", "embeddings.db"] {
        check_cancel(cancelled)?;
        let main = root.join(name);
        let wal = sibling(&main, "-wal");
        let Some(meta) = owned_metadata(&wal)? else {
            continue;
        };
        if meta.len() == 0 {
            continue;
        }
        if owned_metadata(&main)?.is_none_or(|m| m.len() == 0) {
            return Err(required(
                &main,
                "missing or empty main with retained WAL",
                None,
            ));
        }
        if owned_metadata(&sibling(&main, "-shm"))?.is_none() {
            return Err(required(
                &wal,
                "orphan WAL requires controlled recovery",
                None,
            ));
        }
        inspect(&wal, &mut || cancelled.load(Ordering::Acquire))?;
    }
    Ok(())
}
fn word(bytes: &[u8]) -> u32 {
    u32::from_be_bytes(bytes.try_into().expect("four bytes"))
}
fn checksum(bytes: &[u8], little: bool, mut state: [u32; 2]) -> [u32; 2] {
    for pair in bytes.chunks_exact(8) {
        let read = |b: &[u8]| {
            if little {
                u32::from_le_bytes(b.try_into().unwrap())
            } else {
                word(b)
            }
        };
        state[0] = state[0]
            .wrapping_add(read(&pair[..4]))
            .wrapping_add(state[1]);
        state[1] = state[1]
            .wrapping_add(read(&pair[4..]))
            .wrapping_add(state[0]);
    }
    state
}
fn read(file: &mut File, bytes: &mut [u8], path: &Path, frame: Option<u64>) -> Result<()> {
    file.read_exact(bytes).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            required(path, "WAL changed or is incomplete", frame)
        } else {
            Error::Io(e)
        }
    })
}
pub(super) fn inspect(path: &Path, cancelled: &mut impl FnMut() -> bool) -> Result<()> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    if cancelled() {
        return Err(Error::Cancelled);
    }
    if length < 32 {
        return Err(required(path, "incomplete WAL header", None));
    }
    let mut header = [0; 32];
    read(&mut file, &mut header, path, None)?;
    let magic = word(&header[..4]);
    let size = word(&header[8..12]);
    if !matches!(magic, 0x377f0682 | 0x377f0683)
        || word(&header[4..8]) != 3_007_000
        || !(512..=65536).contains(&size)
        || !size.is_power_of_two()
    {
        return Err(required(path, "unsupported or damaged WAL header", None));
    }
    let little = magic & 1 == 0;
    let mut state = checksum(&header[..24], little, [0, 0]);
    if state != [word(&header[24..28]), word(&header[28..32])] {
        return Err(required(path, "WAL header checksum mismatch", None));
    }
    // At most one SQLite page, not an allocation proportional to WAL length.
    let mut page = vec![0; size as usize];
    let mut position = 32u64;
    let mut frame = 0u64;
    while position < length {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        frame += 1;
        if length - position < 24 + u64::from(size) {
            return Err(required(
                path,
                "partial WAL frame; commit history is uncertain",
                Some(frame),
            ));
        }
        let mut frame_header = [0; 24];
        read(&mut file, &mut frame_header, path, Some(frame))?;
        read(&mut file, &mut page, path, Some(frame))?;
        if frame_header[8..16] != header[16..24] {
            return Err(required(
                path,
                "uncertain WAL generation tail (may be legitimate reuse)",
                Some(frame),
            ));
        }
        if word(&frame_header[..4]) == 0 {
            return Err(required(path, "invalid WAL page number", Some(frame)));
        }
        state = checksum(&page, little, checksum(&frame_header[..8], little, state));
        if state != [word(&frame_header[16..20]), word(&frame_header[20..24])] {
            return Err(required(path, "WAL frame checksum mismatch", Some(frame)));
        }
        position += 24 + u64::from(size);
    }
    if cancelled() {
        return Err(Error::Cancelled);
    }
    if file.metadata()?.len() != length {
        return Err(required(
            path,
            "WAL changed during quiescent inspection",
            None,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checksum_uses_declared_word_order_and_wrapping() {
        let bytes = [255; 8];
        assert_eq!(checksum(&bytes, true, [0, 0]), [u32::MAX, u32::MAX - 1]);
        assert_eq!(checksum(&[1, 0, 0, 0, 2, 0, 0, 0], true, [0, 0]), [1, 3]);
        assert_eq!(checksum(&[0, 0, 0, 1, 0, 0, 0, 2], false, [0, 0]), [1, 3]);
    }
    #[test]
    fn cancelled_scan_does_not_touch_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("wal");
        std::fs::write(&path, b"retained bytes").unwrap();
        assert!(matches!(
            inspect(&path, &mut || true),
            Err(Error::Cancelled)
        ));
        assert_eq!(std::fs::read(path).unwrap(), b"retained bytes");
    }
}
