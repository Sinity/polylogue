//! Checks WAL files before the first SQLite connection opens, so a damaged or
//! unexpected WAL fails with `RecoveryRequired` instead of being replayed. This
//! is a check, not a recovery engine. The caller holds the archive lock.
use super::{Error, Result, owned_metadata, sibling};
use polylogue_model::{DurableTier, RecoveredWal};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
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
                "WAL present but the main database is missing or empty",
                None,
            ));
        }
        if owned_metadata(&sibling(&main, "-shm"))?.is_none() {
            return Err(required(&wal, "WAL present without its -shm file", None));
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
    scan(path, cancelled, None).map(|_| ())
}

/// Like `inspect`, but accepts and reports a tail of frames left over from the
/// previous WAL generation. Only copy recovery uses this.
pub(super) fn recovery_tail(
    path: &Path,
    cancelled: &mut impl FnMut() -> bool,
    tier: DurableTier,
) -> Result<Option<RecoveredWal>> {
    scan(path, cancelled, Some(tier))
}
fn scan(
    path: &Path,
    cancelled: &mut impl FnMut() -> bool,
    recover: Option<DurableTier>,
) -> Result<Option<RecoveredWal>> {
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
    // One page buffer, reused for every frame.
    let mut page = vec![0; size as usize];
    let mut position = 32u64;
    let mut frame = 0u64;
    let mut committed = 0u64;
    let mut tail: Option<([u8; 8], u64)> = None;
    let mut previous_state = None;
    while position < length {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        frame += 1;
        if length - position < 24 + u64::from(size) {
            return Err(required(path, "WAL ends in a partial frame", Some(frame)));
        }
        let mut frame_header = [0; 24];
        read(&mut file, &mut frame_header, path, Some(frame))?;
        read(&mut file, &mut page, path, Some(frame))?;
        if frame_header[8..16] != header[16..24] {
            if recover.is_none() {
                return Err(required(
                    path,
                    "WAL ends with frames from a previous generation",
                    Some(frame),
                ));
            }
            let salt: [u8; 8] = frame_header[8..16].try_into().unwrap();
            if tail.is_none() {
                if committed != frame - 1
                    || committed == 0
                    || word(&salt[..4]).wrapping_add(1) != word(&header[16..20])
                {
                    return Err(required(
                        path,
                        "unsupported WAL generation boundary",
                        Some(frame),
                    ));
                }
                tail = Some((salt, frame));
            }
            if tail.as_ref().unwrap().0 != salt || word(&frame_header[..4]) == 0 {
                return Err(required(
                    path,
                    "mixed or malformed previous WAL generation",
                    Some(frame),
                ));
            }
            let recorded = [word(&frame_header[16..20]), word(&frame_header[20..24])];
            // The first old frame cannot be checked because the frames before
            // it were overwritten; its recorded checksum seeds the later ones.
            if let Some(previous) = previous_state {
                let computed = checksum(
                    &page,
                    little,
                    checksum(&frame_header[..8], little, previous),
                );
                if computed != recorded {
                    return Err(required(
                        path,
                        "previous WAL tail checksum mismatch",
                        Some(frame),
                    ));
                }
            }
            previous_state = Some(recorded);
            position += 24 + u64::from(size);
            continue;
        }
        if tail.is_some() {
            return Err(required(
                path,
                "current generation after previous WAL generation",
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
        if word(&frame_header[4..8]) != 0 {
            committed = frame;
        }
        position += 24 + u64::from(size);
    }
    if cancelled() {
        return Err(Error::Cancelled);
    }
    if file.metadata()?.len() != length {
        return Err(required(path, "WAL changed during inspection", None));
    }
    Ok(tail.map(|(_, first)| RecoveredWal {
        tier: recover.unwrap(),
        committed_frame: committed,
        first_previous_generation_frame: first,
    }))
}

/// The schema version as of the last committed WAL frame, read without letting
/// SQLite touch the files. The caller has already checked the WAL.
pub(super) fn schema_version(main: &Path) -> Result<u32> {
    let mut header = [0; 100];
    File::open(main)?.read_exact(&mut header)?;
    if &header[..16] != b"SQLite format 3\0" {
        return Err(Error::Schema(main.into()));
    }
    let mut version = word(&header[60..64]);
    let path = sibling(main, "-wal");
    let Some(meta) = owned_metadata(&path)? else {
        return Ok(version);
    };
    if meta.len() == 0 {
        return Ok(version);
    }
    let mut file = File::open(&path)?;
    let mut wal = [0; 32];
    file.read_exact(&mut wal)?;
    let size = word(&wal[8..12]);
    let mut proposed = version;
    let mut frame = [0; 24];
    while file.stream_position()? < meta.len() {
        file.read_exact(&mut frame)?;
        if frame[8..16] != wal[16..24] {
            break;
        }
        if word(&frame[..4]) == 1 {
            file.read_exact(&mut header)?;
            proposed = word(&header[60..64]);
            file.seek(SeekFrom::Current(i64::from(size) - 100))?;
        } else {
            file.seek(SeekFrom::Current(i64::from(size)))?;
        }
        if word(&frame[4..8]) != 0 {
            version = proposed;
        }
    }
    Ok(version)
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
    fn cancelled_scan_leaves_the_file_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("wal");
        std::fs::write(&path, b"wal bytes").unwrap();
        assert!(matches!(
            inspect(&path, &mut || true),
            Err(Error::Cancelled)
        ));
        assert_eq!(std::fs::read(path).unwrap(), b"wal bytes");
    }
}
