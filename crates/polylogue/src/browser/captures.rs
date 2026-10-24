//! Durable append-only capture snapshots. The lock names the stable location.
use polylogue_model::snapshot_log::{HEADER_LEN, Header, MAGIC};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

pub(super) fn publish(path: &Path, bytes: &[u8]) -> io::Result<bool> {
    let dir = path
        .parent()
        .ok_or_else(|| io::Error::other("capture has no parent"))?;
    fs::create_dir_all(dir)?;
    let lock_path = path.with_extension("lock");
    for target in [path, &lock_path] {
        if fs::symlink_metadata(target).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(io::Error::other("symlink in the spool"));
        }
    }
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .mode(0o600)
        .open(lock_path)?;
    lock.lock()?;
    let mut file = open(path)?;
    let submitted = Header::of(bytes);
    let encoded = submitted.encode();
    let mut prefix = vec![0; MAGIC.len()];
    let read = file.read(&mut prefix)?;
    if read == 0 {
        file.write_all(MAGIC)?;
    } else if read < MAGIC.len() && MAGIC.starts_with(&prefix[..read]) {
        complete_tail(&mut file, 0, &[MAGIC, &encoded, bytes])?;
        return synced(&file, dir);
    } else if &prefix[..read] != MAGIC {
        return Err(invalid("capture file is not a snapshot log"));
    }
    let length = file.metadata()?.len();
    let mut position = MAGIC.len() as u64;
    while position < length {
        file.seek(SeekFrom::Start(position))?;
        let mut header = [0; HEADER_LEN];
        match file.read_exact(&mut header) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                complete_tail(&mut file, position, &[&encoded, bytes])?;
                return synced(&file, dir);
            }
            Err(error) => return Err(error),
        }
        let header = Header::decode(&header).map_err(invalid)?;
        let end = position
            .checked_add(HEADER_LEN as u64)
            .and_then(|start| start.checked_add(header.length))
            .ok_or_else(|| invalid("snapshot range overflow"))?;
        if end > length {
            complete_tail(&mut file, position, &[&encoded, bytes])?;
            return synced(&file, dir);
        }
        if header == submitted {
            let mut previous = vec![0; bytes.len()];
            file.read_exact(&mut previous)?;
            if previous != bytes {
                return Err(invalid("snapshot digest differs from stored bytes"));
            }
            synced(&file, dir)?;
            return Ok(false);
        }
        position = end;
    }
    file.seek(SeekFrom::End(0))?;
    file.write_all(&encoded)?;
    file.write_all(bytes)?;
    synced(&file, dir)
}

/// A tail can contain only common framing, so equality of all existing bytes
/// is the evidence available. Compare before writing and append only a suffix.
fn complete_tail(file: &mut File, start: u64, parts: &[&[u8]]) -> io::Result<()> {
    let length = file
        .metadata()?
        .len()
        .checked_sub(start)
        .ok_or_else(|| invalid("snapshot tail starts beyond EOF"))?;
    let expected: usize = parts.iter().map(|part| part.len()).sum();
    if length > expected as u64 {
        return Err(invalid("existing tail exceeds submitted frame"));
    }
    file.seek(SeekFrom::Start(start))?;
    let mut remaining = length as usize;
    let mut buffer = [0; 8192];
    for part in parts {
        let take = remaining.min(part.len());
        for chunk in part[..take].chunks(buffer.len()) {
            file.read_exact(&mut buffer[..chunk.len()])?;
            if &buffer[..chunk.len()] != chunk {
                return Err(invalid("existing tail differs from submitted frame"));
            }
        }
        remaining -= take;
    }
    file.seek(SeekFrom::End(0))?;
    let mut skip = length as usize;
    for part in parts {
        let before = skip.min(part.len());
        skip -= before;
        file.write_all(&part[before..])?;
    }
    Ok(())
}

fn synced(file: &File, dir: &Path) -> io::Result<bool> {
    file.sync_all()?;
    File::open(dir)?.sync_all()?;
    Ok(true)
}

fn open(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path)
}

fn invalid(error: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
}

#[cfg(test)]
mod tests;
