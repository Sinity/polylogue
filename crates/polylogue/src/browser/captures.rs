//! Durable append-only capture snapshots. The lock names the stable location,
//! including while a legacy single-envelope file is atomically converted.
use super::write_durable;
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
    let mut prefix = vec![0; MAGIC.len()];
    let read = file.read(&mut prefix)?;
    if read == 0 {
        file.write_all(MAGIC)?;
    } else if read < MAGIC.len() && MAGIC.starts_with(&prefix[..read]) {
        return Err(invalid("incomplete snapshot log prefix"));
    } else if read > 0 && &prefix[..read] != MAGIC {
        file.rewind()?;
        let mut old = Vec::new();
        file.read_to_end(&mut old)?;
        serde_json::from_slice::<serde_json::Value>(&old).map_err(invalid)?;
        if old == bytes {
            return Ok(false);
        }
        let mut converted = MAGIC.to_vec();
        converted.extend(Header::of(&old).encode());
        converted.extend(old);
        write_durable(path, &converted)?;
        file = open(path)?;
    }
    let submitted = Header::of(bytes);
    let length = file.metadata()?.len();
    let mut position = MAGIC.len() as u64;
    while position < length {
        file.seek(SeekFrom::Start(position))?;
        let mut header = [0; HEADER_LEN];
        match file.read_exact(&mut header) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                return Err(invalid("incomplete snapshot header"));
            }
            Err(error) => return Err(error),
        }
        let header = Header::decode(&header).map_err(invalid)?;
        let end = position
            .checked_add(HEADER_LEN as u64)
            .and_then(|start| start.checked_add(header.length))
            .ok_or_else(|| invalid("snapshot range overflow"))?;
        if end > length {
            return Err(invalid("incomplete snapshot payload"));
        }
        if header == submitted {
            let mut previous = vec![0; bytes.len()];
            file.read_exact(&mut previous)?;
            if previous != bytes {
                return Err(invalid("snapshot digest differs from stored bytes"));
            }
            return Ok(false);
        }
        position = end;
    }
    file.seek(SeekFrom::End(0))?;
    file.write_all(&submitted.encode())?;
    file.write_all(bytes)?;
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
