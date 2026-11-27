//! Each conversation's captures, appended to one log file and synced. A
//! lock file beside the log serializes writers.
use polylogue_model::capture_log::{HEADER_LEN, Header, MAGIC};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

/// Appends one capture; `false` when the log already ends with it.
pub(super) fn append(path: &Path, bytes: &[u8]) -> io::Result<bool> {
    let dir = path
        .parent()
        .ok_or_else(|| io::Error::other("capture has no parent"))?;
    fs::create_dir_all(dir)?;
    // The provider directory must also survive a power loss.
    if let Some(spool) = dir.parent() {
        File::open(spool)?.sync_all()?;
    }
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
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "capture file is not a capture log".to_string(),
        ));
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
        let header = Header::decode(&header)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        let end = position
            .checked_add(HEADER_LEN as u64)
            .and_then(|start| start.checked_add(header.length))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "capture range overflow".to_string(),
                )
            })?;
        if end > length {
            complete_tail(&mut file, position, &[&encoded, bytes])?;
            return synced(&file, dir);
        }
        if header == submitted {
            let mut previous = vec![0; bytes.len()];
            file.read_exact(&mut previous)?;
            if previous != bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "capture hash differs from the stored bytes".to_string(),
                ));
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
/// is the only check available. Compare before writing and append only a suffix.
fn complete_tail(file: &mut File, start: u64, parts: &[&[u8]]) -> io::Result<()> {
    let length = file.metadata()?.len().checked_sub(start).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "capture tail starts beyond the end of the file".to_string(),
        )
    })?;
    let expected: usize = parts.iter().map(|part| part.len()).sum();
    if length > expected as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "existing tail exceeds submitted frame".to_string(),
        ));
    }
    file.seek(SeekFrom::Start(start))?;
    let mut remaining = length as usize;
    let mut buffer = [0; 8192];
    for part in parts {
        let take = remaining.min(part.len());
        for chunk in part[..take].chunks(buffer.len()) {
            file.read_exact(&mut buffer[..chunk.len()])?;
            if &buffer[..chunk.len()] != chunk {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "existing tail differs from submitted frame".to_string(),
                ));
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

/// Stores the supplied capture bytes before acknowledging them.
pub(super) fn store(
    host: &super::BrowserHost,
    bytes: &[u8],
) -> Result<command_surface::Answer<serde_json::Value>, crate::daemon::Failure> {
    use crate::daemon::Failure;
    use polylogue_model::{JsonExt as _, Reason};
    use serde_json::{Value, json};
    let value: Value =
        serde_json::from_slice(bytes).map_err(|_| Failure::refusal(Reason::CaptureNotJson {}))?;
    if !value.is_object() {
        return Err(Failure::refusal(Reason::CaptureNotJson {}));
    }
    let identity = |key| {
        value
            .pointer(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
    };
    let (Some(provider), Some(conversation)) = (
        identity("/session/provider"),
        identity("/session/provider_session_id"),
    ) else {
        return Err(Failure::refusal(Reason::CaptureNamesNoConversation {}));
    };
    let raw = value
        .object("raw_provider_payload")
        .is_some_and(|o| !o.is_empty());
    let turns = value
        .pointer("/session/turns")
        .and_then(Value::as_array)
        .is_some_and(|t| !t.is_empty());
    if !raw && !turns {
        return Err(Failure::refusal(Reason::CaptureHasNoConversation {}));
    }
    let new = append(&host.capture_log(provider, conversation), bytes).map_err(Failure::io)?;
    Ok(command_surface::Answer::ok(
        json!({"provider": provider, "conversation": conversation,
        "hash": blake3::hash(bytes).to_hex().as_str(), "new": new}),
    ))
}

#[cfg(test)]
mod tests;
