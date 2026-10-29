//! The derived completion snapshot, consumed without opening an archive/socket.
use crate::daemon::warm::Completion;
use serde::{Deserialize, Serialize};
use std::{
    fs, io,
    path::{Path, PathBuf},
};

const MAGIC: &[u8; 8] = b"PLCAT001";
const HEADER: usize = 40;

#[derive(Serialize, Deserialize)]
struct Metadata {
    archive: String,
    position: String,
    written_at: i64,
}

pub fn path(root: &Path) -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_default()
                .join(".cache")
        });
    base.join("polylogue")
        .join(crate::runtime_dir::archive_key(root))
        .join("catalog")
}

pub fn publish(root: &Path, position: &str, data: &Completion, now: i64) -> io::Result<()> {
    write_at(
        &path(root),
        &crate::runtime_dir::archive_key(root),
        position,
        data,
        now,
    )
}

fn write_at(
    path: &Path,
    archive: &str,
    position: &str,
    data: &Completion,
    now: i64,
) -> io::Result<()> {
    let parts = [
        serde_json::to_vec(&Metadata {
            archive: archive.into(),
            position: position.into(),
            written_at: now,
        })?,
        serde_json::to_vec(&data.catalog)?,
        serde_json::to_vec(&(&data.values, &data.sessions))?,
    ];
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    let mut offset = HEADER as u64;
    for part in &parts {
        bytes.extend_from_slice(&offset.to_le_bytes());
        offset += part.len() as u64;
    }
    bytes.extend_from_slice(&offset.to_le_bytes());
    for part in parts {
        bytes.extend(part);
    }
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("completion catalog has no parent"))?;
    fs::create_dir_all(parent)?;
    polylogue_model::atomic_write::write_atomic(path, &bytes, None)
}

pub fn read(root: &Path) -> Option<Completion> {
    read_at(&path(root), &crate::runtime_dir::archive_key(root))
        .ok()
        .flatten()
}

fn read_at(path: &Path, archive: &str) -> io::Result<Option<Completion>> {
    let bytes = fs::read(path)?;
    if bytes.len() < HEADER || &bytes[..8] != MAGIC {
        return Ok(None);
    }
    let offsets: Vec<usize> = bytes[8..HEADER]
        .chunks_exact(8)
        .map(|word| {
            usize::try_from(u64::from_le_bytes(
                word.try_into().expect("eight-byte word"),
            ))
            .ok()
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| io::Error::other("completion offsets cannot be represented"))?;
    if offsets[0] != HEADER
        || offsets[3] != bytes.len()
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Ok(None);
    }
    let metadata: Metadata = serde_json::from_slice(&bytes[offsets[0]..offsets[1]])?;
    if metadata.archive != archive {
        return Ok(None);
    }
    let catalog = serde_json::from_slice(&bytes[offsets[1]..offsets[2]])?;
    let (values, sessions) = serde_json::from_slice(&bytes[offsets[2]..offsets[3]])?;
    Ok(Some(Completion {
        catalog,
        values,
        sessions,
    }))
}

#[cfg(test)]
mod tests;
