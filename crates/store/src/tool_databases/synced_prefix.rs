use super::*;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom};

/// The marker beside a capture naming its completed prefix. Each append is a
/// segment, and `hash` chains them: a segment's link is the BLAKE3 hash of
/// the previous link followed by the segment's bytes. With the file's facts
/// as they were when the marker was written, a capture unchanged since is
/// trusted without reading it; otherwise the chain is computed again from
/// the segments. The marker always names the complete segment chain.
#[derive(Deserialize, Serialize)]
struct Completed {
    length: u64,
    hash: String,
    segments: Vec<u64>,
    facts: [i128; 5],
}

/// The synced prefix: only bytes up to the length named by the adjacent
/// checkpoint file are input; anything after it is an interrupted write.
pub(super) struct SyncedPrefix {
    facts: Option<FileFacts>,
    marker_facts: Option<FileFacts>,
    /// The last link of the segment chain.
    chain: [u8; 32],
    segments: Vec<u64>,
    length: u64,
}

fn facts_array(f: &FileFacts) -> [i128; 5] {
    [
        i128::from(f.device),
        i128::from(f.inode),
        i128::from(f.size),
        f.mtime_ns,
        f.ctime_ns,
    ]
}

/// The next link of the chain after `bytes`.
fn link(previous: &[u8; 32], bytes: &[u8]) -> blake3::Hasher {
    let mut h = blake3::Hasher::new();
    h.update(previous);
    h.update(bytes);
    h
}

pub(super) fn checkpoint(out: &Path) -> PathBuf {
    let mut name = out.as_os_str().to_owned();
    name.push(".completed");
    name.into()
}

fn invalid(reason: &str) -> ExportError {
    std::io::Error::new(std::io::ErrorKind::InvalidData, reason).into()
}

/// The completed prefix's facts, for a reader whose capture worker has
/// checked this marker. The file may already hold a later, unsynced append.
pub fn completed_facts(out: &Path) -> std::io::Result<Option<FileFacts>> {
    let bytes = match fs::read(checkpoint(out)) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let mark: Completed = serde_json::from_slice(&bytes)?;
    let number = |i| u64::try_from(mark.facts[i]).map_err(std::io::Error::other);
    if number(2)? != mark.length {
        return Err(std::io::Error::other(
            "SQLite export marker facts differ from its length",
        ));
    }
    Ok(Some(FileFacts {
        device: number(0)?,
        inode: number(1)?,
        size: mark.length,
        mtime_ns: mark.facts[3],
        ctime_ns: mark.facts[4],
    }))
}

impl SyncedPrefix {
    /// Validate before truncating: an altered committed prefix is never repaired
    /// by deleting bytes. Existing bytes require a completed-prefix marker.
    pub fn open(
        out: &Path,
        progress: &mut dyn FnMut() -> Result<(), ExportError>,
    ) -> Result<Self, ExportError> {
        let mark = match fs::read(checkpoint(out)) {
            Ok(bytes) => {
                Some(serde_json::from_slice::<Completed>(&bytes).map_err(std::io::Error::from)?)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        let mut file = match OpenOptions::new().read(true).write(true).open(out) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && mark.is_none() => {
                return Ok(Self {
                    facts: None,
                    marker_facts: None,
                    chain: [0; 32],
                    segments: Vec::new(),
                    length: 0,
                });
            }
            Err(e) => return Err(e.into()),
        };
        let physical = file.metadata()?.len();
        let length = mark.as_ref().map_or(physical, |m| m.length);
        if length > physical {
            return Err(invalid(
                "SQLite export is shorter than its completed prefix",
            ));
        }
        let now = facts_array(&FileFacts::of(&file.metadata()?));
        let (chain, segments, rewrite) = if let Some(mark) = &mark {
            let total = mark.segments.iter().try_fold(0_u64, |n, segment| {
                if *segment == 0 {
                    None
                } else {
                    n.checked_add(*segment)
                }
            });
            if total != Some(length) {
                return Err(invalid("SQLite export marker segment lengths differ"));
            }
            let expected = decode(&mark.hash)?;
            if mark.facts == now {
                (expected, mark.segments.clone(), false)
            } else {
                let mut chain = [0; 32];
                for &segment in &mark.segments {
                    let mut h = link(&chain, &[]);
                    read_into(&mut file, segment, &mut h, progress)?;
                    chain = *h.finalize().as_bytes();
                }
                if chain != expected {
                    return Err(invalid("SQLite export completed-prefix hash differs"));
                }
                (chain, mark.segments.clone(), true)
            }
        } else if physical == 0 {
            ([0; 32], Vec::new(), false)
        } else {
            return Err(invalid(
                "SQLite export has bytes but no completed-prefix marker",
            ));
        };
        if physical > length {
            file.set_len(length)?;
            file.sync_all()?;
        }
        let mut synced = Self {
            facts: None,
            marker_facts: None,
            chain,
            segments,
            length,
        };
        if rewrite {
            synced.mark(out, &file)?;
        }
        if mark.is_some() {
            fs::File::open(out.parent().unwrap_or(Path::new(".")))?.sync_all()?;
        }
        synced.facts = Some(FileFacts::of(&file.metadata()?));
        synced.marker_facts = fs::metadata(checkpoint(out))
            .ok()
            .map(|m| FileFacts::of(&m));
        Ok(synced)
    }

    /// What the catalog beside the capture is stamped with: the prefix it
    /// was made from.
    pub fn stamp(&self) -> (u64, [u8; 32]) {
        (self.length, self.chain)
    }

    pub fn current(&self, out: &Path) -> bool {
        let facts = |path: &Path| fs::metadata(path).ok().map(|m| FileFacts::of(&m));
        self.facts == facts(out) && self.marker_facts == facts(&checkpoint(out))
    }

    /// Install only after the caller has strictly replayed the existing bytes.
    pub fn establish(&mut self, out: &Path) -> Result<(), ExportError> {
        if !checkpoint(out).exists() {
            let file = OpenOptions::new().create(true).append(true).open(out)?;
            file.sync_all()?;
            self.mark(out, &file)?;
            self.facts = Some(FileFacts::of(&file.metadata()?));
            self.marker_facts = Some(FileFacts::of(&fs::metadata(checkpoint(out))?));
        }
        Ok(())
    }

    /// Writes the marker for the current prefix, with `file`'s facts now.
    fn mark(&self, out: &Path, file: &fs::File) -> Result<(), ExportError> {
        let bytes = serde_json::to_vec(&Completed {
            length: self.length,
            hash: hex(&self.chain),
            segments: self.segments.clone(),
            facts: facts_array(&FileFacts::of(&file.metadata()?)),
        })
        .map_err(std::io::Error::from)?;
        atomicwrites::AtomicFile::new(checkpoint(out), atomicwrites::AllowOverwrite)
            .write(|file| file.write_all(&bytes))
            .map_err(std::io::Error::from)?;
        Ok(())
    }

    pub fn append(&mut self, out: &Path, staged: &mut fs::File) -> Result<(), ExportError> {
        let mut file = OpenOptions::new().create(true).append(true).open(out)?;
        let mut hash = link(&self.chain, &[]);
        staged.seek(SeekFrom::Start(0))?;
        let result = (|| {
            let mut buf = [0; 64 * 1024];
            let mut appended = 0;
            loop {
                let n = staged.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                file.write_all(&buf[..n])?;
                hash.update(&buf[..n]);
                appended += n as u64;
                crate::fault::check("tool_export.after_append", "")
                    .map_err(std::io::Error::other)?;
            }
            file.sync_all()?;
            crate::fault::check("tool_export.before_complete", "")
                .map_err(std::io::Error::other)?;
            if appended > 0 {
                self.chain = *hash.finalize().as_bytes();
                self.segments.push(appended);
                self.length = file.metadata()?.len();
            }
            self.mark(out, &file)?;
            // Once the marker is installed, never roll back these bytes, even
            // if its directory sync failed. Recovery rechecks the marker.
            Ok::<_, ExportError>(())
        })();
        match result {
            Ok(()) => {
                self.facts = Some(FileFacts::of(&file.metadata()?));
                self.marker_facts = Some(FileFacts::of(&fs::metadata(checkpoint(out))?));
                Ok(())
            }
            Err(e) => {
                // Re-read the marker, which might already name the new prefix.
                // On any recovery error the Engine refuses this export.
                *self = Self::open(out, &mut || Ok(()))?;
                Err(e)
            }
        }
    }
}

fn hex(bytes: &[u8; 32]) -> String {
    blake3::Hash::from(*bytes).to_hex().to_string()
}

fn decode(hash: &str) -> Result<[u8; 32], ExportError> {
    blake3::Hash::from_hex(hash)
        .map(|h| *h.as_bytes())
        .map_err(|_| invalid("SQLite export marker hash is not hex"))
}

/// Reads the next `length` bytes of `file` into `sink`.
fn read_into(
    file: &mut fs::File,
    length: u64,
    sink: &mut blake3::Hasher,
    progress: &mut dyn FnMut() -> Result<(), ExportError>,
) -> Result<(), ExportError> {
    let mut remaining = length;
    let mut buf = [0; 64 * 1024];
    while remaining > 0 {
        let n = file.read(&mut buf[..remaining.min(64 * 1024) as usize])?;
        if n == 0 {
            return Err(invalid("SQLite export ended inside its completed prefix"));
        }
        sink.update(&buf[..n]);
        remaining -= n as u64;
        progress()?;
    }
    Ok(())
}
