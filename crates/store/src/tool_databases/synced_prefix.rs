use super::*;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom};

#[derive(Deserialize, Serialize)]
pub(super) struct Completed {
    pub length: u64,
    pub hash: String,
}

/// The synced prefix: only bytes up to the length named by the adjacent
/// checkpoint file are input; anything after it is an interrupted write.
pub(super) struct SyncedPrefix {
    facts: Option<FileFacts>,
    marker_facts: Option<FileFacts>,
    hash: super::prefix_hash::PrefixHash,
    length: u64,
}

type SavedFacts = (u64, u64, u64, String, String);

fn saved_facts(facts: Option<FileFacts>) -> Option<SavedFacts> {
    facts.map(|f| {
        (
            f.device,
            f.inode,
            f.size,
            f.mtime_ns.to_string(),
            f.ctime_ns.to_string(),
        )
    })
}

#[derive(Serialize, Deserialize)]
pub(super) struct SavedPrefix {
    facts: Option<SavedFacts>,
    marker_facts: Option<SavedFacts>,
    hash: super::prefix_hash::PrefixHash,
}

pub(super) fn checkpoint(out: &Path) -> PathBuf {
    let mut name = out.as_os_str().to_owned();
    name.push(".completed");
    name.into()
}

fn invalid(reason: &str) -> ExportError {
    std::io::Error::new(std::io::ErrorKind::InvalidData, reason).into()
}

impl SyncedPrefix {
    pub fn saved(&self) -> SavedPrefix {
        SavedPrefix {
            facts: saved_facts(self.facts),
            marker_facts: saved_facts(self.marker_facts),
            hash: self.hash.clone(),
        }
    }

    /// A catalog is reusable only for the same completed bytes and file facts.
    pub fn resume(out: &Path, saved: SavedPrefix) -> Result<Option<Self>, ExportError> {
        let facts = fs::metadata(out).ok().map(|m| FileFacts::of(&m));
        let marker_facts = fs::metadata(checkpoint(out))
            .ok()
            .map(|m| FileFacts::of(&m));
        if !saved.hash.valid()
            || facts.is_none()
            || marker_facts.is_none()
            || saved.facts != saved_facts(facts)
            || saved.marker_facts != saved_facts(marker_facts)
        {
            return Ok(None);
        }
        let marker: Completed =
            serde_json::from_slice(&fs::read(checkpoint(out))?).map_err(std::io::Error::from)?;
        if marker.length != saved.hash.len()
            || marker.hash != saved.hash.finalize().to_hex().as_str()
            || facts.is_none_or(|f| f.size != marker.length)
        {
            return Ok(None);
        }
        Ok(Some(Self {
            facts,
            marker_facts,
            hash: saved.hash,
            length: marker.length,
        }))
    }
    /// Validate before truncating: an altered committed prefix is never repaired
    /// by deleting bytes. An export without a marker is validated by the catalog loader.
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
        let mut hash = super::prefix_hash::PrefixHash::default();
        let mut file = match OpenOptions::new().read(true).write(true).open(out) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && mark.is_none() => {
                return Ok(Self {
                    facts: None,
                    marker_facts: None,
                    hash,
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
        let mut remaining = length;
        let mut buf = [0; 64 * 1024];
        while remaining > 0 {
            let n = file.read(&mut buf[..remaining.min(64 * 1024) as usize])?;
            if n == 0 {
                return Err(invalid("SQLite export ended inside its completed prefix"));
            }
            hash.update(&buf[..n]);
            remaining -= n as u64;
            progress()?;
        }
        if mark
            .as_ref()
            .is_some_and(|m| m.hash != hash.finalize().to_hex().as_str())
        {
            return Err(invalid("SQLite export completed-prefix hash differs"));
        }
        if physical > length {
            file.set_len(length)?;
            file.sync_all()?;
        }
        if mark.is_some() {
            fs::File::open(out.parent().unwrap_or(Path::new(".")))?.sync_all()?;
        }
        Ok(Self {
            facts: Some(FileFacts::of(&file.metadata()?)),
            marker_facts: fs::metadata(checkpoint(out))
                .ok()
                .map(|m| FileFacts::of(&m)),
            hash,
            length,
        })
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
            self.mark(out, self.length, &self.hash)?;
            self.facts = Some(FileFacts::of(&file.metadata()?));
            self.marker_facts = Some(FileFacts::of(&fs::metadata(checkpoint(out))?));
        }
        Ok(())
    }

    fn mark(
        &self,
        out: &Path,
        length: u64,
        hash: &super::prefix_hash::PrefixHash,
    ) -> Result<(), ExportError> {
        let bytes = serde_json::to_vec(&Completed {
            length,
            hash: hash.finalize().to_hex().to_string(),
        })
        .map_err(std::io::Error::from)?;
        atomicwrites::AtomicFile::new(checkpoint(out), atomicwrites::AllowOverwrite)
            .write(|file| file.write_all(&bytes))
            .map_err(std::io::Error::from)?;
        Ok(())
    }

    pub fn append(&mut self, out: &Path, staged: &mut fs::File) -> Result<(), ExportError> {
        let mut file = OpenOptions::new().create(true).append(true).open(out)?;
        let mut hash = self.hash.clone();
        staged.seek(SeekFrom::Start(0))?;
        let result = (|| {
            let mut buf = [0; 64 * 1024];
            loop {
                let n = staged.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                file.write_all(&buf[..n])?;
                hash.update(&buf[..n]);
                crate::fault::check("tool_export.after_append", "")
                    .map_err(std::io::Error::other)?;
            }
            file.sync_all()?;
            crate::fault::check("tool_export.before_complete", "")
                .map_err(std::io::Error::other)?;
            let length = file.metadata()?.len();
            self.mark(out, length, &hash)?;
            // Once the marker is installed, never roll back these bytes, even
            // if its directory sync failed. Recovery rechecks the marker.
            Ok::<_, ExportError>(length)
        })();
        match result {
            Ok(length) => {
                self.length = length;
                self.hash = hash;
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
