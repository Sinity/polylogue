use super::*;
use serde::{Deserialize, Serialize};
use std::io::Write;

/// Storage identity: the mode the archive was created in. It survives
/// replacing the derived index, and an archive is opened only in its recorded
/// mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Standalone,
    Primary,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Standalone => "standalone",
            Self::Primary => "primary",
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    mode: Mode,
}

pub(super) fn read(root: &Path) -> Result<Option<Mode>> {
    let path = root.join("meta.json");
    match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice::<Metadata>(&bytes)
            .map(|metadata| Some(metadata.mode))
            .map_err(|error| StoreError::stored_data(format!("{}: {error}", path.display()))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Every archive this build opens is standalone; primary mode is not built
/// in this version (`docs/primary-mode.md`).
const REQUESTED: Mode = Mode::Standalone;

pub(super) fn check(archive: &Archive) -> Result<()> {
    match read(&archive.root)? {
        Some(mode) if mode != REQUESTED => Err(polylogue_model::Reason::ArchiveModeMismatch {
            archive: archive.root.display().to_string(),
            recorded: mode.to_string(),
            requested: REQUESTED.to_string(),
        }
        .into()),
        Some(_) => Ok(()),
        // The index is derived and replaced when unreadable; only stores that
        // hold original bytes or paid vectors mark an older-format archive.
        None if ["archive.db", "embeddings.db", "segments"]
            .iter()
            .any(|name| archive.root.join(name).exists()) =>
        {
            Err(StoreError::OlderFormat {
                database: "archive",
            })
        }
        None => Ok(()),
    }
}

/// The caller holds the archive lock. The identity commits before any store
/// is created, so a crash cannot leave stores without their mode.
pub(super) fn create(archive: &Archive) -> Result<()> {
    if read(&archive.root)?.is_some() {
        return Ok(());
    }
    let mut temporary = tempfile::NamedTempFile::new_in(&archive.root)?;
    serde_json::to_writer(temporary.as_file_mut(), &Metadata { mode: REQUESTED })?;
    temporary.write_all(b"\n")?;
    temporary.as_file().sync_all()?;
    temporary
        .persist_noclobber(archive.root.join("meta.json"))
        .map_err(|error| error.error)?;
    File::open(&archive.root)?.sync_all()?;
    Ok(())
}
