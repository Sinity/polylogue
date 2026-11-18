//! Atomic changes to an installed file and its original bytes.
use std::fs::{self, Permissions};
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

/// The file the original bytes are kept in before the first edit.
pub(crate) fn backup_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_owned();
    name.push(".polylogue-backup");
    path.with_file_name(name)
}

pub(crate) struct File {
    path: PathBuf,
    current: Option<Vec<u8>>,
    backup: Option<Vec<u8>>,
    permissions: Permissions,
    backup_permissions: Option<Permissions>,
}

fn read(path: &Path) -> io::Result<Option<(Vec<u8>, Permissions)>> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(io::Error::other(format!(
            "{} is a symlink to a managed file; configure its source instead",
            path.display()
        ))),
        Ok(metadata) => Ok(Some((fs::read(path)?, metadata.permissions()))),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

impl File {
    pub(crate) fn read(path: &Path) -> io::Result<Self> {
        let current = read(path)?;
        let backup = read(&backup_path(path))?;
        let permissions = current
            .as_ref()
            .map(|(_, permissions)| permissions.clone())
            .unwrap_or_else(|| Permissions::from_mode(0o600));
        let backup_permissions = backup.as_ref().map(|(_, permissions)| permissions.clone());
        Ok(Self {
            path: path.to_path_buf(),
            current: current.map(|(bytes, _)| bytes),
            backup: backup.map(|(bytes, _)| bytes),
            permissions,
            backup_permissions,
        })
    }

    pub(crate) fn current(&self) -> Option<&[u8]> {
        self.current.as_deref()
    }

    pub(crate) fn backup(&self) -> Option<&[u8]> {
        self.backup.as_deref()
    }

    pub(crate) fn write(&self, bytes: &[u8]) -> io::Result<()> {
        if self.current() == Some(bytes) {
            return Ok(());
        }
        let parent = self
            .path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        if let Some(original) = self.current()
            && self.backup.is_none()
        {
            polylogue_model::atomic_write::write_atomic(
                &backup_path(&self.path),
                original,
                Some(self.permissions.clone()),
            )?;
        }
        polylogue_model::atomic_write::write_atomic(
            &self.path,
            bytes,
            Some(self.permissions.clone()),
        )
    }

    /// The caller checks whether the current file still belongs to its install.
    pub(crate) fn restore(&self) -> io::Result<()> {
        if let Some(original) = self.backup() {
            if self.current() != Some(original) {
                polylogue_model::atomic_write::write_atomic(
                    &self.path,
                    original,
                    self.backup_permissions.clone(),
                )?;
            }
            fs::remove_file(backup_path(&self.path))?;
        } else if self.current.is_some() {
            fs::remove_file(&self.path)?;
        }
        Ok(())
    }
}
