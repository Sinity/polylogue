//! A derived cache pointer follows the archive's existing scope publication fence.
use super::{CatalogWriter, Metadata};
use crate::{ReadScopeToken, Result, StoreError};
use std::{
    fs,
    os::unix::fs::symlink,
    path::{Path, PathBuf},
};
#[derive(Clone)]
pub struct CatalogPointer {
    path: PathBuf,
}
impl CatalogPointer {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
    /// A rebuild owns fresh database and journal names even when its source
    /// generation is unchanged and an old snapshot still holds the previous WAL.
    pub fn build(
        &self,
        metadata: &Metadata,
        populate: impl FnOnce(&mut CatalogWriter) -> Result<()>,
    ) -> Result<CatalogWriter> {
        let target = self.path.with_file_name(format!(
            "catalog.{}.{}.db",
            blake3::hash(metadata.position.generation.as_bytes()).to_hex(),
            uuid::Uuid::now_v7(),
        ));
        CatalogWriter::build(&target, metadata, populate)
    }
    /// Called synchronously by the native scope owner, never by a commit wake.
    pub fn invalidate(&self) -> Result<()> {
        let previous = fs::read_link(&self.path).ok();
        match fs::remove_file(&self.path) {
            Ok(()) => {
                if let Some(parent) = self.path.parent() {
                    fs::File::open(parent)?.sync_all()?;
                }
                if let Some(previous) = previous {
                    self.retire(&previous);
                }
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }
    /// Only pointer activation holds the native fence; database work precedes it.
    pub fn activate(&self, writer: &CatalogWriter, token: &ReadScopeToken) -> Result<()> {
        let target = &writer.path;
        token.with_current(|| {
            if target.parent() != self.path.parent() || !writer.is_current_file() {
                return Err(StoreError::ScopeChanged);
            }
            let name = target.file_name().expect("materialization file name");
            let previous = fs::read_link(&self.path).ok();
            if previous.as_deref() == Some(Path::new(name)) {
                return Ok(());
            }
            let parent = self
                .path
                .parent()
                .ok_or_else(|| StoreError::Invalid("completion catalog has no parent".into()))?;
            let temporary = parent.join(format!(".catalog-pointer-{}", uuid::Uuid::now_v7()));
            symlink(name, &temporary)?;
            let result = (|| {
                fs::rename(&temporary, &self.path)?;
                fs::File::open(parent)?.sync_all()?;
                Ok(())
            })();
            if result.is_err() {
                let _ = fs::remove_file(&temporary);
            } else if let Some(previous) = previous {
                self.retire(&previous);
            }
            result
        })
    }
    // The pointer owns only these cache-local namespaces. Unlinking a retired
    // namespace leaves pinned SQLite descriptors valid; its pointer is revoked.
    // Cleanup cannot change the outcome of an already published replacement.
    fn retire(&self, target: &Path) {
        let Some(name) = target.to_str() else { return };
        if target.components().count() != 1
            || !matches!(
                target.components().next(),
                Some(std::path::Component::Normal(_))
            )
            || !name.starts_with("catalog.")
            || !name.ends_with(".db")
        {
            return;
        }
        let Some(parent) = self.path.parent() else {
            return;
        };
        for suffix in ["", "-wal", "-shm"] {
            let _ = fs::remove_file(parent.join(format!("{name}{suffix}")));
        }
    }
}
#[cfg(test)]
mod tests;
