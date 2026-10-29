//! A derived cache pointer follows the archive's existing scope publication fence.
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
    pub fn generation_path(&self, generation: &str) -> PathBuf {
        self.path.with_file_name(format!(
            "catalog.{}.db",
            blake3::hash(generation.as_bytes()).to_hex()
        ))
    }
    /// Called synchronously by the native scope owner, never by a commit wake.
    pub fn invalidate(&self) -> Result<()> {
        match fs::remove_file(&self.path) {
            Ok(()) => {
                if let Some(parent) = self.path.parent() {
                    fs::File::open(parent)?.sync_all()?;
                }
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }
    /// Only pointer activation holds the native fence; database work precedes it.
    pub fn activate(&self, generation: &str, token: &ReadScopeToken) -> Result<()> {
        let target = self.generation_path(generation);
        token.with_current(|| {
            fs::metadata(&target)?;
            let name = target.file_name().expect("generation file name");
            if fs::read_link(&self.path).ok().as_deref() == Some(Path::new(name)) {
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
            }
            result
        })
    }
}
#[cfg(test)]
mod tests;
