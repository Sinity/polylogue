//! A file output replaces its target only after the caller validates it.
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub(crate) struct StagedFile {
    temporary: tempfile::NamedTempFile,
    target: PathBuf,
    requested: PathBuf,
    permissions: Option<fs::Permissions>,
}

impl StagedFile {
    pub(crate) fn new(path: &Path) -> io::Result<Self> {
        let target = target(path)?;
        let permissions = match fs::metadata(&target) {
            Ok(metadata) => {
                if !metadata.is_file() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "output target is not a regular file",
                    ));
                }
                Some(metadata.permissions())
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error),
        };
        let temporary = tempfile::NamedTempFile::new_in(target.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "output file has no parent")
        })?)?;
        Ok(Self {
            temporary,
            target,
            requested: path.to_path_buf(),
            permissions,
        })
    }

    pub(crate) fn commit(self) -> io::Result<PathBuf> {
        if let Some(permissions) = self.permissions {
            self.temporary.as_file().set_permissions(permissions)?;
        }
        self.temporary.as_file().sync_all()?;
        self.temporary
            .persist(&self.target)
            .map_err(|error| error.error)?;
        Ok(self.requested)
    }
}

impl Write for StagedFile {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.temporary.write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.temporary.flush()
    }
}

fn target(path: &Path) -> io::Result<PathBuf> {
    match fs::canonicalize(path) {
        Ok(target) => Ok(target),
        Err(error) if error.kind() == io::ErrorKind::NotFound => match fs::symlink_metadata(path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                let link = fs::read_link(path)?;
                if link.is_absolute() {
                    target(&link)
                } else {
                    target(&path.parent().unwrap_or_else(|| Path::new(".")).join(link))
                }
            }
            Ok(_) => Err(error),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let name = path.file_name().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "output file has no name")
                })?;
                let parent = path
                    .parent()
                    .filter(|path| !path.as_os_str().is_empty())
                    .unwrap_or_else(|| Path::new("."));
                Ok(fs::canonicalize(parent)?.join(name))
            }
            Err(error) => Err(error),
        },
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests;
