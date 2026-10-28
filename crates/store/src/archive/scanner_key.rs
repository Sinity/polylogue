use crate::{Archive, Result, StoreError};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

impl Archive {
    /// The archive's persistent credential fingerprint key. Keep `scanner.key`
    /// with archive backups; index rebuilds must use the same key.
    pub fn scanner_key(&self) -> Result<[u8; 32]> {
        let path = self.root.join("scanner.key");
        match OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&path)
        {
            Ok(mut file) => {
                let metadata = file.metadata()?;
                if !metadata.is_file()
                    || metadata.len() != 32
                    || metadata.permissions().mode() & 0o7777 != 0o600
                {
                    return Err(StoreError::Invalid(
                        "scanner.key must be a regular file of 32 bytes with mode 0600".into(),
                    ));
                }
                let mut key = [0; 32];
                file.read_exact(&mut key)?;
                Ok(key)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let mut key = [0; 32];
                File::open("/dev/urandom")?.read_exact(&mut key)?;
                let mut file = tempfile::NamedTempFile::new_in(&self.root)?;
                file.as_file()
                    .set_permissions(std::fs::Permissions::from_mode(0o600))?;
                file.write_all(&key)?;
                file.as_file().sync_all()?;
                // Install complete bytes without replacing a concurrent creator's key.
                match file.persist_noclobber(&path) {
                    Ok(_) => {
                        File::open(&self.root)?.sync_all()?;
                        Ok(key)
                    }
                    Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
                        self.scanner_key()
                    }
                    Err(error) => Err(error.error.into()),
                }
            }
            Err(error) => Err(error.into()),
        }
    }
}

#[cfg(test)]
mod tests;
