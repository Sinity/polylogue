//! Primary mode opens its read cache and durable spool using captured startup inputs.
use polylogue_sinex::{Authentication, Config as SinexConfig, Sinex};
use polylogue_store::{Archive, ArchiveLock};
use std::path::Path;
use std::sync::Arc;

pub fn open(
    root: &Path,
    config: &SinexConfig,
    authentication: &Authentication,
    host: &str,
) -> std::io::Result<(Archive, ArchiveLock, Arc<Sinex>)> {
    let (archive, lock) =
        Archive::open_primary(root, &config.store).map_err(std::io::Error::other)?;
    let sinex = Arc::new(Sinex::open(config, authentication, host).map_err(std::io::Error::other)?);
    Ok((archive, lock, sinex))
}

#[cfg(test)]
mod tests;
