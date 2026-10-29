//! Primary mode opens its read cache and durable producer spool before connecting.
use crate::daemon::service::ArchiveMode;
use polylogue_sinex::{Config as SinexConfig, Mode, Sinex};
use polylogue_store::{Archive, ArchiveLock};
use std::path::Path;
use std::sync::Arc;

pub fn configured() -> ArchiveMode {
    configured_at(&SinexConfig::path())
}

pub fn configured_at(path: &Path) -> ArchiveMode {
    match SinexConfig::load_from(path) {
        Ok(Some(config)) if config.mode == Mode::Primary => ArchiveMode::Primary {
            config,
            path: path.to_owned(),
        },
        Ok(_) => ArchiveMode::Standalone,
        Err(error) => ArchiveMode::Blocked {
            configured: None,
            path: path.to_owned(),
            error: error.to_string(),
        },
    }
}

pub fn open(
    root: &Path,
    config: &SinexConfig,
) -> std::io::Result<(Archive, ArchiveLock, Arc<Sinex>)> {
    let (archive, lock) =
        Archive::open_primary(root, &config.store).map_err(std::io::Error::other)?;
    let sinex = Arc::new(Sinex::open(config).map_err(std::io::Error::other)?);
    Ok((archive, lock, sinex))
}
