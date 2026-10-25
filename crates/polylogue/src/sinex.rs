//! Primary mode in the daemon, built with the `sinex` feature.
//! `$XDG_CONFIG_HOME/polylogue/sinex.toml` chooses the mode; without it the
//! archive is standalone and nothing here runs.
//!
//! In `primary`, Sinex holds the archive: the daemon connects to it, opens
//! the local copy of what it holds (rebuilding that copy from Sinex when it is
//! missing), and the writer and the loop send everything to Sinex before they
//! write it locally. When Sinex cannot be reached at start, reads use the
//! local copy and writes pause.
use crate::daemon::service::ArchiveMode;
use polylogue_sinex::{Config as SinexConfig, Mode, Sinex};
use polylogue_store::primary::{Held, Part, RegisterChange};
use polylogue_store::{Archive, ArchiveLock, ReadPool, StoreError};
use std::path::Path;
use std::sync::Arc;

/// Resolves operator configuration only at the production entry point.
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

/// The archive at `root` that Sinex holds, with the connection to it, or
/// with why Sinex cannot be reached. Without Sinex a missing local copy
/// cannot be rebuilt, and opening fails.
pub fn open(
    root: &Path,
    config: &SinexConfig,
) -> std::io::Result<(Archive, ArchiveLock, Result<Arc<Sinex>, String>)> {
    let opened = |sinex: &dyn polylogue_store::primary::Sinex| {
        Archive::open_primary(root, &config.store, sinex)
            .map_err(|e| std::io::Error::other(format!("opening the archive Sinex holds: {e}")))
    };
    match Sinex::connect(config, root) {
        Ok(sinex) => {
            let sinex = Arc::new(sinex);
            let (archive, lock) = opened(sinex.as_ref())?;
            Ok((archive, lock, Ok(sinex)))
        }
        Err(error) => {
            let reason = error.to_string();
            let (archive, lock) = opened(&Offline(reason.clone()))?;
            Ok((archive, lock, Err(reason)))
        }
    }
}

/// Sinex while it cannot be reached: everything it is asked refuses.
struct Offline(String);

impl Offline {
    fn refused<T>(&self) -> polylogue_store::Result<T> {
        Err(StoreError::Sinex(format!(
            "unreachable, and this archive has no local copy of what it holds: {}",
            self.0
        )))
    }
}

impl polylogue_store::primary::Sinex for Offline {
    fn send_part(&self, _: &Part, _: &mut dyn std::io::Read) -> polylogue_store::Result<String> {
        self.refused()
    }

    fn send_change(&self, _: &RegisterChange) -> polylogue_store::Result<String> {
        self.refused()
    }

    fn excise(&self, _: &[String]) -> polylogue_store::Result<()> {
        self.refused()
    }

    fn held(&self, _: i64) -> polylogue_store::Result<Vec<Held>> {
        self.refused()
    }

    fn send_sessions(&self, _: &ReadPool) -> polylogue_store::Result<()> {
        self.refused()
    }
}
