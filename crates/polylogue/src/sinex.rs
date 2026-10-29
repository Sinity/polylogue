//! Primary mode in the daemon, built with the `sinex` feature.
//! `$XDG_CONFIG_HOME/polylogue/sinex.toml` chooses the mode; without it the
//! archive is standalone and nothing here runs.
//!
//! In `primary`, Sinex holds the archive: the daemon connects to it, opens
//! the local copy of what it holds (rebuilding that copy from Sinex when it is
//! missing), and the writer and the loop send everything to Sinex before they
//! write it locally. When Sinex cannot be reached at start, reads use the
//! local copy and writes pause.
use polylogue_sinex::{Authentication, Config as SinexConfig, Sinex};
use polylogue_store::primary::{Held, Part, RegisterChange};
use polylogue_store::{Archive, ArchiveLock, ReadPool, StoreError};
use std::path::Path;
use std::sync::Arc;

pub fn open(
    root: &Path,
    config: &SinexConfig,
    authentication: &Authentication,
    host: &str,
) -> std::io::Result<(Archive, ArchiveLock, Result<Arc<Sinex>, String>)> {
    let opened = |sinex: &dyn polylogue_store::primary::Sinex| {
        Archive::open_primary(root, &config.store, sinex)
            .map_err(|e| std::io::Error::other(format!("opening the archive Sinex holds: {e}")))
    };
    match Sinex::connect(config, authentication, root, host) {
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

    fn held(&self, _: i64) -> polylogue_store::Result<Vec<Held>> {
        self.refused()
    }

    fn send_sessions(&self, _: &ReadPool) -> polylogue_store::Result<()> {
        self.refused()
    }
}
