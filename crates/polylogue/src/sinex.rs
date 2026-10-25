//! Primary mode in the daemon, built with the
//! `sinex` feature. `$XDG_CONFIG_HOME/polylogue/sinex.toml` chooses the mode;
//! without it the archive is standalone and nothing here runs.
//!
//! In `primary`: stored blobs are also read from Sinex's content store; an
//! empty local archive is first restored from Sinex (the archive the file
//! names), after which the loop builds `index.db` from the stored bytes; the
//! archive is marked as held by Sinex, so excision refuses; and a thread
//! sends Sinex whatever it does not hold yet, once a minute.
use polylogue_engine::Config;
use polylogue_sinex::{Config as SinexConfig, Mode, Sinex};
use polylogue_store::Archive;
use std::thread::JoinHandle;
use std::time::Duration;
use tokio::sync::oneshot;

/// How often the thread asks Sinex what it holds and sends the rest.
const SEND_EVERY: Duration = Duration::from_secs(60);

/// Owns the thread that sends to Sinex, and its connection. A stop drops the current
/// pass; what Sinex has not confirmed is asked about again through `have`
/// on the next pass, under the same names.
pub struct Sending {
    stop: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Sending {
    fn spawn<F, Work>(runtime: tokio::runtime::Runtime, work: F) -> std::io::Result<Self>
    where
        F: FnOnce() -> Work + Send + 'static,
        Work: std::future::Future<Output = ()> + 'static,
    {
        // Construct the pass on its own thread: its SQLite read transaction is
        // borrowed across awaits and is not Sync.
        let (stop, stopped) = oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("polylogue-sinex".into())
            .spawn(move || {
                runtime.block_on(async move {
                    tokio::select! {
                        biased;
                        _ = stopped => eprintln!(
                            "polylogue: Sinex sending stopped; what Sinex has not confirmed is checked on restart"
                        ),
                        _ = work() => {},
                    }
                });
            })?;
        Ok(Self {
            stop: Some(stop),
            thread: Some(thread),
        })
    }

    pub fn shutdown(&mut self) -> std::io::Result<()> {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        self.thread.take().map_or(Ok(()), |thread| {
            thread
                .join()
                .map_err(|_| std::io::Error::other("the Sinex thread panicked"))
        })
    }
}

impl Drop for Sending {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            eprintln!("polylogue: Sinex shutdown: {error}");
        }
    }
}

/// Resolves operator configuration only at the production entry point.
pub fn configured() -> crate::daemon::service::ArchiveMode {
    configured_at(&SinexConfig::path())
}

pub fn configured_at(path: &std::path::Path) -> crate::daemon::service::ArchiveMode {
    use crate::daemon::service::ArchiveMode;
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

pub fn start(
    archive: &Archive,
    config: &mut Config,
    sinex: SinexConfig,
) -> std::io::Result<Sending> {
    config.blob_roots.push(sinex.store.clone());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let empty = polylogue_store::row_batches::Snapshot::open(&archive.root)
        .and_then(|snapshot| snapshot.batches(None, 1))
        .map_err(|e| std::io::Error::other(format!("reading the local archive: {e}")))?
        .is_empty();
    if empty && let Some(id) = &sinex.archive {
        let restored = runtime.block_on(async {
            let s = Sinex::connect(&sinex).await?;
            s.restore(archive, id).await
        });
        match restored {
            Ok(r) => eprintln!(
                "polylogue: restored {} batches ({} rows) of archive {id} from Sinex",
                r.batches, r.rows
            ),
            Err(e) => {
                return Err(std::io::Error::other(format!(
                    "restoring archive {id} from Sinex failed: {e}"
                )));
            }
        }
    }
    polylogue_store::row_batches::mark_on_sinex(archive)
        .map_err(|e| std::io::Error::other(format!("marking the archive as held by Sinex: {e}")))?;
    let root = archive.root.clone();
    Sending::spawn(runtime, move || async move {
        let mut connected: Option<Sinex> = None;
        loop {
            if connected.is_none() {
                match Sinex::connect(&sinex).await {
                    Ok(s) => connected = Some(s),
                    Err(e) => eprintln!("polylogue: Sinex: {e}"),
                }
            }
            if let Some(s) = connected.as_mut()
                && let Err(e) = s.send_missing(&root).await
            {
                eprintln!("polylogue: sending to Sinex: {e}");
                connected = None;
            }
            tokio::time::sleep(SEND_EVERY).await;
        }
    })
}

#[cfg(test)]
mod tests;
