//! The Sinex backend in the daemon (architecture 7.1), built with the
//! `sinex` feature. `$XDG_CONFIG_HOME/polylogue/sinex.toml` chooses the mode;
//! without it the archive is standalone and nothing here runs.
//!
//! In `primary`: stored blobs are also read from Sinex's content store; an
//! empty local archive is first restored from Sinex (the archive the file
//! names), after which the loop builds `index.db` from the stored bytes; and
//! a publisher sends Sinex whatever it does not hold yet, once a minute.
use polylogue_engine::Config;
use polylogue_sinex::{Config as SinexConfig, Mode, Sinex};
use polylogue_store::Archive;
use std::thread::JoinHandle;
use std::time::Duration;
use tokio::sync::oneshot;

/// How often the publisher asks Sinex what it holds and sends the rest.
const PUBLISH_EVERY: Duration = Duration::from_secs(60);

/// Owns the optional publisher and its connection. A stop drops the current
/// async pass; frames without a committed acknowledgment remain unknown and
/// are checked through `have` on the next pass, using their stable frame keys.
pub struct Publisher {
    stop: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Publisher {
    fn spawn<F, Work>(runtime: tokio::runtime::Runtime, work: F) -> std::io::Result<Self>
    where
        F: FnOnce() -> Work + Send + 'static,
        Work: std::future::Future<Output = ()> + 'static,
    {
        // Construct the pass on its owning thread: its canonical SQLite
        // snapshot may be borrowed across awaits without being Sync.
        let (stop, stopped) = oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("polylogue-sinex".into())
            .spawn(move || {
                runtime.block_on(async move {
                    tokio::select! {
                        biased;
                        _ = stopped => eprintln!(
                            "polylogue: Sinex publisher stopped; unacknowledged frames will be checked on restart"
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
                .map_err(|_| std::io::Error::other("Sinex publisher panicked"))
        })
    }
}

impl Drop for Publisher {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            eprintln!("polylogue: Sinex shutdown: {error}");
        }
    }
}

/// Resolves operator configuration only at the production entry point.
pub fn configured() -> crate::daemon::service::Backing {
    configured_at(&SinexConfig::path())
}

pub fn configured_at(path: &std::path::Path) -> crate::daemon::service::Backing {
    use crate::daemon::service::Backing;
    match SinexConfig::load_from(path) {
        Ok(Some(config)) if config.mode == Mode::Primary => Backing::Sinex {
            config,
            path: path.to_owned(),
        },
        Ok(_) => Backing::Standalone,
        Err(error) => Backing::Blocked {
            mode: None,
            path: path.to_owned(),
            error: error.to_string(),
        },
    }
}

pub fn start(
    archive: &Archive,
    config: &mut Config,
    sinex: SinexConfig,
) -> std::io::Result<Publisher> {
    config.blob_roots.push(sinex.store.clone());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let empty = polylogue_store::sinex_rows::Published::open(&archive.root)
        .and_then(|p| p.inventory().map(|u| u.is_empty()))
        .map_err(|e| std::io::Error::other(format!("local Sinex inventory: {e}")))?;
    if empty && let Some(id) = &sinex.archive {
        let restored = runtime.block_on(async {
            let s = Sinex::connect(&sinex).await?;
            s.restore(archive, id).await
        });
        match restored {
            Ok(r) => eprintln!(
                "polylogue: restored {} units ({} rows) of archive {id} from Sinex",
                r.units, r.rows
            ),
            Err(e) => {
                return Err(std::io::Error::other(format!(
                    "restoring archive {id} from Sinex failed: {e}"
                )));
            }
        }
    }
    let root = archive.root.clone();
    Publisher::spawn(runtime, move || async move {
        let mut connected: Option<Sinex> = None;
        let mut retirement = None;
        loop {
            if connected.is_none() {
                match Sinex::connect(&sinex).await {
                    Ok(s) => connected = Some(s),
                    Err(e) => eprintln!("polylogue: Sinex: {e}"),
                }
            }
            if let Some(s) = connected.as_mut() {
                match s.publish(&root).await {
                    Ok(done) => {
                        if retirement != Some(done.physical_retirement)
                            && done.physical_retirement
                                != polylogue_sinex::PhysicalRetirement::NotRequested
                        {
                            eprintln!(
                                "polylogue: Sinex row-batch retirement: {}",
                                done.physical_retirement
                            );
                        }
                        retirement = Some(done.physical_retirement);
                    }
                    Err(e) => {
                        eprintln!("polylogue: publishing to Sinex: {e}");
                        connected = None;
                    }
                }
            }
            tokio::time::sleep(PUBLISH_EVERY).await;
        }
    })
}

#[cfg(test)]
mod tests;
