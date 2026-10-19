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
use std::time::Duration;

/// How often the publisher asks Sinex what it holds and sends the rest.
const PUBLISH_EVERY: Duration = Duration::from_secs(60);

pub fn start(archive: &Archive, config: &mut Config) -> std::io::Result<()> {
    let sinex = match SinexConfig::load() {
        Ok(Some(c)) if c.mode == Mode::Primary => c,
        Ok(_) => return Ok(()),
        Err(e) => {
            // A broken file never switches the mode silently: publishing
            // waits until it parses (`status` shows the error).
            eprintln!("polylogue: Sinex publishing paused: {e}");
            return Ok(());
        }
    };
    config.blob_roots.push(sinex.store.clone());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let empty = polylogue_store::rows::Published::open(&archive.root)
        .and_then(|p| p.units(0))
        .map(|u| u.is_empty())
        .unwrap_or(false);
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
            Err(e) => eprintln!("polylogue: restoring archive {id} from Sinex failed: {e}"),
        }
    }
    let root = archive.root.clone();
    std::thread::Builder::new()
        .name("polylogue-sinex".into())
        .spawn(move || {
            runtime.block_on(async {
                let mut connected: Option<Sinex> = None;
                loop {
                    if connected.is_none() {
                        match Sinex::connect(&sinex).await {
                            Ok(s) => connected = Some(s),
                            Err(e) => eprintln!("polylogue: Sinex: {e}"),
                        }
                    }
                    if let Some(s) = connected.as_mut()
                        && let Err(e) = s.publish(&root).await
                    {
                        eprintln!("polylogue: publishing to Sinex: {e}");
                        connected = None;
                    }
                    tokio::time::sleep(PUBLISH_EVERY).await;
                }
            });
        })?;
    Ok(())
}
