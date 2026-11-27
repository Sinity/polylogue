//! One-time operation. Follows a checked reused index from the earlier cut
//! to the end of the sealed later archive, without scanning local sources or
//! planning unchanged inputs. Both cuts must use the same reading contracts.
//!
//! cargo run -p polylogue-engine --example catch_up_archive_suffix -- <later> <from> <to>

use polylogue_engine::{Config, Engine};
use polylogue_model::WallClock;
use polylogue_store::Archive;
use polylogue_store::archive::writer::Position;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let (Some(root), Some(from), Some(to), None) =
        (args.next(), args.next(), args.next(), args.next())
    else {
        return Err("usage: catch_up_archive_suffix <later sealed copy> <from> <to>".into());
    };
    let root = PathBuf::from(root).canonicalize()?;
    if !root.join("archive.db").is_file() || !root.join("index.db").is_file() {
        return Err("the copy needs its archive and the checked reused index".into());
    }
    let from = Position(from.to_str().ok_or("from is not a position")?.parse()?);
    let to = Position(to.to_str().ok_or("to is not a position")?.parse()?);
    let machine = std::fs::read_to_string("/etc/machine-id")?;
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")?;
    let mut config = Config::new(
        root.clone(),
        Default::default(),
        machine.trim().into(),
        host.trim().into(),
        std::thread::available_parallelism()?.get(),
        WallClock::new(jiff::Timestamp::now),
    )?;
    config.reads_sources = false;
    let (archive, _lock) = Archive::open_or_create(&root)?;
    let mut engine = Engine::new(config, archive)?;
    let pass = engine.catch_up_archive_suffix(from, to)?;
    let written = engine.writer().flush()?;
    println!("{}", serde_json::to_string(&pass)?);
    if !pass.errors.is_empty() || !written.errors.is_empty() {
        return Err("some stored work failed during suffix catch-up".into());
    }
    Ok(())
}
