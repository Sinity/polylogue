//! Regenerate the public wire schema from the canonical record enum.
use polylogue_model::bundle::Record;
use std::{fs::File, io};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = schemars::schema_for!(Record);
    match std::env::args_os().nth(1) {
        Some(path) => serde_json::to_writer_pretty(File::create(path)?, &schema)?,
        None => serde_json::to_writer_pretty(io::stdout().lock(), &schema)?,
    }
    Ok(())
}
