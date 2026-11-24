//! Rebuild `src/encoding/rows.dict` from the synthetic samples.
//! Usage: `cargo run -p polylogue-store --example row_dictionary PATH`.
#[path = "row_dictionary/samples.rs"]
mod samples;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: row_dictionary PATH")?;
    let bytes = samples::dictionary()?;
    std::fs::write(path, &bytes)?;
    println!("{} bytes", bytes.len());
    Ok(())
}
