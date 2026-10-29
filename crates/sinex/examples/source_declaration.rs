//! Export the exact immutable adapter artifact for owner configuration review.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let declaration = polylogue_sinex::kinds::source_declaration();
    serde_json::to_writer_pretty(std::io::stdout().lock(), &declaration)?;
    eprintln!("source fingerprint: {}", declaration.fingerprint());
    Ok(())
}
