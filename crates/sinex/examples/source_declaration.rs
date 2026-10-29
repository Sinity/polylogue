//! Export the exact immutable adapter artifact for owner configuration review.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let producer = std::env::args()
        .nth(1)
        .filter(|value| !value.is_empty())
        .ok_or("pass the configured producer identity")?;
    let declaration = polylogue_sinex::kinds::source_declaration(&producer);
    serde_json::to_writer_pretty(std::io::stdout().lock(), &declaration)?;
    eprintln!("source fingerprint: {}", declaration.fingerprint());
    Ok(())
}
