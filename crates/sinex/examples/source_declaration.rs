//! Export the exact immutable adapter artifact for owner configuration review.
fn describe(declaration: &sinex_model::SourceDeclaration) -> Result<(), serde_json::Error> {
    eprintln!(
        "schema-source: {}",
        serde_json::to_string(&serde_json::json!({
            "source":declaration.source,"fingerprint":declaration.fingerprint(),
        }))?
    );
    for kind in &declaration.kinds {
        eprintln!(
            "schema-kind: {}",
            serde_json::to_string(&serde_json::json!({
                "source":declaration.source,"kind":kind.kind,"definition_hash":kind.definition_hash(),
            }))?
        );
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let declaration = polylogue_sinex::kinds::source_declaration();
    serde_json::to_writer_pretty(std::io::stdout().lock(), &declaration)?;
    describe(&declaration)?;
    if let Some(path) = std::env::args_os().nth(1) {
        let prior: sinex_model::SourceDeclaration =
            serde_json::from_reader(std::fs::File::open(path)?)?;
        describe(&prior)?;
    }
    Ok(())
}
