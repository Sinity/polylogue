//! Finite conversion of a preserved register, before rebuilding the current runtime.
//! Its source and the original producer spool remain preserved separately.
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    let [source, destination, pending @ ..] = arguments.as_slice() else {
        return Err("usage: convert-line-executions PRESERVED_REGISTER NEW_REGISTER_OR_PRODUCTS [PRESERVED_QUERY_JSONL]".into());
    };
    let changed = polylogue_store::query_objects::line_population::convert_preserved(
        Path::new(source),
        Path::new(destination),
        |connection| {
            use std::io::BufRead;
            for path in pending {
                let input = std::io::BufReader::new(std::fs::File::open(path)?);
                for line in input.lines() {
                    let records: Vec<serde_json::Value> = serde_json::from_str(&line?)?;
                    polylogue_store::query_objects::line_population::stage_products(
                        connection, &records,
                    )?;
                }
            }
            Ok(())
        },
    )?;
    println!(
        "{}",
        serde_json::json!({"conversion":changed,"destination":destination.to_string_lossy()})
    );
    Ok(())
}
