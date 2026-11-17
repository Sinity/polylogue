//! Extract preserved query products for the finite register conversion.
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    let [store, producer, output] = arguments.as_slice() else {
        return Err("usage: preserved-line-executions PRESERVED_STORE PRODUCER NEW_JSONL".into());
    };
    let producer = producer.to_str().ok_or("producer is not UTF-8")?;
    let path = Path::new(output);
    let parent = path.parent().ok_or("output needs a parent")?;
    let staged = tempfile::NamedTempFile::new_in(parent)?;
    let mut file = staged.as_file();
    let count = polylogue_sinex::line_population::extract(Path::new(store), producer, &mut file)?;
    file.sync_all()?;
    staged.persist_noclobber(path)?;
    std::fs::File::open(parent)?.sync_all()?;
    println!(
        "{}",
        serde_json::json!({"preserved_query_products":count,"output":output.to_string_lossy()})
    );
    Ok(())
}
