//! Extract preserved query and authored products for the finite register conversion.
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if let [mode, register, config, host] = arguments.as_slice()
        && mode == "restore"
    {
        let config = polylogue_sinex::Config::parse(&std::fs::read_to_string(config)?)?;
        let authentication = match &config.password_file {
            Some(path) => polylogue_sinex::Authentication::Password(
                std::fs::read_to_string(path)?.trim_end().to_owned(),
            ),
            None => polylogue_sinex::Authentication::Anonymous,
        };
        let sinex = polylogue_sinex::Sinex::connect(
            &config,
            &authentication,
            host.to_str().ok_or("host is not UTF-8")?,
        )?;
        let count = polylogue_sinex::line_population::restore_current(Path::new(register), &sinex)?;
        println!(
            "{}",
            serde_json::json!({"submitted_current_products":count})
        );
        return Ok(());
    }
    let [store, producer, output] = arguments.as_slice() else {
        return Err("usage: preserved-line-executions PRESERVED_STORE PRODUCER NEW_JSONL; or restore CURRENT_REGISTER CURRENT_SINEX_CONFIG HOST".into());
    };
    let producer = producer.to_str().ok_or("producer is not UTF-8")?;
    let path = Path::new(output);
    let parent = path.parent().ok_or("output needs a parent")?;
    if parent
        .canonicalize()?
        .starts_with(Path::new(store).canonicalize()?)
    {
        return Err("extracted products must be outside the preserved store".into());
    }
    let staged = tempfile::NamedTempFile::new_in(parent)?;
    let mut file = staged.as_file();
    let count = polylogue_sinex::line_population::extract(Path::new(store), producer, &mut file)?;
    file.sync_all()?;
    staged.persist_noclobber(path)?;
    std::fs::File::open(parent)?.sync_all()?;
    println!(
        "{}",
        serde_json::json!({"preserved_products":count,"output":output.to_string_lossy()})
    );
    Ok(())
}
