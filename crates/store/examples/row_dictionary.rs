//! Build the derived-row dictionary from neutral model examples.
#[path = "row_dictionary/samples.rs"]
mod samples;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let first = args.next().ok_or("usage: row_dictionary [--check] PATH")?;
    let check = first == "--check";
    let path = if check {
        args.next().ok_or("--check needs a path")?
    } else {
        first
    };
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let rows = samples::rows()?;
    let bytes = zstd::dict::from_samples(&rows, 8192)?;
    if check {
        if std::fs::read(&path)? != bytes {
            return Err("dictionary differs from the locked builder output".into());
        }
    } else {
        std::fs::write(&path, &bytes)?;
    }
    println!(
        "{}",
        serde_json::json!({
            "samples": rows.len(),
            "sample_bytes": rows.iter().map(Vec::len).sum::<usize>(),
            "dictionary_bytes": bytes.len(),
            "blake3": blake3::hash(&bytes).to_hex().to_string(),
            "zstd": zstd::zstd_safe::version_string()
        })
    );
    Ok(())
}
