use std::path::PathBuf;

pub(super) struct Selection {
    pub origin: String,
    pub threads: usize,
    pub files: Vec<PathBuf>,
}

pub(super) fn selection(args: impl Iterator<Item = String>) -> Result<Selection, String> {
    let mut args = args;
    let origin = args.next().ok_or("expected origin name")?;
    if polylogue_sources::origin::by_name(&origin).is_none() {
        return Err("unknown origin".into());
    }
    let mut threads = 1;
    let mut files = Vec::new();
    while let Some(arg) = args.next() {
        if arg == "--threads" {
            threads = args
                .next()
                .and_then(|n| n.parse::<usize>().ok())
                .filter(|n| *n > 0)
                .ok_or("--threads requires a positive integer")?;
        } else if let Some(path) = arg.strip_prefix('@') {
            let list = std::fs::read_to_string(path)
                .map_err(|e| format!("path list could not be read: {e}"))?;
            files.extend(
                list.lines()
                    .filter(|line| !line.trim().is_empty())
                    .map(|line| match line.split_once(' ') {
                        Some((size, path))
                            if !size.is_empty() && size.bytes().all(|c| c.is_ascii_digit()) =>
                        {
                            PathBuf::from(path)
                        }
                        _ => PathBuf::from(line),
                    }),
            );
        } else if arg.starts_with("--") {
            return Err("unknown option".into());
        } else {
            files.push(PathBuf::from(arg));
        }
    }
    if files.is_empty() {
        return Err("select at least one input file".into());
    }
    Ok(Selection {
        origin,
        threads,
        files,
    })
}
