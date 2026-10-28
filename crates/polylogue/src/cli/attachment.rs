//! `attachment --out` writes the file only after every byte range and the
//! whole BLAKE3 check out. The temporary file is private and removed on failure.
use super::*;
use base64::Engine as _;

pub(crate) fn output(values: &Values, data: &Value, json: bool) -> ExitCode {
    if !values.flag("reveal") {
        if json {
            println!("{}", polylogue_render::terminal::json(data, false));
        } else {
            print!("{}", print::json(data));
        }
        return ExitCode::SUCCESS;
    }
    match write(values, data) {
        Ok(path) => {
            if json {
                println!("{}", serde_json::json!({"written":path}));
            }
            ExitCode::SUCCESS
        }
        Err(e) => fail(json, "attachment_output", &e, 1),
    }
}

fn write(values: &Values, data: &Value) -> Result<PathBuf, String> {
    crate::daemon::attachment::check_cli()?;
    let path = PathBuf::from(values.text("out").ok_or("--out is required")?);
    let directory = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(directory).map_err(|e| e.to_string())?;
    let command = crate::commands::command("attachment").ok_or("attachment command missing")?;
    let Action::Request(build) = command.action else {
        return Err("attachment request missing".into());
    };
    let mut request = build(
        command,
        values,
        &Caller {
            surface: Surface::Cli,
            cwd: Some(current_dir()),
            json: true,
        },
    )?;
    let mut value = data.clone();
    let mut hasher = blake3::Hasher::new();
    let mut offset = 0;
    let mut selected = None;
    loop {
        let page: crate::daemon::attachment::BytePage =
            serde_json::from_value(value).map_err(|e| e.to_string())?;
        if page.offset != offset {
            return Err("attachment range is not contiguous".into());
        }
        let identity = (page.input.clone(), page.blob, page.size);
        if selected
            .as_ref()
            .is_some_and(|previous| previous != &identity)
        {
            return Err("attachment changed between ranges".into());
        }
        selected = Some(identity);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(page.content_base64)
            .map_err(|e| e.to_string())?;
        let end = offset
            .checked_add(bytes.len() as u64)
            .ok_or("attachment offset overflow")?;
        if end > page.size || bytes.is_empty() && page.next.is_some() {
            return Err("invalid attachment range length".into());
        }
        temporary.write_all(&bytes).map_err(|e| e.to_string())?;
        hasher.update(&bytes);
        offset = end;
        match page.next {
            Some(next) => {
                let Request::Attachment { cursor, .. } = &mut request else {
                    return Err("attachment request changed".into());
                };
                *cursor = Some(next);
                value = match send_when_ready(&Archive::at(&Archive::default_root()), &request)
                    .map_err(|e| e.to_string())?
                {
                    Response::Ok { data } => data,
                    Response::Error { kind, message, .. } => {
                        return Err(format!("{kind}: {message}"));
                    }
                };
            }
            None => {
                if offset != page.size || hasher.finalize().as_bytes() != &page.blob.0 {
                    return Err("complete attachment length or hash differs".into());
                }
                temporary.as_file().sync_all().map_err(|e| e.to_string())?;
                temporary.persist(&path).map_err(|e| e.to_string())?;
                return Ok(path);
            }
        }
    }
}
