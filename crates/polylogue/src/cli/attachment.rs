//! `attachment --out` writes the file only after every byte range and the
//! whole BLAKE3 check out. The temporary file is private and removed on failure.
use super::*;
use base64::Engine as _;
use std::io::Write;
use std::path::PathBuf;

pub(crate) fn output(
    values: &command_surface::ParsedCommand,
    answer: &command_surface::Answer<Value>,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    let data = answer
        .value()
        .expect("output handlers receive answers with values");
    let out = match crate::cli::calling_process::options::<crate::cli::calling_process::Attachment>(
        values,
    ) {
        Ok(options) => options.out,
        Err(e) => {
            return fail(startup.output, json, polylogue_model::Code::BadRequest, &e);
        }
    };
    let Some(out) = out else {
        if json {
            println!(
                "{}",
                polylogue_render::terminal::json(
                    &serde_json::to_value(answer).expect("answer serializes"),
                    false
                )
            );
        } else {
            print!("{}", print::json(data, command_surface::TextStyle::Plain));
        }
        return ExitCode::from(answer.exit_code());
    };
    match write(values, PathBuf::from(out), data, startup) {
        Ok(path) => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string(&command_surface::Answer::ok(
                        serde_json::json!({"written":path})
                    ))
                    .expect("answer serializes")
                );
            }
            ExitCode::SUCCESS
        }
        Err(e) => fail(
            startup.output,
            json,
            polylogue_model::Code::AttachmentOutput,
            &e,
        ),
    }
}

fn write(
    values: &command_surface::ParsedCommand,
    path: PathBuf,
    data: &Value,
    startup: &crate::config::Startup,
) -> Result<PathBuf, String> {
    let mut temporary = super::staged_file::StagedFile::new(&path).map_err(|e| e.to_string())?;
    let arguments: serde_json::Map<String, Value> =
        ["message", "block", "attachment", "input", "length"]
            .into_iter()
            .filter_map(|name| {
                values
                    .request
                    .arguments
                    .get(name)
                    .map(|value| (name.to_owned(), value.clone()))
            })
            .collect();
    let mut request = crate::client::request(
        "attachment",
        Value::Object(arguments),
        polylogue_store::user_writes::new_request_id(),
    );
    request.deadline = values.request.deadline;
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
                request.cursor = Some(next);
                value = match crate::client::call_ready(
                    &Archive::at(&startup.config.archive.clone()),
                    &request,
                )
                .map_err(|e| e.to_string())?
                {
                    command_surface::Answer {
                        outcome: command_surface::Outcome::Error(error),
                        ..
                    } => {
                        return Err(format!(
                            "{}: {}",
                            error.code.as_deref().unwrap_or_default(),
                            error.message
                        ));
                    }
                    answer => answer
                        .value()
                        .cloned()
                        .expect("a non-error answer has a value"),
                };
            }
            None => {
                if offset != page.size || hasher.finalize().as_bytes() != &page.blob.0 {
                    return Err("complete attachment length or hash differs".into());
                }
                return temporary.commit().map_err(|e| e.to_string());
            }
        }
    }
}
