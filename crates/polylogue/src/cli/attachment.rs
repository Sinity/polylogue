//! `attachment:… read --to FILE` writes the file only after every byte range
//! and the whole BLAKE3 check out. The temporary file is private and removed
//! on failure.
use crate::daemon::attachment::BytePage;
use crate::protocol::line::{LineAnswer, StepAnswer, StepOutcome};
use base64::Engine as _;
use command_surface::{Answer, RunnableRequest};
use polylogue_model::Code;
use polylogue_store::Archive;
use serde_json::Value;
use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

/// Whether the line reads an attachment: its selection is one
/// `attachment:` reference and its one step is `read`.
pub(crate) fn reads_attachment(request: &RunnableRequest) -> bool {
    let arguments = &request.arguments;
    arguments["selection"]
        .as_str()
        .and_then(|text| polylogue_model::Reference::parse(text.trim()).ok())
        .is_some_and(|reference| matches!(reference, polylogue_model::Reference::Attachment { .. }))
        && arguments["steps"]
            .as_array()
            .is_some_and(|steps| steps.len() == 1 && steps[0]["verb"] == "read")
}

/// Reads every range of the attachment and writes it to `path`.
pub(crate) fn write_to(
    client: &crate::config::Client,
    request: &RunnableRequest,
    path: &Path,
    json: bool,
) -> ExitCode {
    match write(client, request, path) {
        Ok(written) => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string(&Answer::ok(serde_json::json!({"written":written})))
                        .expect("answer serializes")
                );
            }
            ExitCode::SUCCESS
        }
        Err(Written::Refused(error)) => super::fail_with(client.output, json, &error, &[]),
        Err(Written::Output(message)) => {
            super::fail(client.output, json, Code::AttachmentOutput, &message)
        }
    }
}

enum Written {
    /// The daemon refused a range.
    Refused(command_surface::Failure),
    /// The ranges do not make the attachment, or the file could not be written.
    Output(String),
}

impl From<String> for Written {
    fn from(message: String) -> Self {
        Self::Output(message)
    }
}

/// One range: the line answer's attachment page, or the refusal.
fn range(archive: &Archive, request: &RunnableRequest) -> Result<BytePage, Written> {
    let answer = crate::client::call_ready(archive, request).map_err(|e| e.to_string())?;
    let value: &Value = match &answer.outcome {
        command_surface::Outcome::Error(error) => return Err(Written::Refused(error.clone())),
        _ => answer.value().expect("a non-error answer has a value"),
    };
    let line: LineAnswer = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    let LineAnswer::Steps(steps) = line else {
        return Err("the line returned no attachment".to_owned().into());
    };
    match steps.steps.into_iter().next().map(|step| step.outcome) {
        Some(StepOutcome::Done {
            reply: StepAnswer::Attachment(page),
            ..
        }) => Ok(*page),
        Some(StepOutcome::Failed { error }) => Err(Written::Refused(error)),
        _ => Err("the line returned no attachment".to_owned().into()),
    }
}

fn write(
    client: &crate::config::Client,
    request: &RunnableRequest,
    path: &Path,
) -> Result<std::path::PathBuf, Written> {
    let archive = Archive::at(&client.archive);
    let mut temporary = super::staged_file::StagedFile::new(path).map_err(|e| e.to_string())?;
    let mut request = request.clone();
    let mut hasher = blake3::Hasher::new();
    let mut offset = 0;
    let mut selected = None;
    loop {
        let page = range(&archive, &request)?;
        if page.offset != offset {
            return Err("attachment range is not contiguous".to_owned().into());
        }
        let identity = (page.input.clone(), page.blob, page.size);
        if selected
            .as_ref()
            .is_some_and(|previous| previous != &identity)
        {
            return Err("attachment changed between ranges".to_owned().into());
        }
        selected = Some(identity);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(page.content_base64)
            .map_err(|e| e.to_string())?;
        let end = offset
            .checked_add(bytes.len() as u64)
            .ok_or("attachment offset overflow".to_owned())?;
        if end > page.size || bytes.is_empty() && page.next.is_some() {
            return Err("invalid attachment range length".to_owned().into());
        }
        temporary.write_all(&bytes).map_err(|e| e.to_string())?;
        hasher.update(&bytes);
        offset = end;
        match page.next {
            Some(next) => {
                request.arguments["steps"][0]["cursor"] = Value::String(next);
            }
            None => {
                if offset != page.size || hasher.finalize().as_bytes() != &page.blob.0 {
                    return Err("complete attachment length or hash differs"
                        .to_owned()
                        .into());
                }
                return Ok(temporary.commit().map_err(|e| e.to_string())?);
            }
        }
    }
}
