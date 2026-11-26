//! Logical messages over Chrome's length-prefixed JSON frames. Large
//! messages travel as ordered base64 parts in either direction.

use super::BrowserHost;
use base64::Engine as _;
use command_surface::Answer;
use polylogue_model::Reason;
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{self, Read, Write};
use std::process::ExitCode;
use std::sync::Arc;

const CHUNK: usize = 64 * 1024;
const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

pub fn read_frame(input: &mut impl Read) -> io::Result<Option<Value>> {
    let mut len = [0u8; 4];
    match input.read_exact(&mut len) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    // Memory follows the bytes that arrive, not the length the header claims.
    let len = u64::from(u32::from_ne_bytes(len));
    let mut buf = Vec::new();
    input.take(len).read_to_end(&mut buf)?;
    if buf.len() as u64 != len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("frame ended after {} of {len} bytes", buf.len()),
        ));
    }
    serde_json::from_slice(&buf)
        .map(Some)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn write_frame(out: &mut impl Write, value: &Value) -> io::Result<()> {
    let bytes = serde_json::to_vec(value).map_err(io::Error::other)?;
    let len = u32::try_from(bytes.len()).map_err(io::Error::other)?;
    out.write_all(&len.to_ne_bytes())?;
    out.write_all(&bytes)?;
    out.flush()
}

/// Writes a logical message, splitting its UTF-8 bytes below Chrome's limit.
pub fn write_message(out: &mut impl Write, value: &Value) -> io::Result<()> {
    let bytes = serde_json::to_vec(value).map_err(io::Error::other)?;
    if bytes.len() <= CHUNK {
        return write_frame(out, value);
    }
    let count = bytes.len().div_ceil(CHUNK);
    for (sequence, part) in bytes.chunks(CHUNK).enumerate() {
        write_frame(
            out,
            &json!({"version": 2, "type": "part", "sequence": sequence,
            "data": B64.encode(part), "last": sequence + 1 == count}),
        )?;
    }
    Ok(())
}

fn invalid(problem: impl ToString) -> crate::daemon::Failure {
    crate::daemon::Failure::refusal(Reason::NativeMessageInvalid {
        problem: problem.to_string(),
    })
}

fn version(frame: &Value) -> Result<(), crate::daemon::Failure> {
    if frame["version"] == 2 {
        Ok(())
    } else {
        Err(crate::daemon::Failure::refusal(
            Reason::NativeVersionMismatch {
                version: frame["version"].as_u64(),
            },
        ))
    }
}

fn read_message(input: &mut impl Read) -> Result<Option<Value>, crate::daemon::Failure> {
    let Some(mut frame) = read_frame(input).map_err(invalid)? else {
        return Ok(None);
    };
    version(&frame)?;
    if frame["type"] != "part" {
        return Ok(Some(frame));
    }
    let mut bytes = Vec::new();
    let mut sequence = 0;
    loop {
        version(&frame)?;
        if frame["type"] != "part" || frame["sequence"].as_u64() != Some(sequence) {
            return Err(invalid("parts must arrive in order"));
        }
        let part = frame["data"]
            .as_str()
            .ok_or_else(|| invalid("part needs data"))?;
        bytes.extend(B64.decode(part).map_err(invalid)?);
        match frame["last"].as_bool() {
            Some(true) => break,
            Some(false) => {}
            None => return Err(invalid("part needs last")),
        }
        sequence += 1;
        frame = read_frame(input)
            .map_err(invalid)?
            .ok_or_else(|| invalid("message ended between parts"))?;
    }
    let message = serde_json::from_slice(&bytes).map_err(invalid)?;
    version(&message)?;
    Ok(Some(message))
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Message {
    Call {
        #[serde(flatten)]
        request: command_surface::RunnableRequest,
    },
    Capture {
        capture: String,
    },
    Spooled {
        provider: String,
        ids: Vec<String>,
    },
}

fn answer(host: &BrowserHost, mut message: Value) -> Result<Answer<Value>, crate::daemon::Failure> {
    if let Some(object) = message.as_object_mut() {
        object.remove("version");
    }
    match serde_json::from_value(message).map_err(invalid)? {
        Message::Call { request } => Ok(std::os::unix::net::UnixStream::connect(&host.socket)
            .and_then(|socket| crate::client::Exchange::new(&request).on(socket, |_| Ok(false)))
            .map(|answer| answer.expect("events are consumed until the answer"))
            .unwrap_or_else(|_| {
                crate::daemon::Failure::refusal(Reason::DaemonDidNotAnswer {}).answer()
            })),
        Message::Capture { capture } => super::captures::store(host, capture.as_bytes()),
        Message::Spooled { provider, ids } => {
            let spooled: Vec<_> = ids
                .iter()
                .map(|id| host.capture_log(&provider, id).exists())
                .collect();
            Ok(Answer::ok(
                json!({"provider": provider, "ids": ids, "spooled": spooled}),
            ))
        }
    }
}

/// Serves one logical request and returns the whole shared answer.
pub fn serve(
    host: Arc<BrowserHost>,
    input: &mut impl Read,
    out: &mut impl Write,
) -> io::Result<()> {
    let answer = match read_message(input) {
        Ok(Some(message)) => answer(&host, message).unwrap_or_else(|failure| failure.answer()),
        Ok(None) => return Ok(()),
        Err(failure) => failure.answer(),
    };
    write_message(
        out,
        &json!({"version": 2, "type": "answer", "answer": answer}),
    )
}

/// The command Chrome runs for an extension allowed by the host manifest.
pub fn run(
    config: &polylogue_engine::Config,
    archive: &polylogue_store::Archive,
    _paths: &crate::config::Paths,
) -> ExitCode {
    let host = match BrowserHost::at(config, archive) {
        Ok(host) => Arc::new(host),
        Err(error) => {
            let answer: Answer<Value> = crate::daemon::Failure::io(&error).answer();
            let _ = write_message(
                &mut io::stdout().lock(),
                &json!({"version": 2, "type": "answer", "answer": answer}),
            );
            eprintln!("polylogue internal browser-host: {error}");
            return ExitCode::FAILURE;
        }
    };
    match serve(host, &mut io::stdin().lock(), &mut io::stdout().lock()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("polylogue internal browser-host: {error}");
            ExitCode::FAILURE
        }
    }
}

/// The manifest Chrome reads: this binary, allowing only `extension_id`.
pub fn manifest(binary: &std::path::Path, extension_id: &str) -> Value {
    json!({"name": super::HOST_NAME, "description": "Polylogue browser capture host",
        "path": binary, "type": "stdio",
        "allowed_origins": [format!("chrome-extension://{extension_id}/")]})
}
