//! `session-read`: streams the JSON session document in text frames and ends
//! with a frame that reports whether the document is complete.
use crate::{
    reference_read::{FRAME, frame, text_frames},
    session_document,
};
use polylogue_model::{
    SessionId,
    types::{ErrorKind, Failure},
};
use polylogue_store::ReadHandle;
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    os::unix::net::UnixStream,
    path::Path,
    sync::Mutex,
    time::{Duration, Instant},
};
const TIME: Duration = Duration::from_secs(30);
fn failed(kind: ErrorKind, message: impl ToString) -> Value {
    json!({"type":"end","outcome":"error","error":Failure::new(kind,message.to_string())})
}
fn completion(status: &Value, missing: &[String]) -> Value {
    let complete = status["outcome"] == "ok";
    // Diagnostics such as a counted text repair can accompany a complete read;
    // they are not missing content.
    let mut end = json!({"type":"end","outcome":status["outcome"],
        "canonical_complete":complete,"missing":missing,"notes":status["notes"],
        "unknown":status["unknown"]});
    if let Some(diagnostics) = status.get("diagnostics") {
        end["diagnostics"] = diagnostics.clone();
    }
    end
}
pub(crate) fn serve(
    stream: &mut UnixStream,
    request: &Value,
    reader: &Mutex<ReadHandle>,
) -> io::Result<()> {
    stream.set_write_timeout(Some(TIME))?;
    let result = (|| -> Result<Value, Value> {
        if request.as_object().is_none_or(|o| {
            o.len() != 3
                || o.keys()
                    .any(|k| !matches!(k.as_str(), "command" | "reference" | "format"))
        }) || request["command"] != "session-read"
            || request["format"] != "json"
        {
            return Err(failed(
                ErrorKind::InvalidQuery,
                "session-read needs a reference and format json",
            ));
        }
        let reference = request["reference"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| failed(ErrorKind::InvalidQuery, "missing session reference"))?;
        let id =
            SessionId::from_stored(reference).map_err(|e| failed(ErrorKind::InvalidQuery, e))?;
        let session = reader
            .lock()
            .map_err(|_| failed(ErrorKind::Io, "reader lock poisoned"))?
            .selected_session(&id, 65536)
            .map_err(|e| failed(ErrorKind::Io, e))?
            .ok_or_else(|| failed(ErrorKind::NotFound, "no such session"))?;
        let h = session.header().map_err(|e| failed(ErrorKind::Io, e))?;
        let deadline = Instant::now() + TIME;
        // IDs cannot be masked without breaking references, so refuse them.
        crate::output_json::check_reference(h.id.as_str(), deadline)
            .map_err(|e| failed(ErrorKind::NotYetAvailable, e))?;
        let native =
            polylogue_model::NativeId::from_wire(&h.native_id, h.native_id_encoding.as_deref())
                .map_err(|_| {
                    failed(
                        ErrorKind::NotYetAvailable,
                        "invalid native Session identity",
                    )
                })?;
        crate::output_json::check_native_identity(&native, deadline)
            .map_err(|e| failed(ErrorKind::NotYetAvailable, e))?;
        frame(stream,json!({"type":"session","id":h.id.as_str(),"origin":h.origin.as_str(),"native_id":h.native_id,"native_id_encoding":h.native_id_encoding,"format":"polylogue-session","version":1})).map_err(|e|failed(ErrorKind::Io,e))?;
        let cancellation = stream.try_clone().map_err(|e| failed(ErrorKind::Io, e))?;
        let summary = text_frames(stream, |out| {
            session_document::write(&session, out, deadline, &mut || peer_gone(&cancellation))
        })
        .map_err(|e| {
            failed(
                if e.kind() == io::ErrorKind::TimedOut {
                    ErrorKind::Timeout
                } else if e.kind() == io::ErrorKind::InvalidData {
                    ErrorKind::NotYetAvailable
                } else {
                    ErrorKind::Io
                },
                e,
            )
        })?;
        let mut end = completion(&summary.status, &summary.missing);
        end["messages"] = json!(summary.messages);
        end["view"] = json!(format!(
            "{}:{}",
            session.position().generation,
            session.position().counter
        ));
        end["up_to_date"] = json!(summary.up_to_date);
        if let Some(recovery) = summary.recovery {
            end["recovery"] = json!(recovery);
        }
        Ok(end)
    })();
    frame(
        stream,
        match result {
            Ok(end) | Err(end) => end,
        },
    )
}
// POLLHUP fires only when the peer fully closes, so a client that shuts down
// its write half after the request is not treated as gone.
pub(crate) fn peer_gone(stream: &UnixStream) -> bool {
    use rustix::event::{PollFd, PollFlags, Timespec, poll};
    let mut fds = [PollFd::new(stream, PollFlags::HUP | PollFlags::ERR)];
    match poll(
        &mut fds,
        Some(&Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        }),
    ) {
        Ok(_) => fds[0]
            .revents()
            .intersects(PollFlags::HUP | PollFlags::ERR | PollFlags::NVAL),
        Err(error) if error == rustix::io::Errno::INTR => false,
        Err(_) => true,
    }
}

pub fn request(socket: &Path, reference: &str, out: &mut impl Write) -> io::Result<Value> {
    if reference.is_empty() || reference.len() > 16384 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "session reference is empty or longer than 16 KiB",
        ));
    }
    let mut stream = UnixStream::connect(socket)?;
    stream.set_write_timeout(Some(TIME))?;
    frame(
        &mut stream,
        json!({"command":"session-read","reference":reference,"format":"json"}),
    )?;
    decode(stream, reference, out, Instant::now() + TIME)
}
/// Renders the same document as the JSON read, buffered in memory up to
/// 16 MiB.
pub fn request_terminal(socket: &Path, reference: &str, out: &mut impl Write) -> io::Result<Value> {
    struct Document(Vec<u8>);
    impl Write for Document {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > (16 * 1024 * 1024usize).saturating_sub(self.0.len()) {
                return Err(io::Error::other(
                    "session too large for terminal output (over 16 MiB); use --format json",
                ));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut bytes = Document(Vec::new());
    let end = request(socket, reference, &mut bytes)?;
    if end["outcome"] == "error" {
        return Ok(end);
    }
    let document = polylogue_model::RawJson::parse(&bytes.0).map_err(io::Error::other)?;
    render_terminal(&document, out)?;
    Ok(end)
}
fn render_terminal(document: &polylogue_model::RawJson, out: &mut impl Write) -> io::Result<()> {
    use crate::reference_read::terminal_text;
    use polylogue_model::RawJson;
    fn text<'a>(value: &'a RawJson, key: &str) -> io::Result<Option<&'a str>> {
        // Display fields are plain strings. Refuse one with lone surrogates
        // rather than show a repaired copy.
        let Some(raw) = value.get(key).and_then(RawJson::native) else {
            return Ok(None);
        };
        if raw.encoding().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "terminal typed string contains non-scalar units",
            ));
        }
        Ok(Some(raw.as_str()))
    }
    if let Some(session) = document.get("session") {
        for key in ["id", "display_title"] {
            if let Some(text) = text(session, key)? {
                terminal_text(out, text, false)?;
                out.write_all(b"\n")?;
            }
        }
    }
    for message in document
        .get("messages")
        .and_then(RawJson::as_array)
        .into_iter()
        .flatten()
    {
        out.write_all(b"\n")?;
        for key in ["id", "role", "at"] {
            if let Some(text) = text(message, key)? {
                terminal_text(out, text, false)?;
                out.write_all(b"  ")?;
            }
        }
        out.write_all(b"\n")?;
        for block in message
            .get("blocks")
            .and_then(RawJson::as_array)
            .into_iter()
            .flatten()
        {
            let kind = text(block, "kind")?;
            if kind == Some("tool_result") {
                out.write_all(b"[tool_result ")?;
                terminal_text(out, text(block, "outcome")?.unwrap_or("unknown"), false)?;
                if let Some(call) = block.get("call_id") {
                    out.write_all(b" ")?;
                    if let Some(call) = text(block, "call_id")? {
                        terminal_text(out, call, false)?;
                    } else {
                        let canonical =
                            String::from_utf8(call.canonical()).map_err(io::Error::other)?;
                        terminal_text(out, &canonical, false)?;
                    }
                }
                out.write_all(b"]\n")?;
            } else if kind == Some("thinking") {
                out.write_all(b"[thinking]\n")?;
            }
            if let Some(body) = text(block, "text")? {
                terminal_text(out, body, true)?;
                out.write_all(b"\n")?;
            }
            if kind == Some("tool_call") {
                terminal_text(out, text(block, "tool")?.unwrap_or("tool"), false)?;
                out.write_all(b" ")?;
                if let Some(input) = block.get("input") {
                    let canonical =
                        String::from_utf8(input.canonical()).map_err(io::Error::other)?;
                    terminal_text(out, &canonical, true)?;
                }
                out.write_all(b"\n")?;
            }
            if let Some(data) = block.get("data") {
                let canonical = String::from_utf8(data.canonical()).map_err(io::Error::other)?;
                terminal_text(out, &canonical, true)?;
                out.write_all(b"\n")?;
            }
        }
    }
    Ok(())
}

fn decode(
    stream: UnixStream,
    reference: &str,
    out: &mut impl Write,
    deadline: Instant,
) -> io::Result<Value> {
    // The transport checks one absolute deadline around every socket read, so
    // a slow trickle within a line cannot extend it.
    let mut reader = BufReader::new(crate::MetadataTransport {
        stream,
        deadline: Some(deadline),
        cancellation: None,
    });
    let mut started = false;
    let mut identity = None;
    loop {
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Session read deadline exceeded",
            ));
        }
        let mut line = String::new();
        reader
            .by_ref()
            .take(FRAME + 1)
            .read_line(&mut line)
            .map_err(|error| {
                // A Unix socket receive timeout shows up as WouldBlock on Linux.
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) {
                    io::Error::new(io::ErrorKind::TimedOut, "Session read deadline exceeded")
                } else {
                    error
                }
            })?;
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Session read deadline exceeded",
            ));
        }
        if line.len() > FRAME as usize || !line.ends_with('\n') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "session stream ended without a completion frame",
            ));
        }
        let mut item: Value = serde_json::from_str(&line)?;
        match item["type"].as_str() {
            Some("session")
                if !started
                    && item["id"] == reference
                    && item["format"] == "polylogue-session"
                    && item["version"] == 1 =>
            {
                started = true;
                identity = Some(item.clone());
            }
            Some("text") if started => out.write_all(
                item["data"]
                    .as_str()
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "text frame has no data")
                    })?
                    .as_bytes(),
            )?,
            Some("end")
                if item["outcome"] == "error"
                    || started
                        && ((item["outcome"] == "degraded"
                            && item["canonical_complete"] == false
                            && item["missing"].as_array().is_some_and(|v| !v.is_empty()))
                            || (item["outcome"] == "ok"
                                && item["canonical_complete"] == true
                                && item["missing"].as_array().is_some_and(|v| v.is_empty()))) =>
            {
                if started && item["outcome"] != "error" {
                    out.write_all(b"\n")?;
                    if let Some(header) = &identity {
                        for key in ["origin", "native_id"] {
                            if let Some(value) = header.get(key) {
                                item[key] = value.clone();
                            }
                        }
                    }
                }
                return Ok(item);
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unexpected frame or wrong session in session stream",
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests;
