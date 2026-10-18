//! Framed JSON Session reads. The normalized document has one owning writer;
//! transport completion always retains the reader's incomplete catalog coverage.
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
fn completion(qualification: &Value, incomplete_reasons: &[String]) -> Value {
    let complete = qualification["outcome"] == "ok";
    // Advisory diagnostics (such as a counted text repair) can accompany a
    // complete read. They are not missing content and must not invalidate its
    // strictly checked completion frame.
    let mut end = json!({"type":"end","outcome":qualification["outcome"],
        "canonical_complete":complete,"missing":incomplete_reasons,"notes":qualification["notes"],
        "unknown":qualification["unknown"]});
    if let Some(diagnostics) = qualification.get("diagnostics") {
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
                "Session read requires an exact reference and json format",
            ));
        }
        let reference = request["reference"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| failed(ErrorKind::InvalidQuery, "missing Session identity"))?;
        let id =
            SessionId::from_stored(reference).map_err(|e| failed(ErrorKind::InvalidQuery, e))?;
        let session = reader
            .lock()
            .map_err(|_| failed(ErrorKind::Io, "reader lock poisoned"))?
            .selected_session(&id, 65536)
            .map_err(|e| failed(ErrorKind::Io, e))?
            .ok_or_else(|| {
                failed(
                    ErrorKind::NotFound,
                    "Session reference is absent from the retained catalog",
                )
            })?;
        let h = session.header().map_err(|e| failed(ErrorKind::Io, e))?;
        let deadline = Instant::now() + TIME;
        // Opaque identities cannot be masked into different references.
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
        let coverage = text_frames(stream, |out| {
            session_document::write_with_cancel(&session, out, deadline, &mut || {
                peer_gone(&cancellation)
            })
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
        let mut end = completion(&coverage.qualification, &coverage.incomplete_reasons);
        end["messages"] = json!(coverage.messages);
        end["view"] = json!(format!(
            "{}:{}",
            session.position().generation,
            session.position().counter
        ));
        end["up_to_date"] = json!(coverage.up_to_date);
        if let Some(recovery) = coverage.recovery {
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
// POLLHUP detects full peer close without treating a request's write-half close
// as cancellation. Safe rustix polling neither changes flags nor spawns helpers.
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
            "Session reference exceeds admission or is empty",
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
/// Terminal rendering consumes the same qualified document as JSON. The
/// bounded client buffer is an explicit surface admission, not another query.
pub fn request_terminal(socket: &Path, reference: &str, out: &mut impl Write) -> io::Result<Value> {
    struct Document(Vec<u8>);
    impl Write for Document {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > (16 * 1024 * 1024usize).saturating_sub(self.0.len()) {
                return Err(io::Error::other(
                    "terminal Session document exceeds 16 MiB admission",
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
        // Typed display fields are scalar strings. Refuse a non-scalar value
        // rather than activating the retained decoder's text-repair projection.
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
    // The shared transport checks the SAME deadline before and after each
    // underlying read, even when read_line needs several partial socket reads.
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
                // Unix socket SO_RCVTIMEO expiry is WouldBlock on Linux.
                // Keep the same absolute deadline and expose its meaning.
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
                "Session stream ended without bounded completion frame",
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
                        io::Error::new(io::ErrorKind::InvalidData, "missing Session document piece")
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
                    "unexpected Session frame or identity mismatch",
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn exchange(frames: Vec<Value>, expected: &str) -> (io::Result<Value>, Vec<u8>) {
        let (reader, mut writer) = UnixStream::pair().unwrap();
        let send = std::thread::spawn(move || {
            for item in frames {
                if frame(&mut writer, item).is_err() {
                    break;
                }
            }
        });
        let mut out = Vec::new();
        let result = decode(
            reader,
            expected,
            &mut out,
            Instant::now() + Duration::from_secs(2),
        );
        send.join().unwrap();
        (result, out)
    }
    fn header() -> Value {
        json!({"type":"session","id":"opaque:session:with:colons","format":"polylogue-session","version":1})
    }
    fn end() -> Value {
        json!({"type":"end","outcome":"degraded","canonical_complete":false,"missing":["source_catalog_coverage"]})
    }
    #[test]
    fn exact_session_identity_precedes_document_and_degraded_completion() {
        let expected = "opaque:session:with:colons";
        let frames = vec![
            header(),
            json!({"type":"text","data":"{\"text\":\"λ🧪\"}"}),
            end(),
        ];
        let (result, out) = exchange(frames.clone(), expected);
        assert_eq!(result.unwrap()["canonical_complete"], false);
        assert_eq!(String::from_utf8(out).unwrap(), "{\"text\":\"λ🧪\"}\n");
        let mut wrong = frames;
        wrong[0]["id"] = json!("other:opaque:session");
        let (result, out) = exchange(wrong, expected);
        assert!(result.is_err());
        assert!(out.is_empty());
    }
    #[test]
    fn malformed_completion_or_frame_order_never_claims_success() {
        let expected = "opaque:session:with:colons";
        let mut complete = end();
        complete["canonical_complete"] = json!(true);
        let mut false_complete = end();
        false_complete["outcome"] = json!("ok");
        false_complete["canonical_complete"] = json!(true);
        let mut missing = end();
        missing["missing"] = json!([]);
        for frames in [
            vec![header()],
            vec![end()],
            vec![header(), header(), end()],
            vec![header(), complete],
            vec![header(), false_complete],
            vec![header(), missing],
            vec![json!({"type":"text","data":"private"}), end()],
        ] {
            let (result, _) = exchange(frames, expected);
            assert!(result.is_err());
        }
        let failure = failed(ErrorKind::NotFound, "Session absent");
        let (result, out) = exchange(vec![failure], expected);
        assert!(out.is_empty());
        assert_eq!(result.unwrap()["outcome"], "error");
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;
    #[test]
    fn only_full_peer_close_cancels_preflight_not_request_half_close() {
        let (server, client) = UnixStream::pair().unwrap();
        assert!(!peer_gone(&server));
        client.shutdown(std::net::Shutdown::Write).unwrap();
        assert!(!peer_gone(&server));
        drop(client);
        assert!(peer_gone(&server));
    }
}

#[cfg(test)]
#[path = "session_read/deadline_tests.rs"]
mod deadline_tests;
#[cfg(test)]
#[path = "session_read/retained_tests.rs"]
mod retained_tests;
