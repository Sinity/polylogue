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
        crate::output_json::check_identity(h.id.as_str(), deadline)
            .map_err(|e| failed(ErrorKind::NotYetAvailable, e))?;
        crate::output_json::check_identity(&h.native_id, deadline)
            .map_err(|e| failed(ErrorKind::NotYetAvailable, e))?;
        frame(stream,json!({"type":"session","id":h.id.as_str(),"origin":h.origin.as_str(),"native_id":h.native_id,"format":"polylogue-session","version":1})).map_err(|e|failed(ErrorKind::Io,e))?;
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
        match coverage.catalog {
            polylogue_store::contributions::SessionCoverage::KnownCatalogOnly => Ok(
                json!({"type":"end","outcome":"degraded","canonical_complete":false,"messages":coverage.messages,"missing":coverage.qualification["notes"],"notes":coverage.qualification["notes"],"unknown":coverage.qualification["unknown"]}),
            ),
        }
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
fn peer_gone(stream: &UnixStream) -> bool {
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
    decode(
        BufReader::new(stream),
        reference,
        out,
        Instant::now() + TIME,
    )
}
fn decode(
    mut reader: BufReader<UnixStream>,
    reference: &str,
    out: &mut impl Write,
    deadline: Instant,
) -> io::Result<Value> {
    let mut started = false;
    loop {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::TimedOut, "Session read deadline exceeded")
            })?;
        reader.get_ref().set_read_timeout(Some(remaining))?;
        let mut line = String::new();
        reader.by_ref().take(FRAME + 1).read_line(&mut line)?;
        if line.len() > FRAME as usize || !line.ends_with('\n') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Session stream ended without bounded completion frame",
            ));
        }
        let item: Value = serde_json::from_str(&line)?;
        match item["type"].as_str() {
            Some("session")
                if !started
                    && item["id"] == reference
                    && item["format"] == "polylogue-session"
                    && item["version"] == 1 =>
            {
                started = true
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
                        && item["outcome"] == "degraded"
                        && item["canonical_complete"] == false
                        && item["missing"].as_array().is_some_and(|v| !v.is_empty()) =>
            {
                if started && item["outcome"] != "error" {
                    out.write_all(b"\n")?;
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
            BufReader::new(reader),
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
        let mut missing = end();
        missing["missing"] = json!([]);
        for frames in [
            vec![header()],
            vec![end()],
            vec![header(), header(), end()],
            vec![header(), complete],
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
