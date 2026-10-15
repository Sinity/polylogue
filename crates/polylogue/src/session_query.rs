//! Ordinary query transport. Binding precedes reader acquisition, and both CLI
//! renderings consume this one daemon result without a second filter engine.
use crate::reference_read::{FRAME, frame, text_frames};
use polylogue_model::types::{ErrorKind, Failure};
use polylogue_store::ReadHandle;
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, BufReader, Write},
    os::unix::net::UnixStream,
    path::Path,
    sync::Mutex,
    time::{Duration, Instant},
};
const TIME: Duration = Duration::from_secs(30);
fn failed(kind: ErrorKind, message: impl ToString) -> Value {
    json!({"type":"end","outcome":"error","error":Failure::new(kind,message.to_string())})
}
fn store_failure(error: polylogue_store::Error) -> Value {
    let kind = match &error {
        polylogue_store::Error::Cancelled => ErrorKind::Timeout,
        polylogue_store::Error::ParserAdmission { .. } => ErrorKind::NotYetAvailable,
        polylogue_store::Error::ExcisionCheckRequired => ErrorKind::Refused {
            reason: "excision-aware query reader is not registered".into(),
        },
        polylogue_store::Error::MessagePredecessorChanged => ErrorKind::Conflict,
        _ => ErrorKind::Io,
    };
    failed(kind, error)
}
pub(crate) fn serve(
    stream: &mut UnixStream,
    request: &Value,
    reader: &Mutex<ReadHandle>,
) -> io::Result<()> {
    stream.set_write_timeout(Some(TIME))?;
    let result = (|| -> Result<Value, Value> {
        if request.as_object().is_none_or(|o| {
            o.len() != 2 || o.keys().any(|k| !matches!(k.as_str(), "command" | "query"))
        }) || !request["query"].is_string()
        {
            return Err(failed(
                ErrorKind::InvalidQuery,
                "query request requires only command and query text",
            ));
        }
        let plan = polylogue_query::bind(request["query"].as_str().unwrap())
            .map_err(|e| json!({"type":"end","outcome":"error","error":e}))?;
        let deadline = Instant::now() + TIME;
        let catalog = reader
            .lock()
            .map_err(|_| failed(ErrorKind::Io, "reader lock poisoned"))?
            .session_catalog(65536)
            .map_err(store_failure)?;
        let cancellation = stream.try_clone().map_err(|e| failed(ErrorKind::Io, e))?;
        let result =
            polylogue_engine::session_query::execute(&catalog, &plan, deadline, &mut || {
                crate::session_read::peer_gone(&cancellation)
            })
            .map_err(store_failure)?;
        for row in result["rows"].as_array().unwrap() {
            for key in ["id", "native_id"] {
                crate::output_json::check_identity(row[key].as_str().unwrap(), deadline)
                    .map_err(|e| failed(ErrorKind::NotYetAvailable, e))?;
            }
            if let Some(hits) = row["hits"].as_array() {
                for hit in hits {
                    for key in ["message", "session"] {
                        crate::output_json::check_identity(hit[key].as_str().unwrap(), deadline)
                            .map_err(|e| failed(ErrorKind::NotYetAvailable, e))?;
                    }
                }
            }
        }
        // Encode before publishing a start frame. Credential refusal must not
        // leak a partial result, and output admission fails the whole answer.
        let mut bytes = Vec::new();
        let mut encoder = crate::output_json::Encoder::new(&mut bytes, deadline);
        encoder
            .value(&result)
            .and_then(|_| encoder.finish())
            .map_err(|e| failed(ErrorKind::NotYetAvailable, e))?;
        if bytes.len() > 4 * 1024 * 1024 {
            return Err(failed(
                ErrorKind::NotYetAvailable,
                "query response exceeds 4 MiB admission; narrow the query or use | limit",
            ));
        }
        frame(stream, json!({"type":"query","unit":"sessions"}))
            .map_err(|e| failed(ErrorKind::Io, e))?;
        text_frames(stream, |out| out.write_all(&bytes)).map_err(|e| failed(ErrorKind::Io, e))?;
        Ok(
            json!({"type":"end","outcome":result["outcome"],"count":result["count"],"notes":result["notes"]}),
        )
    })();
    frame(
        stream,
        match result {
            Ok(value) | Err(value) => value,
        },
    )
}
pub fn request(socket: &Path, query: &str) -> io::Result<Value> {
    let deadline = Instant::now() + TIME;
    let mut stream = crate::metadata_transport(socket, Some(deadline))?;
    frame(&mut stream, json!({"command":"query","query":query}))?;
    let mut reader = BufReader::new(stream);
    let mut started = false;
    let mut bytes = Vec::new();
    loop {
        let mut line = Vec::new();
        use std::io::Read;
        (&mut reader)
            .take(FRAME as u64 + 1)
            .read_until(b'\n', &mut line)?;
        if line.is_empty() || line.len() > FRAME as usize || !line.ends_with(b"\n") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid query response frame",
            ));
        }
        let frame: Value = serde_json::from_slice(&line)?;
        match frame["type"].as_str() {
            Some("query") if !started && bytes.is_empty() => started = true,
            Some("text") if started => {
                let text = frame["data"].as_str().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid query text")
                })?;
                bytes.extend_from_slice(text.as_bytes());
                if bytes.len() > 4 * 1024 * 1024 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "query response exceeds admission",
                    ));
                }
            }
            Some("end") if frame["outcome"] == "error" => return Ok(frame),
            Some("end") if started => {
                let result: Value = serde_json::from_slice(&bytes)?;
                if result["outcome"] != frame["outcome"] || result["count"] != frame["count"] {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "query completion mismatch",
                    ));
                }
                return Ok(result);
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unexpected query frame",
                ));
            }
        }
    }
}
