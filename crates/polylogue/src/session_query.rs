//! Session queries over the daemon socket. The query is parsed before the
//! reader is locked, and the CLI's JSON and terminal output render this result.
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
const MAX_RESPONSE: usize = 4 * 1024 * 1024;
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
        polylogue_store::Error::StaleMessage => ErrorKind::Conflict,
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
        let tool_plan = polylogue_query::bind_tools(request["query"].as_str().unwrap())
            .map_err(|e| json!({"type":"end","outcome":"error","error":e}))?;
        let deadline = Instant::now() + TIME;
        let mut result = if let Some(plan) = tool_plan {
            let guard = reader
                .lock()
                .map_err(|_| failed(ErrorKind::Io, "reader lock poisoned"))?;
            crate::native_tools::query(&guard, &plan, deadline).map_err(store_failure)?
        } else {
            let plan = polylogue_query::bind(request["query"].as_str().unwrap())
                .map_err(|e| json!({"type":"end","outcome":"error","error":e}))?;
            let catalog = reader
                .lock()
                .map_err(|_| failed(ErrorKind::Io, "reader lock poisoned"))?
                .session_catalog(65536)
                .map_err(store_failure)?;
            let cancellation = stream.try_clone().map_err(|e| failed(ErrorKind::Io, e))?;
            polylogue_engine::session_query::execute(&catalog, &plan, deadline, &mut || {
                crate::session_read::peer_gone(&cancellation)
            })
            .map_err(store_failure)?
        };
        let guard = reader
            .lock()
            .map_err(|_| failed(ErrorKind::Io, "reader lock poisoned"))?;
        let pending =
            crate::pending_codex_references(&guard, deadline, None).map_err(store_failure)?;
        drop(guard);
        if pending["examined_snapshots"]
            .as_u64()
            .is_some_and(|n| n > 0)
            || pending["scan_complete"] == false
        {
            result["pending_references"] = pending;
            result["outcome"] = json!("degraded");
            result["up_to_date"] = json!(false);
        }
        for row in result["rows"].as_array().unwrap() {
            if let Some(id) = row["id"].as_str() {
                crate::output_json::check_reference(id, deadline)
                    .map_err(|e| failed(ErrorKind::NotYetAvailable, e))?;
            }
            if let Some(value) = row["native_id"].as_str() {
                let native =
                    polylogue_model::NativeId::from_wire(value, row["native_id_encoding"].as_str())
                        .map_err(|_| {
                            failed(
                                ErrorKind::NotYetAvailable,
                                "invalid native Session identity",
                            )
                        })?;
                crate::output_json::check_native_identity(&native, deadline)
                    .map_err(|e| failed(ErrorKind::NotYetAvailable, e))?;
            }
            if let Some(hits) = row["hits"].as_array() {
                for hit in hits {
                    for key in ["message", "session"] {
                        crate::output_json::check_reference(hit[key].as_str().unwrap(), deadline)
                            .map_err(|e| failed(ErrorKind::NotYetAvailable, e))?;
                    }
                }
            }
        }
        // Encode the whole result before the first frame, so a masking failure
        // or the size limit fails the answer without sending part of it.
        let mut bytes = Vec::new();
        let mut encoder = crate::output_json::Encoder::new(&mut bytes, deadline);
        encoder
            .value(&result)
            .and_then(|_| encoder.finish())
            .map_err(|e| failed(ErrorKind::NotYetAvailable, e))?;
        if bytes.len() > MAX_RESPONSE && result.get("pending_references").is_some() {
            crate::summarize_pending_references(&mut result);
            bytes.clear();
            let mut encoder = crate::output_json::Encoder::new(&mut bytes, deadline);
            encoder
                .value(&result)
                .and_then(|_| encoder.finish())
                .map_err(|e| failed(ErrorKind::NotYetAvailable, e))?;
        }
        if bytes.len() > MAX_RESPONSE {
            return Err(failed(
                ErrorKind::NotYetAvailable,
                "query response exceeds 4 MiB; narrow the query or use | limit N",
            ));
        }
        frame(
            stream,
            json!({"type":"query","unit":result["unit"].as_str().unwrap_or("sessions")}),
        )
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
    request_inner(socket, query, None)
}
pub fn request_cancellable(
    socket: &Path,
    query: &str,
    cancellation: crate::read_cancellation::ReadCancellation,
) -> io::Result<Value> {
    request_inner(socket, query, Some(cancellation))
}
fn request_inner(
    socket: &Path,
    query: &str,
    cancellation: Option<crate::read_cancellation::ReadCancellation>,
) -> io::Result<Value> {
    let deadline = Instant::now() + TIME;
    let mut stream = crate::metadata_transport_cancellable(socket, Some(deadline), cancellation)?;
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
                if bytes.len() > MAX_RESPONSE {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "query response exceeds 4 MiB",
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
