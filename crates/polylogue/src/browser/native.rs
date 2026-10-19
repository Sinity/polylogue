//! `polylogue browser-host`: the extension's native messaging host. Chrome
//! starts one process per request and talks to it in native messaging
//! frames (a 32-bit native-endian length, then JSON) on stdin and stdout.
//!
//! One exchange: the extension sends `request` (method, path, headers),
//! then the body as `body` frames of at most 64 KiB, each answered with
//! `body_ack`, then `body_end`. The host serves it with the browser
//! router in process and answers `response` (status, headers, its receiver
//! ID and API schema), then one `response_body` frame per `response_next`
//! until `response_end`. `cancel` or end of input stops it; a failure is
//! an `error` frame.

use super::{API_SCHEMA, Ctx};
use axum::body::{Body, to_bytes};
use axum::http::{Method, Request};
use base64::Engine as _;
use serde_json::{Value, json};
use std::io::{self, Read, Write};
use std::process::ExitCode;
use std::sync::Arc;
use tower::ServiceExt;

/// The extension sends bodies in pieces of this size; responses go back the
/// same way, well under Chrome's 1 MB limit per message to the extension.
const CHUNK: usize = 64 * 1024;

pub fn read_frame(input: &mut impl Read) -> io::Result<Option<Value>> {
    let mut len = [0u8; 4];
    match input.read_exact(&mut len) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let mut buf = vec![0u8; u32::from_ne_bytes(len) as usize];
    input.read_exact(&mut buf)?;
    serde_json::from_slice(&buf)
        .map(Some)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn write_frame(out: &mut impl Write, v: &Value) -> io::Result<()> {
    let bytes = serde_json::to_vec(v).map_err(io::Error::other)?;
    let len = u32::try_from(bytes.len()).map_err(io::Error::other)?;
    out.write_all(&len.to_ne_bytes())?;
    out.write_all(&bytes)?;
    out.flush()
}

fn fail(out: &mut impl Write, code: &str) -> io::Result<()> {
    write_frame(out, &json!({ "type": "error", "error": code }))
}

/// Serves one exchange from `input` to `out`.
pub fn serve(ctx: Arc<Ctx>, input: &mut impl Read, out: &mut impl Write) -> io::Result<()> {
    let Some(request) = read_frame(input)? else {
        return Ok(());
    };
    if request["type"] != "request" || request["version"] != 1 {
        return fail(out, "native_request_invalid");
    }
    let (Some(method), Some(path)) = (request["method"].as_str(), request["path"].as_str()) else {
        return fail(out, "native_request_invalid");
    };
    let Ok(method) = Method::from_bytes(method.as_bytes()) else {
        return fail(out, "native_request_invalid");
    };
    if !path.starts_with("/v1/") {
        return fail(out, "native_route_forbidden");
    }
    let mut body = Vec::new();
    let mut sequence = 0u64;
    loop {
        let Some(frame) = read_frame(input)? else {
            return Ok(());
        };
        match frame["type"].as_str() {
            Some("body") if frame["sequence"].as_u64() == Some(sequence) => {
                let Some(bytes) = frame["data"]
                    .as_str()
                    .and_then(|d| base64::engine::general_purpose::STANDARD.decode(d).ok())
                else {
                    return fail(out, "native_request_body_invalid");
                };
                body.extend_from_slice(&bytes);
                write_frame(out, &json!({ "type": "body_ack", "sequence": sequence }))?;
                sequence += 1;
            }
            Some("body_end") => break,
            Some("cancel") => return Ok(()),
            _ => return fail(out, "native_request_frame_invalid"),
        }
    }
    let mut http = Request::builder().method(method).uri(path);
    if let Some(headers) = request["headers"].as_object() {
        for (k, v) in headers {
            if let Some(v) = v.as_str() {
                http = http.header(k, v);
            }
        }
    }
    let Ok(http) = http.body(Body::from(body)) else {
        return fail(out, "native_request_invalid");
    };
    let receiver_id = ctx.receiver_id.clone();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let (status, headers, bytes) = rt.block_on(async move {
        let r = super::routes::router(ctx)
            .oneshot(http)
            .await
            .unwrap_or_else(|e| match e {});
        let status = r.status().as_u16();
        let headers: serde_json::Map<String, Value> = r
            .headers()
            .iter()
            .filter_map(|(k, v)| Some((k.as_str().to_owned(), json!(v.to_str().ok()?))))
            .collect();
        let bytes = to_bytes(r.into_body(), usize::MAX)
            .await
            .unwrap_or_default();
        (status, headers, bytes)
    });
    write_frame(
        out,
        &json!({
            "type": "response",
            "status": status,
            "receiver_id": receiver_id,
            "api_schema": API_SCHEMA,
            "headers": headers,
        }),
    )?;
    let mut chunks = bytes.chunks(CHUNK).enumerate();
    loop {
        let Some(frame) = read_frame(input)? else {
            return Ok(());
        };
        match frame["type"].as_str() {
            Some("response_next") => match chunks.next() {
                Some((i, chunk)) => write_frame(
                    out,
                    &json!({
                        "type": "response_body",
                        "sequence": i,
                        "data": base64::engine::general_purpose::STANDARD.encode(chunk),
                    }),
                )?,
                None => return write_frame(out, &json!({ "type": "response_end" })),
            },
            Some("cancel") => return Ok(()),
            _ => return fail(out, "native_response_frame_invalid"),
        }
    }
}

/// The command Chrome runs. It passes the calling extension's origin as the
/// first argument; Chrome itself admits only the origins the host's
/// manifest lists.
pub fn run() -> ExitCode {
    let ctx = match Ctx::for_user() {
        Ok(c) => Arc::new(c),
        Err(e) => {
            let _ = fail(&mut io::stdout().lock(), "host_storage_unavailable");
            eprintln!("polylogue browser-host: {e}");
            return ExitCode::FAILURE;
        }
    };
    match serve(ctx, &mut io::stdin().lock(), &mut io::stdout().lock()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("polylogue browser-host: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The host manifest Chrome reads from its `NativeMessagingHosts`
/// directory: this binary, admitting only `extension_id`.
pub fn manifest(binary: &std::path::Path, extension_id: &str) -> Value {
    json!({
        "name": super::HOST_NAME,
        "description": "Polylogue browser capture host",
        "path": binary,
        "type": "stdio",
        "allowed_origins": [format!("chrome-extension://{extension_id}/")],
    })
}
