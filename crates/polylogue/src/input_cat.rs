//! `input cat`: prints a stored input's bytes as UTF-8 with credentials
//! masked, reading the archive copy, never the provider file. The masked
//! length is computed first and sent in a header, so a storage failure
//! midway cannot pass for a shorter, successful output.
use polylogue_model::types::{ErrorKind, Failure};
use polylogue_store::ReadHandle;
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    os::unix::net::UnixStream,
    path::Path,
    sync::Mutex,
    time::Duration,
};
pub(crate) fn serve(
    stream: &mut UnixStream,
    request: &Value,
    reader: &Mutex<ReadHandle>,
) -> io::Result<()> {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let peer = stream.try_clone()?;
    let mut check = || {
        if std::time::Instant::now() >= deadline {
            Err(Failure::new(
                ErrorKind::Timeout,
                "input read deadline exceeded",
            ))
        } else if crate::session_read::peer_gone(&peer) {
            Err(Failure::new(ErrorKind::Io, "input reader disconnected"))
        } else {
            Ok(())
        }
    };
    let result = || -> Result<_, Failure> {
        let reader = reader
            .lock()
            .map_err(|_| Failure::new(ErrorKind::Io, "reader lock poisoned"))?
            .fork()
            .map_err(store_failure)?;
        if request
            .as_object()
            .is_none_or(|o| o.keys().any(|k| !matches!(k.as_str(), "command" | "input")))
        {
            return Err(Failure::new(
                ErrorKind::InvalidQuery,
                "unknown field in input-cat request",
            ));
        }
        let id = request["input"]
            .as_str()
            .filter(|id| !id.is_empty())
            .ok_or_else(|| Failure::new(ErrorKind::InvalidQuery, "input ID required"))?;
        let source = reader
            .source(id)
            .map_err(store_failure)?
            .ok_or_else(|| Failure::new(ErrorKind::NotFound, "no such input"))?;
        Ok((reader, source))
    };
    let prepared = result().and_then(|(reader, source)| {
        let length = preflight(&reader, &source, &mut check)?;
        Ok((reader, source, length))
    });
    let (reader, source, length) = match prepared {
        Ok(value) => value,
        Err(failure) => {
            let mut encoder = crate::output_json::Encoder::new(&mut *stream, deadline);
            encoder.value(&json!({"outcome":"error","error":failure}))?;
            encoder.syntax(b"\n")?;
            return Ok(());
        }
    };
    let mut output = crate::MetadataTransport {
        stream: stream.try_clone()?,
        deadline: Some(deadline),
        cancellation: None,
    };
    serde_json::to_writer(
        &mut output,
        &json!({"outcome":"ok","length":length,"source_length":source.input.length,"presentation":"credential-output-v1"}),
    )?;
    output.write_all(b"\n")?;
    present(&reader, &source, &mut output, &mut check).map_err(|f| io::Error::other(f.message))?;
    Ok(())
}
fn io_failure(error: io::Error) -> Failure {
    let kind = if matches!(
        error.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    ) {
        ErrorKind::Timeout
    } else {
        ErrorKind::Io
    };
    Failure::new(kind, error.to_string())
}
fn store_failure(error: polylogue_store::Error) -> Failure {
    match error {
        polylogue_store::Error::Io(error) => io_failure(error),
        polylogue_store::Error::Cancelled => {
            Failure::new(ErrorKind::Timeout, "input check deadline exceeded")
        }
        polylogue_store::Error::ExcisionCheckRequired => Failure::new(
            ErrorKind::Refused {
                reason: "excision-aware read required".into(),
            },
            "excision-aware read required",
        ),
        other => Failure::new(ErrorKind::Io, other.to_string()),
    }
}
fn unsupported(message: &str) -> Failure {
    Failure::new(ErrorKind::NotYetAvailable, message)
}
fn preflight(
    reader: &ReadHandle,
    source: &polylogue_store::StoredSource,
    check: &mut impl FnMut() -> Result<(), Failure>,
) -> Result<u64, Failure> {
    let mut stopped = None;
    reader
        .verify_source(source, || {
            check().map_err(|failure| {
                stopped = Some(failure);
                polylogue_store::Error::Cancelled
            })
        })
        .map_err(|error| stopped.unwrap_or_else(|| store_failure(error)))?;
    // Mask once to count the bytes, so the header carries the exact length.
    let mut count = Count(0);
    present(reader, source, &mut count, check)?;
    Ok(count.0)
}
struct Count(u64);
impl Write for Count {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(b.len() as u64)
            .ok_or_else(|| io::Error::other("masked length overflow"))?;
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn present(
    reader: &ReadHandle,
    source: &polylogue_store::StoredSource,
    out: &mut impl Write,
    check: &mut impl FnMut() -> Result<(), Failure>,
) -> Result<(), Failure> {
    let mut masker = credential_masking::Masker::new();
    let mut offset = 0;
    let mut pending = Vec::new();
    while offset < source.input.length {
        check()?;
        let mut stopped = None;
        let chunk = reader
            .read_source_range_checked(
                &source.input.id,
                source.input.chain,
                source.binding,
                offset,
                (source.input.length - offset).min(65536),
                || {
                    check().map_err(|failure| {
                        stopped = Some(failure);
                        polylogue_store::Error::Cancelled
                    })
                },
            )
            .map_err(|error| stopped.unwrap_or_else(|| store_failure(error)))?;
        check()?;
        offset += chunk.len() as u64;
        pending.extend_from_slice(&chunk);
        if pending
            .iter()
            .any(|b| *b == b'\\' || (*b < 32 && !matches!(*b, b'\n' | b'\r' | b'\t')))
        {
            return Err(unsupported(
                "input has JSON escapes or control bytes, which input cat cannot mask; read the session instead",
            ));
        }
        let valid = match std::str::from_utf8(&pending) {
            Ok(_) => pending.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => return Err(unsupported("input cat supports UTF-8 only")),
        };
        masker
            .push(
                std::str::from_utf8(&pending[..valid])
                    .map_err(|_| unsupported("input cat supports UTF-8 only"))?,
                out,
            )
            .map_err(io_failure)?;
        pending.drain(..valid);
    }
    if !pending.is_empty() {
        return Err(unsupported("input ends inside a UTF-8 character"));
    }
    check()?;
    masker.finish(out).map_err(io_failure)
}
pub fn request(socket: &Path, id: &str, out: &mut impl Write) -> Result<(), Failure> {
    let mut exchange = || -> io::Result<Result<(), Failure>> {
        let mut stream = crate::metadata_transport_cancellable(
            socket,
            Some(std::time::Instant::now() + Duration::from_secs(30)),
            None,
        )?;
        serde_json::to_writer(&mut stream, &json!({"command":"input-cat","input":id}))?;
        stream.write_all(b"\n")?;
        let mut reader = BufReader::new(stream);
        let mut header = String::new();
        (&mut reader).take(65537).read_line(&mut header)?;
        if header.len() > 65536 || !header.ends_with('\n') {
            return Err(io::Error::other("invalid input-cat header"));
        }
        let header: Value = serde_json::from_str(&header)?;
        if header["outcome"] != "ok" {
            return Ok(Err(serde_json::from_value(header["error"].clone())?));
        }
        let length = header["length"]
            .as_u64()
            .ok_or_else(|| io::Error::other("input-cat header has no length"))?;
        let copied = io::copy(&mut (&mut reader).take(length), out)?;
        if copied != length {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "input stream ended early",
            ));
        }
        let mut extra = [0];
        if reader.read(&mut extra)? != 0 {
            return Err(io::Error::other(
                "input stream is longer than its header says",
            ));
        }
        Ok(Ok(()))
    };
    exchange().map_err(io_failure)?
}

#[cfg(test)]
mod tests;
