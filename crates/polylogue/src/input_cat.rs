//! Masked UTF-8 presentation of retained bytes. No provider path is opened. A length header
//! prevents a midstream storage failure from becoming successful short output.
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
                "input presentation deadline",
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
            .map_err(|_| Failure::new(ErrorKind::Io, "reader lock"))?
            .fork()
            .map_err(store_failure)?;
        if request
            .as_object()
            .is_none_or(|o| o.keys().any(|k| !matches!(k.as_str(), "command" | "input")))
        {
            return Err(Failure::new(
                ErrorKind::InvalidQuery,
                "unknown raw input read field",
            ));
        }
        let id = request["input"]
            .as_str()
            .filter(|id| !id.is_empty())
            .ok_or_else(|| Failure::new(ErrorKind::InvalidQuery, "input ID required"))?;
        let source = reader
            .source(id)
            .map_err(store_failure)?
            .ok_or_else(|| Failure::new(ErrorKind::NotFound, "unknown retained input"))?;
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
            Failure::new(ErrorKind::Timeout, "input verification deadline")
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
    source: &polylogue_store::RetainedSource,
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
    // Count through the same masker before publishing a successful length header.
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
            .ok_or_else(|| io::Error::other("presentation length overflow"))?;
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn present(
    reader: &ReadHandle,
    source: &polylogue_store::RetainedSource,
    out: &mut impl Write,
    check: &mut impl FnMut() -> Result<(), Failure>,
) -> Result<(), Failure> {
    let mut masker = credential_output::Masker::new();
    let mut offset = 0;
    let mut pending = Vec::new();
    while offset < source.input.length {
        check()?;
        let chunk = reader
            .read_source_range(
                &source.input.id,
                source.input.chain,
                source.binding,
                offset,
                (source.input.length - offset).min(65536),
            )
            .map_err(store_failure)?;
        check()?;
        offset += chunk.len() as u64;
        pending.extend_from_slice(&chunk);
        if pending
            .iter()
            .any(|b| *b == b'\\' || (*b < 32 && !matches!(*b, b'\n' | b'\r' | b'\t')))
        {
            return Err(unsupported(
                "input cat presentation refuses escaped JSON or binary/control bytes; use Session read",
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
        return Err(unsupported("input ends within UTF-8 character"));
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
            return Err(io::Error::other("invalid raw input header"));
        }
        let header: Value = serde_json::from_str(&header)?;
        if header["outcome"] != "ok" {
            return Ok(Err(serde_json::from_value(header["error"].clone())?));
        }
        let length = header["length"]
            .as_u64()
            .ok_or_else(|| io::Error::other("missing raw input length"))?;
        let copied = io::copy(&mut (&mut reader).take(length), out)?;
        if copied != length {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "retained input stream truncated",
            ));
        }
        let mut extra = [0];
        if reader.read(&mut extra)? != 0 {
            return Err(io::Error::other("unexpected retained input suffix"));
        }
        Ok(Ok(()))
    };
    exchange().map_err(io_failure)?
}

#[cfg(test)]
mod tests {
    use super::*;
    use polylogue_store::{ImportObservation, InputFormat, Writer};
    fn source(bytes: &[u8]) -> (tempfile::TempDir, Writer, polylogue_store::RetainedSource) {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = Writer::open(&dir.path().join("archive")).unwrap();
        let hash = writer.put_file_blob(bytes).unwrap();
        let retained = writer
            .retain_immutable_input(
                hash,
                InputFormat::ClaudeCodeJsonl,
                &["claude-main", "synthetic"],
                &ImportObservation {
                    id: "input",
                    machine: "m",
                    host: "h",
                    path: "/synthetic.jsonl",
                    seen_at: 1,
                },
                4096,
            )
            .unwrap();
        let source = writer.retained_source(&retained.id).unwrap().unwrap();
        (dir, writer, source)
    }
    #[test]
    fn counted_masked_presentation_matches_stream_across_utf8_boundary() {
        let secret = format!("ghp_{}", "A".repeat(36));
        let text = format!("{}🙂 {secret}\n", "a".repeat(65535));
        let (_dir, writer, source) = source(text.as_bytes());
        let reader = writer.read_handle().unwrap();
        reader.verify_source(&source, || Ok(())).unwrap();
        let mut check = || Ok(());
        let mut count = Count(0);
        present(&reader, &source, &mut count, &mut check).unwrap();
        let mut result = Vec::new();
        present(&reader, &source, &mut result, &mut check).unwrap();
        assert_eq!(count.0, result.len() as u64);
        assert_eq!(
            String::from_utf8(result).unwrap(),
            text.replace(&secret, "[credential: GitHub token, masked]")
        );
        assert_eq!(
            reader
                .read_source_range(&source.input.id, 1, source.binding, 0, source.input.length)
                .unwrap(),
            text.as_bytes()
        );
    }
    #[test]
    fn undecodable_or_escaped_presentations_are_refused_during_preflight() {
        for bytes in [
            b"{\"text\":\"\\u0067hp_secret\"}\n".as_slice(),
            b"binary\0",
            b"bad\xff",
        ] {
            let (_dir, writer, source) = source(bytes);
            let reader = writer.read_handle().unwrap();
            assert!(present(&reader, &source, &mut Count(0), &mut || Ok(())).is_err());
        }
    }
    #[test]
    fn disconnected_preflight_stops_before_full_verification_without_changing_source() {
        let bytes = vec![b'a'; 1024 * 1024];
        let (_dir, writer, source) = source(&bytes);
        let reader = writer.read_handle().unwrap();
        let mut full_checks = 0;
        reader
            .verify_source(&source, || {
                full_checks += 1;
                Ok(())
            })
            .unwrap();
        let (server, client) = UnixStream::pair().unwrap();
        let mut client = Some(client);
        let mut checks = 0;
        let result = preflight(&reader, &source, &mut || {
            checks += 1;
            if checks == 2 {
                drop(client.take());
            }
            if crate::session_read::peer_gone(&server) {
                Err(Failure::new(ErrorKind::Io, "input reader disconnected"))
            } else {
                Ok(())
            }
        });
        assert_eq!(result.unwrap_err().kind, ErrorKind::Io);
        assert!(checks < full_checks);
        let after = reader.source(&source.input.id).unwrap().unwrap();
        assert_eq!(
            (after.input.id, after.input.chain, after.input.length),
            (
                source.input.id.clone(),
                source.input.chain,
                source.input.length
            )
        );
        reader.verify_source(&source, || Ok(())).unwrap();
    }
    #[test]
    fn preflight_preserves_expiry_and_unsupported_presentation_kinds() {
        let (_dir, writer, source) = source(b"escaped\\u0061");
        let reader = writer.read_handle().unwrap();
        assert_eq!(
            preflight(&reader, &source, &mut || Ok(()))
                .unwrap_err()
                .kind,
            ErrorKind::NotYetAvailable
        );
        assert_eq!(
            preflight(&reader, &source, &mut || Err(Failure::new(
                ErrorKind::Timeout,
                "expired"
            )))
            .unwrap_err()
            .kind,
            ErrorKind::Timeout
        );
    }
    #[test]
    fn daemon_and_client_preserve_typed_unknown_input_failure() {
        let (dir, writer, _) = source(b"neutral");
        let socket = dir.path().join("read.sock");
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        let reader = Mutex::new(writer.read_handle().unwrap());
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut line = String::new();
            BufReader::new(stream.try_clone().unwrap())
                .read_line(&mut line)
                .unwrap();
            serve(&mut stream, &serde_json::from_str(&line).unwrap(), &reader).unwrap();
        });
        let mut output = Vec::new();
        let failure = request(&socket, "unknown", &mut output).unwrap_err();
        assert_eq!(failure.kind, ErrorKind::NotFound);
        assert!(output.is_empty());
        server.join().unwrap();
    }
}
