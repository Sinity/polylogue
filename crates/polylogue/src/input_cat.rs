//! Masked UTF-8 presentation of retained bytes. No provider path is opened. A length header
//! prevents a midstream storage failure from becoming successful short output.
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
    let reader = reader
        .lock()
        .map_err(|_| io::Error::other("reader lock"))?
        .fork()
        .map_err(io::Error::other)?;
    let id = request["input"].as_str().filter(|id| !id.is_empty());
    let result = || -> Result<_, String> {
        if request
            .as_object()
            .is_none_or(|o| o.keys().any(|k| !matches!(k.as_str(), "command" | "input")))
        {
            return Err("unknown raw input read field".into());
        }
        let id = id.ok_or("input ID required")?;
        reader
            .source(id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| "unknown retained input".to_string())
    }();
    let source = match result {
        Ok(s) => s,
        Err(e) => {
            serde_json::to_writer(&mut *stream, &json!({"outcome":"error","message":e}))?;
            stream.write_all(b"\n")?;
            return Ok(());
        }
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let preflight = || -> io::Result<u64> {
        reader
            .verify_source(&source, || {
                if std::time::Instant::now() >= deadline {
                    Err(polylogue_store::Error::Cancelled)
                } else {
                    Ok(())
                }
            })
            .map_err(io::Error::other)?;
        // Count a presentation through the same masker before emitting the header.
        // Escaped JSON spellings need decoded-value masking; refuse them until that
        // presentation exists, instead of leaking an escaped credential spelling.
        let mut count = Count(0);
        present(&reader, &source, &mut count, deadline)?;
        Ok(count.0)
    };
    let length = match preflight() {
        Ok(length) => length,
        Err(error) => {
            serde_json::to_writer(
                &mut *stream,
                &json!({"outcome":"error","message":error.to_string()}),
            )?;
            stream.write_all(b"\n")?;
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
    present(&reader, &source, &mut output, deadline)?;
    Ok(())
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
    deadline: std::time::Instant,
) -> io::Result<()> {
    let mut masker = credential_output::Masker::new();
    let mut offset = 0;
    let mut pending = Vec::new();
    while offset < source.input.length {
        if std::time::Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "input presentation deadline",
            ));
        }
        let chunk = reader
            .read_source_range(
                &source.input.id,
                source.input.chain,
                source.binding,
                offset,
                (source.input.length - offset).min(65536),
            )
            .map_err(io::Error::other)?;
        offset += chunk.len() as u64;
        pending.extend_from_slice(&chunk);
        if pending
            .iter()
            .any(|b| *b == b'\\' || (*b < 32 && !matches!(*b, b'\n' | b'\r' | b'\t')))
        {
            return Err(io::Error::other(
                "input cat presentation refuses escaped JSON or binary/control bytes; use Session read",
            ));
        }
        let valid = match std::str::from_utf8(&pending) {
            Ok(_) => pending.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => return Err(io::Error::other("input cat supports UTF-8 only")),
        };
        masker.push(
            std::str::from_utf8(&pending[..valid]).map_err(io::Error::other)?,
            out,
        )?;
        pending.drain(..valid);
    }
    if !pending.is_empty() {
        return Err(io::Error::other("input ends within UTF-8 character"));
    }
    masker.finish(out)
}
pub fn request(socket: &Path, id: &str, out: &mut impl Write) -> io::Result<()> {
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
        return Err(io::Error::other(header.to_string()));
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
    Ok(())
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
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut count = Count(0);
        present(&reader, &source, &mut count, deadline).unwrap();
        let mut result = Vec::new();
        present(&reader, &source, &mut result, deadline).unwrap();
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
            assert!(
                present(
                    &reader,
                    &source,
                    &mut Count(0),
                    std::time::Instant::now() + Duration::from_secs(10)
                )
                .is_err()
            );
        }
    }
}
