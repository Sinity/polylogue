//! Streaming terminal reads of one whole selected Message snapshot. This is not
//! the session JSON export surface; history/provenance coverage stays degraded.
use polylogue_model::types::{ErrorKind, Failure};
use polylogue_store::{ReadHandle, contributions::MessageLookup};
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, BufReader, Read, Write},
    os::unix::net::UnixStream,
    path::Path,
    time::Duration,
};
pub(crate) const FRAME: u64 = 65536;
const PIECE: usize = 4096;
fn failed(kind: ErrorKind, error: impl ToString) -> Value {
    json!({"type":"end","outcome":"error","error":Failure::new(kind,error.to_string())})
}
pub(crate) fn frame(out: &mut impl Write, value: Value) -> io::Result<()> {
    struct Encoded(Vec<u8>);
    impl Write for Encoded {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > (FRAME as usize - 1).saturating_sub(self.0.len()) {
                return Err(io::Error::other("read frame exceeds encoded admission"));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut encoded = Encoded(Vec::new());
    serde_json::to_writer(&mut encoded, &value)?;
    encoded.0.push(b'\n');
    out.write_all(&encoded.0)
}

// Masking can expand dense short credentials far beyond the input piece.
// Encode settled output into bounded frames; the masker still owns undecided
// bytes. UTF-8 fragments across individual Write calls retain at most3 bytes.
pub(crate) fn text_frames<W: Write, T>(
    stream: &mut W,
    run: impl FnOnce(&mut TextFrames<'_, W>) -> io::Result<T>,
) -> io::Result<T> {
    let mut output = TextFrames {
        stream,
        bytes: Vec::new(),
    };
    let result = run(&mut output)?;
    output.emit(true)?;
    Ok(result)
}
pub(crate) struct TextFrames<'a, W: Write> {
    stream: &'a mut W,
    bytes: Vec<u8>,
}
impl<W: Write> TextFrames<'_, W> {
    fn emit(&mut self, complete: bool) -> io::Result<()> {
        let valid = match std::str::from_utf8(&self.bytes) {
            Ok(_) => self.bytes.len(),
            Err(error) if error.error_len().is_none() && !complete => error.valid_up_to(),
            Err(error) => return Err(io::Error::new(io::ErrorKind::InvalidData, error)),
        };
        if valid != 0 {
            let text = std::str::from_utf8(&self.bytes[..valid])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            // PIECE source bytes need at most6*PIECE JSON escaping bytes, below
            // FRAME. Only this one bounded encoding is retained while writing.
            frame(self.stream, json!({"type":"text","data":text}))?;
            self.bytes.drain(..valid);
        }
        Ok(())
    }
}
impl<W: Write> Write for TextFrames<'_, W> {
    fn write(&mut self, mut bytes: &[u8]) -> io::Result<usize> {
        let length = bytes.len();
        while !bytes.is_empty() {
            let take = bytes.len().min(PIECE - self.bytes.len());
            self.bytes
                .try_reserve_exact(take)
                .map_err(io::Error::other)?;
            self.bytes.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.bytes.len() == PIECE {
                self.emit(false)?;
            }
        }
        Ok(length)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.emit(false)
    }
}

pub(crate) fn serve(
    stream: &mut UnixStream,
    request: &Value,
    reader: &std::sync::Mutex<ReadHandle>,
) -> io::Result<()> {
    let mut recovery = None;
    let mut run = || -> Result<(), Value> {
        if request.as_object().is_none_or(|o| {
            o.keys()
                .any(|k| !matches!(k.as_str(), "command" | "reference" | "format"))
        }) || request["format"] != "terminal"
        {
            return Err(failed(
                ErrorKind::InvalidQuery,
                "read requires an exact reference and terminal format",
            ));
        }
        let id = request["reference"]
            .as_str()
            .and_then(|s| s.strip_prefix("message:"))
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                failed(
                    ErrorKind::NotYetAvailable,
                    "only explicit message: references are registered by this read path",
                )
            })?;
        let selected = reader
            .lock()
            .map_err(|_| failed(ErrorKind::Io, "reader lock poisoned"))?
            .message_reference(id, 16384)
            .map_err(|e| failed(ErrorKind::Io, e))?;
        let message = match selected {
            MessageLookup::Absent => {
                return Err(failed(ErrorKind::NotFound, "Message reference is absent"));
            }
            MessageLookup::Pending => {
                return Err(failed(
                    ErrorKind::NotYetAvailable,
                    "Message has no selected whole canonical body; finalization remains pending",
                ));
            }
            MessageLookup::Ready(v) => v,
        };
        let header = message.header().map_err(|e| failed(ErrorKind::Io, e))?;
        recovery = message.recovery().map_err(|e| failed(ErrorKind::Io, e))?;
        frame(stream,json!({"type":"message","id":message.id().as_str(),"session":message.id().session().as_str(),"role":header.role,"written_by":header.written_by})).map_err(|e|failed(ErrorKind::Io,e))?;
        let mut masker = credential_output::Masker::new();
        let mut index = 0u64;
        loop {
            let Some(block) = message
                .block_at(index)
                .map_err(|e| failed(ErrorKind::Io, e))?
            else {
                break;
            };
            if !matches!(block.kind.as_str(), "text" | "thinking") {
                return Err(failed(
                    ErrorKind::NotYetAvailable,
                    "selected block kind has no terminal renderer",
                ));
            }
            frame(stream,json!({"type":"block","kind":block.kind,"slot":block.slot,"encrypted":block.encrypted})).map_err(|e|failed(ErrorKind::Io,e))?;
            let mut offset = 0;
            let mut pending = Vec::with_capacity(4);
            loop {
                let Some(range) = block
                    .text_range(offset, PIECE)
                    .map_err(|e| failed(ErrorKind::Io, e))?
                else {
                    break;
                };
                let mut bytes = pending;
                bytes.extend_from_slice(&range.bytes);
                let valid = match std::str::from_utf8(&bytes) {
                    Ok(_) => bytes.len(),
                    Err(e) if e.error_len().is_none() => e.valid_up_to(),
                    Err(_) => return Err(failed(ErrorKind::Io, "selected UTF-8 value is invalid")),
                };
                let text =
                    std::str::from_utf8(&bytes[..valid]).map_err(|e| failed(ErrorKind::Io, e))?;
                if !text.is_empty() {
                    text_frames(stream, |output| masker.push(text, output))
                        .map_err(|e| failed(ErrorKind::Io, e))?;
                }
                pending = bytes[valid..].to_vec();
                if range.eof {
                    if !pending.is_empty() {
                        return Err(failed(
                            ErrorKind::Io,
                            "selected UTF-8 value ends within a character",
                        ));
                    }
                    break;
                }
                if range.next_offset <= offset {
                    return Err(failed(ErrorKind::Io, "selected range made no progress"));
                }
                offset = range.next_offset;
            }
            text_frames(stream, |output| masker.block_boundary(output))
                .map_err(|e| failed(ErrorKind::Io, e))?;
            index = index
                .checked_add(1)
                .ok_or_else(|| failed(ErrorKind::Io, "block position overflow"))?;
        }
        text_frames(stream, |output| masker.finish(output))
            .map_err(|e| failed(ErrorKind::Io, e))?;
        Ok(())
    };
    match run() {
        Ok(()) => frame(
            stream,
            json!({"type":"end","outcome":"degraded","missing":["canonical_history_coverage","source_provenance"],"notes":if recovery.is_some(){vec![polylogue_model::RecoveryQualification::NOTE]}else{vec![]},"recovery":recovery,"canonical_complete":false}),
        ),
        Err(error) => frame(stream, error),
    }
}
/// Frames are transport-only. The terminal renderer writes one bounded text
/// piece at a time; it never collects a Message or mistakes EOF for success.
pub(crate) fn terminal_text(out: &mut impl Write, text: &str, layout: bool) -> io::Result<()> {
    let mut start = 0;
    for (index, ch) in text.char_indices() {
        if ch.is_control() && !(layout && matches!(ch, '\n' | '\t')) {
            out.write_all(&text.as_bytes()[start..index])?;
            write!(out, "\\x{:02x}", ch as u32)?;
            start = index + ch.len_utf8();
        }
    }
    out.write_all(&text.as_bytes()[start..])
}
pub fn request(socket: &Path, reference: &str, out: &mut impl Write) -> io::Result<Value> {
    let expected = reference
        .strip_prefix("message:")
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "expected a message reference")
        })?;
    let mut stream = UnixStream::connect(socket)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    frame(
        &mut stream,
        json!({"command":"read","reference":reference,"format":"terminal"}),
    )?;
    let mut reader = BufReader::new(stream);
    let mut started = false;
    loop {
        let mut line = String::new();
        reader.by_ref().take(FRAME + 1).read_line(&mut line)?;
        if line.len() > FRAME as usize || !line.ends_with('\n') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "read stream ended without a bounded completion frame",
            ));
        }
        let item: Value = serde_json::from_str(&line)?;
        match item["type"].as_str() {
            Some("message") if !started => {
                let id = item["id"]
                    .as_str()
                    .ok_or_else(|| io::Error::other("missing Message identity"))?;
                if id != expected {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Message identity does not match requested reference",
                    ));
                }
                let role = item["role"]
                    .as_str()
                    .ok_or_else(|| io::Error::other("missing role"))?;
                let by = item["written_by"]
                    .as_str()
                    .ok_or_else(|| io::Error::other("missing author"))?;
                if !matches!(role, "user" | "assistant" | "system" | "tool")
                    || !matches!(by, "human" | "model" | "harness" | "unknown")
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid Message role/author",
                    ));
                }
                terminal_text(out, id, false)?;
                writeln!(out, "\n{role} · {by}")?;
                started = true;
            }
            Some("block")
                if started && matches!(item["kind"].as_str(), Some("text" | "thinking")) =>
            {
                writeln!(out)?;
                if item["kind"] == "thinking" {
                    match item["encrypted"].as_bool() {
                        Some(true) => writeln!(out, "[encrypted thinking]")?,
                        Some(false) => writeln!(out, "[thinking]")?,
                        None => writeln!(out, "[thinking; encryption unknown]")?,
                    }
                }
            }
            Some("text") if started => terminal_text(
                out,
                item["data"]
                    .as_str()
                    .ok_or_else(|| io::Error::other("missing text piece"))?,
                true,
            )?,
            Some("end")
                if item["outcome"] == "error"
                    || (started
                        && item["outcome"] == "degraded"
                        && item["canonical_complete"] == false) =>
            {
                if started {
                    writeln!(out)?;
                }
                if item.get("recovery").is_some_and(|v| !v.is_null()) {
                    writeln!(out, "{}", polylogue_model::RecoveryQualification::NOTE)?;
                }
                return Ok(item);
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unexpected read stream frame",
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn exchange(frames: Vec<Value>) -> (io::Result<Value>, Vec<u8>) {
        let dir = tempfile::tempdir_in("/tmp").unwrap();
        let path = dir.path().join("socket");
        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let join = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = String::new();
            BufReader::new(stream.try_clone().unwrap())
                .read_line(&mut request)
                .unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&request).unwrap()["command"],
                "read"
            );
            for value in frames {
                frame(&mut stream, value).unwrap();
            }
        });
        let mut output = Vec::new();
        let result = request(&path, "message:synthetic:m", &mut output);
        join.join().unwrap();
        (result, output)
    }
    fn initial() -> Vec<Value> {
        vec![
            json!({"type":"message","id":"synthetic:m","role":"assistant","written_by":"model"}),
            json!({"type":"block","kind":"text"}),
            json!({"type":"text","data":"retained text"}),
        ]
    }
    #[test]
    fn partial_transport_cannot_become_a_completed_read() {
        for end in [
            None,
            Some(json!({"type":"end","outcome":"ok"})),
            Some(json!({"type":"end","outcome":"degraded","canonical_complete":true})),
        ] {
            let mut frames = initial();
            if let Some(end) = end {
                frames.push(end);
            }
            let (result, output) = exchange(frames);
            assert!(result.is_err());
            assert!(String::from_utf8(output).unwrap().contains("retained text"));
        }
        let mut frames = initial();
        frames.push(json!({"type":"end","outcome":"degraded","canonical_complete":false,"missing":["history"]}));
        let (result, _) = exchange(frames);
        assert_eq!(result.unwrap()["canonical_complete"], false);
    }
    #[test]
    fn message_header_must_match_the_exact_requested_opaque_identity() {
        for id in ["synthetic:other", "m", "", "synthetic:m:extra"] {
            let mut frames = initial();
            frames[0]["id"] = json!(id);
            let (result, output) = exchange(vec![frames.remove(0)]);
            assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
            assert!(output.is_empty());
        }
        let mut frames = initial();
        frames.push(json!({"type":"end","outcome":"degraded","canonical_complete":false}));
        let (result, output) = exchange(frames);
        assert_eq!(result.unwrap()["outcome"], "degraded");
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "synthetic:m\nassistant · model\n\nretained text\n"
        );
    }
    #[test]
    fn absent_reference_keeps_stdout_empty_and_retains_typed_failure() {
        let (result, output) = exchange(vec![failed(ErrorKind::NotFound, "reference absent")]);
        assert!(output.is_empty());
        assert_eq!(result.unwrap()["error"]["kind"], "not_found");
    }
    #[test]
    fn terminal_controls_cannot_rewrite_a_readers_display() {
        let mut output = Vec::new();
        terminal_text(&mut output, "hello\u{1b}[2J\r\u{85}\tworld\n", true).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "hello\\x1b[2J\\x0d\\x85\tworld\n"
        );
        let mut output = Vec::new();
        terminal_text(&mut output, "id\nwith\ttabs", false).unwrap();
        assert_eq!(String::from_utf8(output).unwrap(), "id\\x0awith\\x09tabs");
    }
}
