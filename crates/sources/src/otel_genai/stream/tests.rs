use super::*;
use std::io::{BufRead, Cursor};

struct CheckedInput {
    bytes: Cursor<Vec<u8>>,
    rewinds: usize,
    fail_at: Option<u64>,
}
impl Read for CheckedInput {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.fail_at.is_some_and(|at| self.bytes.position() >= at) {
            return Err(io::Error::other("synthetic read failure"));
        }
        let n = out.len().min(1);
        self.bytes.read(&mut out[..n])
    }
}
impl BufRead for CheckedInput {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.bytes.fill_buf()
    }
    fn consume(&mut self, n: usize) {
        self.bytes.consume(n);
    }
}
impl Input for CheckedInput {
    fn rewind(&mut self) -> io::Result<()> {
        self.rewinds += 1;
        self.bytes.set_position(0);
        Ok(())
    }
}
#[derive(Default)]
struct Counter {
    accepted: u64,
    checkpoint: Option<u64>,
    fail_at: Option<u64>,
}
impl ItemSink for Counter {
    fn push(
        &mut self,
        _: std::ops::Range<u64>,
        _: Vec<u8>,
        _: Option<&str>,
        _: &serde_json::value::RawValue,
    ) -> Result<(), String> {
        self.accepted += 1;
        if self.fail_at == Some(self.accepted) {
            return Err("synthetic sink failure".into());
        }
        Ok(())
    }
    fn begin(&mut self) -> Result<(), String> {
        assert!(self.checkpoint.is_none());
        self.checkpoint = Some(self.accepted);
        Ok(())
    }
    fn commit(&mut self) -> Result<(), String> {
        self.checkpoint.take().unwrap();
        Ok(())
    }
    fn rollback(&mut self) -> Result<(), String> {
        self.accepted = self.checkpoint.take().unwrap();
        Ok(())
    }
}

#[test]
fn input_and_sink_failures_abort_without_jsonl_fallback() {
    let bytes =
        br#"{"resourceSpans":[{"scopeSpans":[{"spans":[{"spanId":"a"},{"spanId":"b"}]}]}]}"#;
    for input_failure in [true, false] {
        let mut input = CheckedInput {
            bytes: Cursor::new(bytes.to_vec()),
            rewinds: 0,
            fail_at: input_failure.then_some(65),
        };
        let mut sink = Counter {
            fail_at: (!input_failure).then_some(2),
            ..Default::default()
        };
        let error = scan(&mut input, &mut sink).unwrap_err();
        if input_failure {
            assert!(matches!(error, ScanError::Input(_)), "{error:?}");
        } else {
            assert!(matches!(error, ScanError::Callback(_)), "{error:?}");
        }
        assert_eq!(input.rewinds, 0);
        assert_eq!(sink.accepted, 0);
        assert!(sink.checkpoint.is_none());
    }
}

mod chunking;
mod resource;
