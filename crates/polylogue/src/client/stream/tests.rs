use super::*;

struct Bytes<'a>(&'a [u8]);
impl Read for Bytes<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let count = output.len().min(self.0.len()).min(1);
        output[..count].copy_from_slice(&self.0[..count]);
        self.0 = &self.0[count..];
        Ok(count)
    }
}
fn refusal() -> Answer<serde_json::Value> {
    Answer::Error {
        error: crate::protocol::failure("scope_changed", "scope changed: λ".into(), None),
        notes: vec!["acquire a new reader".into()],
        next: vec![],
    }
}

#[test]
fn an_early_canonical_refusal_keeps_its_kind_and_notes_through_split_reads() {
    let expected = refusal();
    let mut bytes = serde_json::to_vec(&expected).unwrap();
    bytes.push(b'\n');
    let result = receive(Bytes(&bytes), |_| panic!("a refusal is not a record"));
    let Err(Error::Answer(actual)) = result else {
        panic!("expected refusal")
    };
    assert_eq!(actual, expected);
}

#[test]
fn a_late_refusal_after_split_unicode_never_delivers_completion() {
    let mut bytes = Vec::new();
    crate::protocol::stream::event(&mut bytes, "begin", &serde_json::json!({"title":"λ"})).unwrap();
    crate::protocol::stream::event(&mut bytes, "error", &refusal()).unwrap();
    let mut events = Vec::new();
    let result = receive(Bytes(&bytes), |event| {
        events.push(event.event);
        Ok(false)
    });
    assert_eq!(events, ["begin"]);
    let Err(Error::Answer(actual)) = result else {
        panic!("expected late refusal")
    };
    assert_eq!(actual, refusal());
}
