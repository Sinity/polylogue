use super::*;
use std::{
    cell::Cell,
    time::{Duration, Instant},
};
fn encoder() -> Encoder<Vec<u8>> {
    Encoder::new(Vec::new(), Instant::now() + Duration::from_secs(30))
}

#[test]
fn page_refuses_before_retaining_over_budget_leaf_and_stops_the_suffix() {
    let raw = serde_json::to_vec(&"a".repeat(super::super::LEAF - 2)).unwrap();
    let mut fields = OpaqueFields::new("attrs", true);
    let mut e = encoder();
    let attempted = Cell::new(0);
    let result = (|| -> io::Result<()> {
        for index in 0..1000 {
            attempted.set(attempted.get() + 1);
            fields.push(&mut e, &format!("field{index:04}"), &raw)?;
        }
        Ok(())
    })();
    assert_eq!(
        result.unwrap_err().to_string(),
        "selected opaque metadata exceeds admission"
    );
    let (retained, bytes) = fields.page.as_ref().unwrap();
    assert_eq!(attempted.get(), retained.len() + 1);
    assert!(attempted.get() < 1000);
    assert!(*bytes <= METADATA);
    assert!(!retained.contains_key(&format!("field{:04}", retained.len()).into()));
    // Nothing was serialized or handed to the downstream privacy clone yet.
    assert!(e.finish().unwrap().is_empty());
}

#[test]
fn direct_document_streams_without_collecting_and_sink_failure_stops_next_leaf() {
    struct Stop;
    impl Write for Stop {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::ErrorKind::WouldBlock.into())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut fields = OpaqueFields::new("data", false);
    assert!(fields.page.is_none());
    let mut e = Encoder::new(Stop, Instant::now() + Duration::from_secs(30));
    let mut acquired = 0;
    let result = (|| -> io::Result<()> {
        for key in ["first", "must-not-be-read"] {
            acquired += 1;
            fields.push(&mut e, key, b"null")?;
        }
        Ok(())
    })();
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::WouldBlock);
    assert_eq!(acquired, 1);
}

#[test]
fn streamed_document_and_sorted_page_envelope_preserve_the_same_fields() {
    for model_page in [false, true] {
        let mut e = encoder();
        e.syntax(b"{\"_\":null").unwrap();
        let mut fields = OpaqueFields::new("attrs", model_page);
        fields
            .push(&mut e, "z", br#"{"n":1e+02,"s":"\ud800"}"#)
            .unwrap();
        fields.push(&mut e, "a", b"null").unwrap();
        fields.finish(&mut e).unwrap();
        e.syntax(b"}").unwrap();
        let encoded = e.finish().unwrap();
        let mut result = RetainedJson::parse(&encoded).unwrap();
        if model_page {
            let inner = result["attrs"]["json"].text().unwrap().to_owned();
            result = RetainedJson::parse(inner.as_bytes()).unwrap();
            assert!(inner.starts_with("{\"a\":null,\"z\":"));
        } else {
            result = result["attrs"].clone();
        }
        assert_eq!(
            result,
            RetainedJson::parse(br#"{"a":null,"z":{"n":1e+02,"s":"\ud800"}}"#).unwrap()
        );
    }
}
