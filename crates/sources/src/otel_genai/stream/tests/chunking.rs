use super::*;
use std::ops::Range;

#[derive(Default)]
struct Records {
    items: Vec<(Range<u64>, Vec<u8>, String)>,
    checkpoint: Option<usize>,
}
impl ItemSink for Records {
    fn push(
        &mut self,
        range: Range<u64>,
        key: Vec<u8>,
        _: Option<&str>,
        raw: &[u8],
    ) -> Result<(), String> {
        self.items
            .push((range, key, String::from_utf8(raw.to_vec()).unwrap()));
        Ok(())
    }
    fn begin(&mut self) -> Result<(), String> {
        assert!(self.checkpoint.is_none());
        self.checkpoint = Some(self.items.len());
        Ok(())
    }
    fn commit(&mut self) -> Result<(), String> {
        self.checkpoint.take().unwrap();
        Ok(())
    }
    fn rollback(&mut self) -> Result<(), String> {
        self.items.truncate(self.checkpoint.take().unwrap());
        Ok(())
    }
}

#[test]
fn every_small_read_boundary_keeps_exact_span_bytes_across_wrappers_and_bad_lines() {
    let first = r#"{"resourceSpans":[{"resource":{"ignored":"🦆"},"scopeSpans":[{"spans":[{"spanId":"a","name":"é\\\"x"}]},{"spans":[{"spanId":"b"}]}]},{"scopeSpans":[{"spans":[{"spanId":"c"}]}]}]}"#;
    let last = r#"{"resourceSpans":[{"scopeSpans":[{"spans":[{"spanId":"last"}]}]}]}"#;
    let bad_utf8 = b"{\"resourceSpans\":[{\"scopeSpans\":[{\"spans\":[{\"spanId\":\"bad\",\"name\":\"\xff\"}]}]}]}";
    let surrogate =
        br#"{"resourceSpans":[{"scopeSpans":[{"spans":[{"spanId":"bad","name":"\ud800"}]}]}]}"#;
    for bad in [bad_utf8.as_slice(), surrogate.as_slice()] {
        let mut bytes = format!("{first}\r\n\n").into_bytes();
        bytes.extend(bad);
        bytes.extend(format!("\n{last}").bytes());
        let mut expected = None;
        for capacity in 1..=65 {
            let mut input = std::io::BufReader::with_capacity(capacity, Cursor::new(&bytes));
            let mut sink = Records::default();
            let (stats, refused) = scan(&mut input, &mut sink).unwrap();
            assert_eq!(stats.elements, 4);
            assert_eq!(refused, 1);
            assert!(sink.checkpoint.is_none());
            let ids: Vec<_> = sink
                .items
                .iter()
                .map(|(range, _, raw)| {
                    assert_eq!(
                        raw.as_bytes(),
                        &bytes[range.start as usize..range.end as usize]
                    );
                    serde_json::from_str::<Json>(raw).unwrap()["spanId"]
                        .as_str()
                        .unwrap()
                        .to_owned()
                })
                .collect();
            assert_eq!(ids, ["a", "b", "c", "last"]);
            if let Some(expected) = &expected {
                assert_eq!(&sink.items, expected, "capacity {capacity}");
            } else {
                expected = Some(sink.items);
            }
        }
    }
}
