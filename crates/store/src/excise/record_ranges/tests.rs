pub(super) mod oracle;

use super::*;
use serde_json::{Value, json};

fn rewritten(bytes: &[u8], ranges: &[Range]) -> Vec<u8> {
    let planned = plan(bytes, ranges).unwrap();
    let mut out = bytes.to_vec();
    for &(a, b) in &planned.ranges {
        for byte in &mut out[a..b] {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
    }
    assert_eq!(out.len(), bytes.len());
    for (index, &byte) in bytes.iter().enumerate() {
        if byte == b'\n' || !planned.ranges.iter().any(|&(a, b)| a <= index && index < b) {
            assert_eq!(out[index], byte);
        }
    }
    out
}

#[test]
fn every_subset_of_array_members_preserves_all_other_values_and_syntax() {
    for count in 0..=7 {
        let values: Vec<Value> = (0..count).map(|n| json!({"number":n})).collect();
        let bytes =
            serde_json::to_vec_pretty(&json!({"records":values,"untouched":"still here"})).unwrap();
        for mask in 0..1usize << count {
            let mut ranges = Vec::new();
            for (index, _) in values.iter().enumerate() {
                if mask & (1 << index) == 0 {
                    continue;
                }
                // Locate the object by its scalar, then find its surrounding
                // braces; test values have exactly one field and no braces in text.
                let key = format!("\"number\": {index}");
                let at = std::str::from_utf8(&bytes).unwrap().find(&key).unwrap();
                let start = bytes[..at].iter().rposition(|&b| b == b'{').unwrap();
                let end = at + bytes[at..].iter().position(|&b| b == b'}').unwrap() + 1;
                ranges.push((start, end));
            }
            let got: Value = serde_json::from_slice(&rewritten(&bytes, &ranges)).unwrap();
            let expected: Vec<_> = values
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) == 0)
                .map(|(_, v)| v.clone())
                .collect();
            assert_eq!(
                got,
                json!({"records":expected,"untouched":"still here"}),
                "count={count}, mask={mask}"
            );
        }
    }
}

#[test]
fn nested_arrays_and_jsonl_keep_exact_unaffected_bytes() {
    let bytes =
        br#"{"before":"keep","nested":[[{"drop":1},{"keep":2}],{"keep":3}],"after":"keep"}"#;
    let needle = br#"{"drop":1}"#;
    let a = bytes
        .windows(needle.len())
        .position(|v| v == needle)
        .unwrap();
    let got: Value = serde_json::from_slice(&rewritten(bytes, &[(a, a + needle.len())])).unwrap();
    assert_eq!(
        got,
        json!({"before":"keep","nested":[[{"keep":2}],{"keep":3}],"after":"keep"})
    );
    let bytes = b"  {\"a\":1} \n {\"b\":2}\n";
    let result = rewritten(bytes, &[(0, 12)]);
    assert_eq!(
        serde_json::from_slice::<Value>(&result).unwrap(),
        json!({"b":2})
    );
}

#[test]
fn partial_scalars_and_bad_ranges_are_refused() {
    let (bytes, needle) = (br#"[{"text":"1234"}]"#.as_slice(), b"23".as_slice());
    let at = bytes
        .windows(needle.len())
        .position(|v| v == needle)
        .unwrap();
    assert!(plan(bytes, &[(at, at + needle.len())]).is_err());
    assert!(plan(b"[1]", &[(0, 10)]).is_err());
    assert!(plan(b"[1] trailing", &[]).is_err());
    assert!(plan(b"{", &[]).is_err());
}

#[test]
fn every_subset_of_object_values_preserves_other_members_and_original_coordinates() {
    use std::collections::BTreeMap;
    let bytes=r#"{"first":"","escaped\"\\key":"λ🙂\n\"quoted\"","nested":[{"inside":[1,2]},3],"object":{"child":false},"last":null}"#.as_bytes();
    let fields: BTreeMap<String, &serde_json::value::RawValue> =
        serde_json::from_slice(bytes).unwrap();
    let keys: Vec<_> = fields.keys().cloned().collect();
    for mask in 0..1usize << keys.len() {
        let mut ranges = Vec::new();
        let mut expected: Value = serde_json::from_slice(bytes).unwrap();
        for (i, key) in keys.iter().enumerate() {
            if mask & (1 << i) == 0 {
                continue;
            }
            let raw = fields[key];
            let start = (raw.get().as_ptr() as usize) - (bytes.as_ptr() as usize);
            ranges.push((start, start + raw.get().len()));
            expected.as_object_mut().unwrap().remove(key);
        }
        let after = rewritten(bytes, &ranges);
        assert_eq!(
            serde_json::from_slice::<Value>(&after).unwrap(),
            expected,
            "mask={mask}"
        );
        for (key, raw) in &fields {
            if expected.get(key).is_none() {
                continue;
            }
            let start = (raw.get().as_ptr() as usize) - (bytes.as_ptr() as usize);
            assert_eq!(&after[start..start + raw.get().len()], raw.get().as_bytes());
        }
    }
}
