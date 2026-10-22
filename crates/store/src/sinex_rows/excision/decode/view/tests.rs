use super::*;
use crate::sinex_rows::{ExcisionPublication, Unit};

fn manifest(size: usize) -> ExcisionPublication {
    let mut value = ExcisionPublication {
        version: 1,
        replacements: (0..size)
            .map(|n| Replacement {
                unit: Unit {
                    commit_seq: n as i64 + 1,
                    kind: "note".into(),
                    key: format!("{n}:\0\"é🦀"),
                },
                batch: (n % 2 == 0).then(|| format!("blake3:{}", "ab".repeat(32))),
                bytes: u64::from(n % 2 == 0),
                rows: u64::from(n % 2 == 0),
                blobs: if n % 2 == 0 {
                    vec![format!("blake3:{}", "01".repeat(32))]
                } else {
                    vec![]
                },
            })
            .collect(),
        retired_batches: vec![format!("blake3:{}", "cd".repeat(32)); size % 3],
        digest: String::new(),
    };
    value.digest = value.computed_digest().unwrap();
    value
}

#[test]
fn borrowed_arrays_preserve_the_existing_digest_and_typed_entries() {
    for size in 0..80 {
        let expected = manifest(size);
        let mut value = serde_json::json!({"publication":expected});
        value["publication"]["future"] = serde_json::json!({"not":"hashed"});
        for replacement in value["publication"]["replacements"].as_array_mut().unwrap() {
            replacement["unknown"] = serde_json::json!([1, 2, 3]);
        }
        let json = serde_json::to_string_pretty(&value).unwrap();
        let view = View::decode(&json).unwrap().unwrap();
        assert_eq!(view.computed_digest().unwrap(), expected.digest);
        let mut actual = vec![];
        view.each_replacement(|replacement| {
            actual.push(replacement.clone());
            Ok(())
        })
        .unwrap();
        assert_eq!(actual, expected.replacements);
        assert_eq!(super::super::publication(&json).unwrap(), Some(expected));
        let array = view.0.replacements.get();
        assert!(array.as_ptr() >= json.as_ptr());
        assert!((array.as_ptr() as usize + array.len()) <= json.as_ptr() as usize + json.len());
    }
}

#[test]
fn legacy_duplicate_null_and_malformed_values_agree_with_owned_decoding() {
    let expected = manifest(3);
    let raw = serde_json::to_string(&expected).unwrap();
    for json in [
        "[]".into(),
        "null".into(),
        "false".into(),
        "123".into(),
        r#""legacy""#.into(),
        r#"{"ignored":[]} "#.into(),
        format!(r#"{{"publication":null,"publication":{raw}}}"#),
        format!(r#"{{"publ\u0069cation":{raw}}}"#),
        format!(r#"{{"publication":{raw},"publication":null}}"#),
        format!(r#"{{"publication":{raw}}} garbage"#),
        r#"{"publication":{},"ignored":[1,}"#.into(),
    ] {
        let owned = super::super::publication(&json);
        let borrowed = View::decode(&json);
        assert_eq!(owned.is_ok(), borrowed.is_ok(), "{json}");
        if let (Ok(owned), Ok(borrowed)) = (owned, borrowed) {
            assert_eq!(owned.is_some(), borrowed.is_some());
        }
    }
    for field in ["version", "digest", "replacements", "retired_batches"] {
        let duplicate = format!(
            r#"{{"publication":{{"{field}":null,{} }}}}"#,
            &raw[1..raw.len() - 1]
        );
        assert!(super::super::publication(&duplicate).is_err());
        assert!(View::decode(&duplicate).is_err());
    }
}

#[test]
fn invalid_final_entries_and_hashes_expose_no_publication_to_consumers() {
    for mutation in 0..10 {
        let mut publication = manifest(32);
        let last = publication.replacements.last_mut().unwrap();
        match mutation {
            0 => last.unit.commit_seq = 0,
            1 => last.unit.key.clear(),
            2 => last.unit.kind = "unknown".into(),
            3 => last.unit.commit_seq = 1,
            4 => last.bytes = 1,
            5 => last.batch = Some("wrong hash".into()),
            6 => last.blobs.push("wrong blob".into()),
            7 => publication
                .retired_batches
                .push("wrong retired hash".into()),
            8 => publication.version = 2,
            _ => {}
        }
        publication.digest = publication.computed_digest().unwrap();
        if mutation == 9 {
            publication.digest = "wrong digest".into();
        }
        let json = serde_json::json!({"publication":publication}).to_string();
        assert!(
            super::super::publication(&json).is_err(),
            "mutation {mutation}"
        );
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE excision(commit_seq INTEGER,id TEXT,refused_json TEXT)")
            .unwrap();
        conn.execute(
            "INSERT INTO excision VALUES (100,'late-refusal',?1)",
            [&json],
        )
        .unwrap();
        let mut calls = 0;
        let result = crate::sinex_rows::excision::visit_publications(&conn, |_, view| {
            calls += 1;
            view.each_replacement(|_| Ok(()))
        });
        assert!(result.is_err(), "mutation {mutation}");
        assert_eq!(calls, 0, "mutation {mutation}");
    }
}

#[test]
fn large_historical_arrays_visit_in_order_and_preserve_callback_errors() {
    let publication = manifest(10_000);
    let json = serde_json::json!({"publication":publication}).to_string();
    let view = View::decode(&json).unwrap().unwrap();
    let mut count = 0;
    view.each_replacement(|replacement| {
        count += 1;
        assert_eq!(replacement.unit.commit_seq, count);
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 10_000);
    assert!(
        matches!(view.each_replacement(|_| Err(StoreError::NotFound("synthetic".into()))), Err(StoreError::NotFound(message)) if message == "synthetic")
    );
}
