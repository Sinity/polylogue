use super::*;

fn old_digest(version: i64, json: &str) -> Option<blake3::Hash> {
    if version != 1 {
        return None;
    }
    let names: Vec<String> = serde_json::from_str(json).ok()?;
    let mut previous = None;
    for name in &names {
        let hash = crate::encoding::unhex(name).ok()?;
        if hash.len() != 32
            || crate::encoding::hex(&hash) != *name
            || previous.as_ref().is_some_and(|previous| previous >= &hash)
        {
            return None;
        }
        previous = Some(hash);
    }
    if serde_json::to_string(&names).ok()? != json {
        return None;
    }
    let mut hash = blake3::Hasher::new();
    hash.update(b"polylogue-vector-deletion-v1\0");
    hash.update(json.as_bytes());
    Some(hash.finalize())
}

#[test]
fn canonical_validation_and_digest_match_the_previous_codec() {
    for size in 0..128 {
        let names: Vec<_> = (0..size).map(|n| format!("{n:064x}")).collect();
        let canonical = serde_json::to_string(&names).unwrap();
        assert_eq!(
            super::super::digest(1, &canonical).ok(),
            old_digest(1, &canonical)
        );
        for json in [
            format!(" {canonical}"),
            format!("{canonical}\n"),
            canonical.replace(',', ", "),
            serde_json::to_string_pretty(&names).unwrap(),
            canonical.replace('0', "\\u0030"),
            canonical.replace('a', "A"),
        ] {
            assert_eq!(
                super::super::digest(1, &json).ok(),
                old_digest(1, &json),
                "{json}"
            );
        }
    }
    for json in [
        "null",
        "{}",
        "[null]",
        "[1]",
        "[\"00\"]",
        "[\"éé\"]",
        "[",
        "[]{}",
        "[ ]",
    ] {
        assert!(super::super::digest(1, json).is_err(), "{json}");
    }
    let name = "00".repeat(32);
    let next = "01".repeat(32);
    for names in [vec![&name, &name], vec![&next, &name]] {
        let json = serde_json::to_string(&names).unwrap();
        assert!(super::super::digest(1, &json).is_err());
    }
    assert!(super::super::digest(2, "[]").is_err());
}

#[test]
fn large_arrays_visit_in_order_and_callback_errors_keep_their_type() {
    use std::fmt::Write as _;
    let mut json = String::from("[");
    for n in 0..100_000 {
        if n != 0 {
            json.push(',');
        }
        write!(&mut json, "\"{n:064x}\"").unwrap();
    }
    json.push(']');
    let mut count = 0u64;
    visit(1, &json, |hash| {
        assert!(hash[..24].iter().all(|b| *b == 0));
        assert_eq!(u64::from_be_bytes(hash[24..].try_into().unwrap()), count);
        count += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 100_000);
    let mut count = 0;
    let result = visit(1, &json, |_| {
        count += 1;
        if count == 27 {
            return Err(StoreError::NotFound("synthetic failure".into()));
        }
        Ok(())
    });
    assert!(matches!(result, Err(StoreError::NotFound(message)) if message == "synthetic failure"));
    assert_eq!(count, 27);
}
