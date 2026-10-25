use super::*;
use crate::excise::text_matches::TextMatches;

#[test]
fn equal_matches_deduplicate_but_overlaps_and_patch_order_remain_distinct() {
    let plan = Plan::new().unwrap();
    plan.matches("input", 1, &[(3, 6), (2, 5), (2, 5)]).unwrap();
    plan.patch("input", 1, 4, b"Q").unwrap();
    plan.patch("input", 1, 0, b"AA").unwrap();
    plan.normalize("input", 1).unwrap();
    assert_eq!(plan.counts().unwrap().get("input"), Some(&2));
    let mut changed = Vec::new();
    plan.changed(|input, chain, from, to| {
        changed.push((input.to_owned(), chain, from, to));
        Ok(())
    })
    .unwrap();
    assert_eq!(
        changed,
        vec![
            ("input".into(), 1, 2, 5),
            ("input".into(), 1, 3, 6),
            ("input".into(), 1, 4, 5),
            ("input".into(), 1, 0, 2),
        ]
    );
    let mut bytes = *b"abcdefgh";
    let key = ("input".into(), 1);
    plan.edits(&key, 0)
        .unwrap()
        .rewrite(0, &mut bytes, b'*')
        .unwrap();
    assert_eq!(&bytes, b"AA**Q*gh");
    let mut part = *b"def";
    plan.edits(&key, 3)
        .unwrap()
        .rewrite(3, &mut part, b'*')
        .unwrap();
    assert_eq!(&part, b"*Q*");
}

#[test]
fn blanking_keeps_line_breaks_across_chunks() {
    let plan = Plan::new().unwrap();
    plan.matches("input", 1, &[(1, 7)]).unwrap();
    plan.normalize("input", 1).unwrap();
    let key = ("input".into(), 1);
    let mut edits = plan.edits(&key, 0).unwrap();
    let mut first = *b"ab\nd";
    let mut second = *b"efgh";
    edits.blank(0, &mut first, b' ').unwrap();
    edits.blank(4, &mut second, b' ').unwrap();
    assert_eq!(&first, b"a \n ");
    assert_eq!(&second, b"   h");
    assert!(edits.blank(0, &mut first, b' ').is_err());
}

#[test]
fn affected_observations_preserve_overlap_and_patch_order_before_extent_changes() {
    let archive = Connection::open_in_memory().unwrap();
    archive
        .execute_batch(
            "CREATE TABLE extent(input TEXT,chain INTEGER,offset INTEGER,
        length INTEGER,blob BLOB,blob_offset INTEGER);",
        )
        .unwrap();
    let hash = BlobHash([1; 32]);
    archive
        .execute(
            "INSERT INTO extent VALUES ('input',1,0,10,?1,7)",
            [&hash.0[..]],
        )
        .unwrap();
    archive
        .execute(
            "INSERT INTO extent VALUES ('other',4,0,2,?1,20)",
            [&hash.0[..]],
        )
        .unwrap();
    let plan = Plan::new().unwrap();
    plan.matches("input", 1, &[(2, 4), (3, 5)]).unwrap();
    plan.patch("input", 1, 0, b"XY").unwrap();
    plan.stage_affected(&archive).unwrap();
    let mut blobs = Vec::new();
    plan.visit_blobs(|hash| {
        blobs.push(hash);
        Ok(())
    })
    .unwrap();
    assert_eq!(blobs, [hash]);
    let mut observations = Vec::new();
    plan.visit_observations(|input, value| {
        observations.push((input.to_owned(), value));
        Ok(())
    })
    .unwrap();
    assert_eq!(
        observations,
        [(
            "input".into(),
            serde_json::json!([
                [crate::excise::hex(&hash.0), 9, 2],
                [crate::excise::hex(&hash.0), 10, 2],
                [crate::excise::hex(&hash.0), 7, 2]
            ])
        )]
    );
    plan.freeze_extents(&archive, hash).unwrap();
    archive.execute("DELETE FROM extent", []).unwrap();
    let mut inputs = Vec::new();
    plan.visit_extents(|e| {
        inputs.push(e.input);
        if plan.placement(&hash)?.is_none() {
            plan.remember(&hash, 0, 30)?;
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(inputs, ["input", "other"]);
    assert_eq!(plan.placement(&hash).unwrap(), Some((0, 30)));
}

#[test]
fn hits_inside_a_rejected_value_never_become_matches() {
    for (source, expected) in [
        (r#"{"a":"the secret here","b":"secret"}"#, 2),
        (r#"{"secret":"kept"}"#, 0),
    ] {
        let plan = Plan::new().unwrap();
        let mut hits = plan.hits("input", 1);
        let mut scanner = match TextMatches::new("secret", &mut hits) {
            Ok(scanner) => scanner,
            Err(_) => panic!("scanner"),
        };
        let fed = scanner.feed(source.as_bytes(), &mut hits);
        let finished = fed.and_then(|()| scanner.finish(&mut hits));
        match finished {
            Ok(accepted) => {
                hits.finish(accepted.scope).unwrap();
                assert_eq!(
                    plan.counts().unwrap().get("input").copied().unwrap_or(0),
                    expected,
                    "{source}"
                );
            }
            Err(_) => assert_eq!(expected, 0, "{source}"),
        }
    }
}

#[test]
fn a_huge_value_keeps_bounded_matcher_state() {
    let plan = Plan::new().unwrap();
    let mut hits = plan.hits("input", 1);
    let mut scanner = TextMatches::new("secret", &mut hits).unwrap_or_else(|_| panic!("scanner"));
    assert!(scanner.feed(br#"{"a":""#, &mut hits).is_ok());
    let chunk = b"secret padding ".repeat(4096);
    for _ in 0..16 {
        assert!(scanner.feed(&chunk, &mut hits).is_ok());
        assert!(scanner.held_bytes() < 64 * 1024);
    }
    assert!(scanner.feed(br#""}"#, &mut hits).is_ok());
    let accepted = scanner
        .finish(&mut hits)
        .unwrap_or_else(|_| panic!("finish"));
    hits.finish(accepted.scope).unwrap();
    assert_eq!(plan.counts().unwrap()["input"], 16 * 4096);
}
