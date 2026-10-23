use super::*;

#[test]
fn equal_matches_deduplicate_but_overlaps_and_patch_order_remain_distinct() {
    let plan = Plan::new().unwrap();
    plan.matches("input", 1, &[(3, 6), (2, 5), (2, 5)]).unwrap();
    plan.patch("input", 1, 0, 4, b"Q").unwrap();
    plan.patch("input", 1, 1, 0, b"AA").unwrap();
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
fn many_chains_keep_complete_sorted_rows_and_stop_on_visitor_error() {
    let plan = Plan::new().unwrap();
    for chain in (0..4097).rev() {
        plan.matches("opaque\0input:é", chain, &[(2, 3), (7, 8)])
            .unwrap();
    }
    assert_eq!(
        plan.inputs().unwrap().into_iter().collect::<Vec<_>>(),
        ["opaque\0input:é"]
    );
    assert_eq!(plan.counts().unwrap()["opaque\0input:é"], 8194);
    let mut ordinal = 0;
    plan.changed(|input, chain, start, end| {
        assert_eq!(input, "opaque\0input:é");
        assert_eq!(chain, ordinal / 2);
        assert_eq!((start, end), if ordinal % 2 == 0 { (2, 3) } else { (7, 8) });
        ordinal += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(ordinal, 8194);
    let mut called = 0;
    assert!(
        plan.changed(|_, _, _, _| {
            called += 1;
            Err(StoreError::Invalid("stop".into()))
        })
        .is_err()
    );
    assert_eq!(called, 1);
    assert_eq!(
        plan.conn
            .pragma_query_value(None, "cache_size", |r| r.get::<_, i64>(0))
            .unwrap(),
        -1024
    );
    assert_eq!(
        plan.conn
            .pragma_query_value(None, "temp_store", |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert!(!plan.conn.is_autocommit());
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
    plan.patch("input", 1, 0, 0, b"XY").unwrap();
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
        // The source cursor stays usable while another plan table changes.
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
fn position_overflow_refuses_without_recording_a_placement() {
    let plan = Plan::new().unwrap();
    let hash = BlobHash([3; 32]);
    assert!(plan.remember(&hash, u64::MAX, 1).is_err());
    assert!(plan.placement(&hash).unwrap().is_none());
    assert!(plan.patch("input", 1, 0, u64::MAX, b"x").is_err());
}

mod matching;
