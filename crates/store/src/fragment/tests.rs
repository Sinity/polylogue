use super::*;
use polylogue_model::bundle::fragments::{
    TextCoverage, TextFragment, TextFragmentProvenance, TextFragmentRecord, TextSpanOrigin,
};
use polylogue_model::{Reference, SpanUnit};

pub(crate) fn record(
    start: u64,
    text: &str,
    total: Option<u64>,
    version: BlobHash,
) -> TextFragmentRecord {
    let container = Reference::parse("message:claude-code:s-m").unwrap();
    let end = start + text.len() as u64;
    let fragment = TextFragment {
        source_version: TextSourceVersion::ContentHash {
            record: container.clone(),
            hash: version,
        },
        container: container.clone(),
        coverage: TextCoverage {
            start,
            end,
            total_bytes: total,
        },
        text: text.into(),
        text_hash: BlobHash::of(text.as_bytes()),
        provenance: TextFragmentProvenance {
            spans: vec![TextSpanOrigin {
                reference: Reference::Span {
                    container: Box::new(container),
                    unit: SpanUnit::Bytes,
                    start,
                    end,
                },
                start,
                end,
            }],
            sources: vec![],
            imports: vec![],
        },
    };
    TextFragmentRecord {
        reference: fragment.reference().to_string(),
        commit: 900,
        fragment,
    }
}

fn output(record: TextFragmentRecord) -> Output {
    Output {
        fragments: vec![FragmentRow {
            offset: 2,
            end: 9,
            record,
        }],
        ..Output::default()
    }
}
fn database() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("PRAGMA foreign_keys=ON; CREATE TABLE input_ref(pk INTEGER PRIMARY KEY,id TEXT,format TEXT); INSERT INTO input_ref VALUES(1,'receiving-input','polylogue-bundle');").unwrap();
    conn.execute_batch(include_str!("../schema/text_fragments.sql"))
        .unwrap();
    conn
}

#[test]
fn fragment_preparation_binds_actual_ranges_and_never_uses_sender_commit() {
    let mut conn = database();
    let mut output = output(record(
        0,
        "hé",
        Some(6),
        BlobHash::of(b"complete provider value"),
    ));
    let prepared = PreparedFragments::prepare(&output).unwrap();
    prepared.validate(&output).unwrap();
    {
        let tx = conn.transaction().unwrap();
        prepared.apply(&tx, 1, 1, 9, |_| Ok(Some(130))).unwrap();
        tx.rollback().unwrap();
    }
    assert_eq!(
        conn.query_row("SELECT count(*) FROM text_fragment_source", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    for _ in 0..2 {
        prepared.apply(&conn, 1, 1, 9, |_| Ok(Some(130))).unwrap();
    }
    let actual: (i64, i64, i64, i64) = conn
        .query_row(
            "SELECT chain,start,end,admitted FROM text_fragment_source",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .unwrap();
    assert_eq!(actual, (1, 2, 9, 130));
    assert_eq!(
        conn.query_row("SELECT count(*) FROM text_fragment", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    output.fragments[0].record.commit = 901;
    assert!(prepared.validate(&output).is_err());
    output.fragments[0].record.commit = 900;
    prepared.validate(&output).unwrap();
    assert!(prepared.apply(&conn, 1, 1, 8, |_| Ok(None)).is_err());
}

#[test]
fn fragment_conflicts_remain_distinct_claims_and_retraction_is_per_source() {
    let conn = database();
    conn.execute(
        "INSERT INTO input_ref VALUES(2,'other-input','polylogue-bundle')",
        [],
    )
    .unwrap();
    let version = BlobHash::of(b"canonical version");
    for text in ["alpha", "other"] {
        let output = output(record(0, text, Some(5), version));
        PreparedFragments::prepare(&output)
            .unwrap()
            .apply(&conn, 1, 1, 9, |_| Ok(None))
            .unwrap();
    }
    let output = output(record(0, "alpha", Some(5), version));
    PreparedFragments::prepare(&output)
        .unwrap()
        .apply(&conn, 1, 2, 9, |_| Ok(Some(200)))
        .unwrap();
    PreparedFragments::prepare(&output)
        .unwrap()
        .apply(&conn, 2, 1, 9, |_| Ok(Some(300)))
        .unwrap();
    retract(&conn, 1, 1).unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM text_fragment", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM text_fragment_source", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        2
    );
    retract(&conn, 1, 2).unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM text_fragment", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        conn.query_row("SELECT input_pk FROM text_fragment_source", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        2
    );
    retract(&conn, 2, 1).unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM text_fragment", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn malformed_fragments_fail_before_writer_apply() {
    let original = output(record(0, "valid", None, BlobHash::of(b"source")));
    for case in 0..4 {
        let mut bad = original.clone();
        match case {
            0 => bad.fragments[0].record.fragment.text_hash = BlobHash::of(b"forged"),
            1 => bad.fragments[0].offset = 10,
            2 => bad.fragments[0].end = u64::MAX,
            3 => bad.fragments[0].record.reference = "span:message:claude-code:s-m#b=1-5".into(),
            _ => unreachable!(),
        }
        assert!(PreparedFragments::prepare(&bad).is_err(), "case {case}");
    }
    // Foreign UTF-8 coordinates retain their full u64 domain independently of
    // actual local byte positions, which use SQLite's checked integer domain.
    let high = output(record(
        u64::MAX - 5,
        "valid",
        Some(u64::MAX),
        BlobHash::of(b"source"),
    ));
    let conn = database();
    PreparedFragments::prepare(&high)
        .unwrap()
        .apply(&conn, 1, 1, 9, |_| Ok(None))
        .unwrap();
    let end: Vec<u8> = conn
        .query_row("SELECT end FROM text_fragment", [], |r| r.get(0))
        .unwrap();
    assert_eq!(end, u64::MAX.to_be_bytes());
}

#[test]
fn streamed_chain_scratch_preserves_prepared_fragment_claims() {
    let output = output(record(
        0,
        "selected",
        None,
        BlobHash::of(b"complete source"),
    ));
    let expected = output.clone();
    let mut step = crate::ParseStep::new(output, Default::default(), 0..9, true);
    let mut reading = crate::writer::ChainReading::new().unwrap();
    reading.push(&mut step).unwrap();
    let mut visits = 0;
    reading
        .visit_source(|step| {
            assert_eq!(step.complete_output()?, expected);
            visits += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(visits, 1);
}
