use super::*;
use polylogue_sources::selection::{PathPart, RecordRange, SelectedRaw};
use source::origins::RecordOrigins;
#[test]
fn consecutive_citations_keep_interleaved_suppliers_old_cursors_and_cancellation() {
    let a = RecordRange { start: 0, end: 10 };
    let b = RecordRange { start: 11, end: 20 };
    let mut origins = RecordOrigins {
        envelope: Some(a),
        ..Default::default()
    };
    for (key, record) in [("a", a), ("b", a), ("c", b), ("d", a)] {
        origins.fields.insert(
            vec![key.into()],
            SelectedRaw {
                record,
                path: vec![PathPart::Key(key.into())],
            },
        );
    }
    let flag = AtomicBool::new(false);
    let end = consecutive_origins(&origins, None, a, &flag)
        .unwrap()
        .unwrap();
    assert_eq!(end, vec!["b"]);
    assert_eq!(
        consecutive_origins(&origins, Some(&["a".into()]), a, &flag).unwrap(),
        Some(end.clone())
    );
    let (key, origin) = origins.next(Some(&end)).unwrap();
    assert_eq!(origin.record, b);
    assert_eq!(
        consecutive_origins(&origins, Some(key), b, &flag).unwrap(),
        Some(vec!["c".into()])
    );
    assert_eq!(
        consecutive_origins(&origins, Some(&["c".into()]), a, &flag).unwrap(),
        Some(vec!["d".into()])
    );
    assert!(matches!(
        consecutive_origins(&origins, None, a, &AtomicBool::new(true)),
        Err(Error::Cancelled)
    ));
}
