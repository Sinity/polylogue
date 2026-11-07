use super::*;
use crate::read::VisitStart;

fn message(id: &str, session: &str) -> Message {
    text_message(id, session, Role::User, id, None)
}

fn records(
    reader: &crate::Reader<'_>,
    session: &str,
    parents: bool,
    start: usize,
) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    reader
        .select_messages(&sid(session), parents)
        .unwrap()
        .visit_from_with_record_evidence(
            VisitStart {
                message: start,
                key: None,
            },
            |message, _, _, order| {
                out.push((message.id.to_string(), order));
                std::ops::ControlFlow::Continue(())
            },
        )
        .unwrap();
    out
}

#[test]
fn record_ordinals_precede_subsets_and_use_record_positions_before_ids() {
    let w = test_archive();
    write(
        &w,
        batch(
            "middle",
            0,
            0,
            vec![
                row(20, message("z", "s")),
                row(20, message("a", "s")),
                row(90, message("m", "s")),
            ],
            &["s"],
        ),
    );
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        records(&reader, "s", false, 0),
        vec![
            ("claude-code:z".into(), 0),
            ("claude-code:a".into(), 1),
            ("claude-code:m".into(), 2)
        ]
    );
    assert_eq!(
        records(&reader, "s", false, 1),
        vec![("claude-code:a".into(), 1), ("claude-code:m".into(), 2)]
    );
    for (id, expected) in [("z", 0), ("a", 1), ("m", 2)] {
        assert_eq!(
            reader.message_order(&message(id, "s").id).unwrap(),
            Some(expected)
        );
    }
    assert_eq!(
        reader.message_order(&message("absent", "s").id).unwrap(),
        None
    );
    // A new earlier input changes the next snapshot, while the pinned read
    // keeps its original ordinals. These are not native source counters.
    write(
        &w,
        batch(
            "aaa-earlier",
            0,
            0,
            vec![row(0, message("new", "s"))],
            &["s"],
        ),
    );
    assert_eq!(
        reader.message_order(&message("z", "s").id).unwrap(),
        Some(0)
    );
    let next = pool.reader().unwrap();
    assert_eq!(next.message_order(&message("z", "s").id).unwrap(), Some(1));
}

#[test]
fn fork_prefixes_keep_parent_ordinals_and_the_childs_own_order() {
    let w = test_archive();
    write(
        &w,
        batch(
            "parent",
            0,
            0,
            vec![
                row(0, message("p0", "s")),
                row(20, message("p1", "s")),
                row(80, message("p2", "s")),
            ],
            &["s"],
        ),
    );
    let mut child = batch(
        "child",
        0,
        0,
        vec![row(10, message("c", "child"))],
        &["s", "child"],
    );
    child.parse[0].output.links.push(polylogue_model::Link {
        kind: LinkKind::Fork,
        parent: sid("s"),
        child: sid("child"),
        at: Some(message("p1", "s").id),
        call_id: None,
        evidence: "synthetic".into(),
    });
    write(&w, child);
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        records(&reader, "child", true, 0),
        vec![
            ("claude-code:p0".into(), 0),
            ("claude-code:p1".into(), 1),
            ("claude-code:c".into(), 0)
        ]
    );
    assert_eq!(
        records(&reader, "child", true, 1),
        vec![("claude-code:p1".into(), 1), ("claude-code:c".into(), 0)]
    );
}

#[test]
fn copied_parent_subsets_do_not_renumber_their_original_own_session() {
    let w = test_archive();
    write(
        &w,
        batch(
            "aaa-parent",
            0,
            0,
            vec![
                row(0, message("p0", "s")),
                row(20, message("p1", "s")),
                row(80, message("p2", "s")),
            ],
            &["s"],
        ),
    );
    let mut child = batch(
        "zzz-child",
        0,
        0,
        vec![
            row(20, message("p1", "s")),
            row(80, message("p2", "s")),
            row(90, message("c", "child")),
        ],
        &["s", "child"],
    );
    child.parse[0].output.links.push(polylogue_model::Link {
        kind: LinkKind::Copy,
        parent: sid("s"),
        child: sid("child"),
        at: None,
        call_id: None,
        evidence: "synthetic".into(),
    });
    write(&w, child);
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        records(&reader, "child", true, 0),
        vec![
            ("claude-code:p1".into(), 1),
            ("claude-code:p2".into(), 2),
            ("claude-code:c".into(), 0)
        ]
    );
}

#[test]
fn metadata_order_has_an_id_tiebreak_and_never_reads_message_bodies() {
    let w = test_archive();
    write(
        &w,
        batch(
            "ties",
            0,
            0,
            vec![
                row(20, message("z", "s")),
                row(20, message("a", "s")),
                row(90, message("m", "s")),
            ],
            &["s"],
        ),
    );
    let index = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    index
        .execute("UPDATE message SET record_position=0", [])
        .unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    reader
        .conn()
        .unwrap()
        .authorizer(Some(|context: AuthContext<'_>| match context.action {
            AuthAction::Read {
                table_name: "block" | "text" | "message_version" | "message_projection",
                ..
            } => Authorization::Deny,
            _ => Authorization::Allow,
        }))
        .unwrap();
    assert_eq!(
        reader.message_order(&message("a", "s").id).unwrap(),
        Some(0)
    );
    assert_eq!(
        reader.message_order(&message("z", "s").id).unwrap(),
        Some(1)
    );
    reader
        .conn()
        .unwrap()
        .authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
}
