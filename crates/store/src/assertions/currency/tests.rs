//! Tests of the stored currency index and the assertion write path, run
//! through the real writer on a standalone archive.
use crate::fixture::authored::{operator_request, request};
use crate::fixture::writer::WriterArchive;
use crate::{ReadPool, StoreError};
use authored::{Header, History, Revision, Target};
use polylogue_model::{
    Author, Reference, RevisionReference, Timestamp,
    assertion::{
        AssertionFact, AssertionKind, AssertionLink, AssertionPatch, Change, LinkRole,
        NewAssertion, NewAssertionLink,
    },
    authorship::{Authorship, Channel},
    user::{UserOutcome, UserWrite},
};
use proptest::prelude::*;
use rusqlite::Connection;

fn r(text: &str) -> Reference {
    Reference::parse(text).unwrap()
}

fn statement(kind: AssertionKind, key: Option<&str>, value: serde_json::Value) -> NewAssertion {
    NewAssertion {
        kind,
        key: key.map(str::to_owned),
        value: Some(value),
        valid_until: None,
        links: vec![NewAssertionLink {
            role: LinkRole::About,
            reference: r("session:codex:subject"),
        }],
    }
}

struct World {
    archive: WriterArchive,
    next: u32,
}

impl World {
    fn new() -> Self {
        Self {
            archive: WriterArchive::new(),
            next: 0,
        }
    }
    fn id(&mut self) -> String {
        self.next += 1;
        format!("request-{:03}", self.next)
    }
    fn write(&mut self, write: UserWrite) -> crate::Result<UserOutcome> {
        let id = self.id();
        self.archive.writer.user(operator_request(id, write))
    }
    fn write_via(&mut self, via: Channel, write: UserWrite) -> crate::Result<UserOutcome> {
        let id = self.id();
        self.archive.writer.user(request(
            id,
            write,
            Authorship {
                author: Author::Operator,
                via,
            },
        ))
    }
    fn note(&mut self, text: &str) -> String {
        self.write(UserWrite::Assert {
            assertions: vec![statement(AssertionKind::Note, None, text.into())],
        })
        .unwrap()
        .id
    }
    fn revise(&mut self, root: &str, base: &str, text: &str) -> crate::Result<UserOutcome> {
        self.write(UserWrite::ReviseAssertions {
            bases: vec![RevisionReference::exact(root.into(), base.into())],
            patch: AssertionPatch {
                value: Change::Set(text.into()),
                ..Default::default()
            },
        })
    }
    fn withdraw(&mut self, target: &str) -> String {
        self.write(UserWrite::WithdrawAssertions {
            targets: vec![r(target)],
            reason: None,
        })
        .unwrap()
        .id
    }
    fn heads(&self, root: &str) -> Vec<String> {
        let reads = ReadPool::open(&self.archive.archive).unwrap();
        let reader = reads.reader().unwrap();
        reader
            .assertion_family_currency(&RevisionReference {
                root: root.into(),
                revision: None,
            })
            .unwrap()
            .map(|state| {
                state
                    .heads
                    .into_iter()
                    .map(|head| match head {
                        Reference::Assertion(revision) => revision.revision.unwrap(),
                        other => panic!("{other}"),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
    fn archive_db(&self) -> Connection {
        crate::archive::open_archive_writer(&self.archive.archive).unwrap()
    }
    fn count(&self, table: &str) -> i64 {
        self.archive_db()
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap()
    }
}

#[test]
fn withdrawing_a_middle_revision_leaves_the_newest_current_without_a_conflict() {
    let mut world = World::new();
    let r1 = world.note("first");
    let r2 = world.revise(&r1, &r1, "second").unwrap().id;
    let r3 = world.revise(&r1, &r2, "third").unwrap().id;
    assert_eq!(world.heads(&r1), vec![r3.clone()]);
    let withdraw_r2 = world.withdraw(&format!("assertion:{r1}@{r2}"));
    assert_eq!(world.heads(&r1), vec![r3.clone()]);
    // Withdrawing the withdrawal of R2 changes nothing visible: R3 stays.
    world.withdraw(&format!("assertion:{withdraw_r2}@{withdraw_r2}"));
    assert_eq!(world.heads(&r1), vec![r3.clone()]);
    // With R3 withdrawn, R2 is current again; with R2 withdrawn too, R1.
    world.withdraw(&format!("assertion:{r1}@{r3}"));
    assert_eq!(world.heads(&r1), vec![r2.clone()]);
    world.withdraw(&format!("assertion:{r1}@{r2}"));
    assert_eq!(world.heads(&r1), vec![r1.clone()]);
    // Withdrawing the root withdraws every revision.
    world.withdraw(&format!("assertion:{r1}"));
    assert!(world.heads(&r1).is_empty());
}

#[test]
fn a_command_on_a_stale_base_is_refused_and_a_forking_channel_makes_a_visible_conflict() {
    let mut world = World::new();
    let r1 = world.note("first");
    let r2 = world.revise(&r1, &r1, "second").unwrap().id;
    let before = world.count("assertion");
    match world.revise(&r1, &r1, "stale") {
        Err(StoreError::Stale { reference, current }) => {
            assert_eq!(reference, format!("assertion:{r1}@{r1}"));
            assert_eq!(current, vec![format!("assertion:{r1}@{r2}")]);
        }
        other => panic!("expected a stale refusal, got {other:?}"),
    }
    assert_eq!(world.count("assertion"), before);
    let fork = world
        .write_via(
            Channel::Markup,
            UserWrite::ReviseAssertions {
                bases: vec![RevisionReference::exact(r1.clone(), r1.clone())],
                patch: AssertionPatch {
                    value: Change::Set("forked".into()),
                    ..Default::default()
                },
            },
        )
        .unwrap()
        .id;
    let mut heads = vec![r2.clone(), fork.clone()];
    heads.sort();
    assert_eq!(world.heads(&r1), heads);
    let resolved = world
        .write(UserWrite::ReviseAssertions {
            bases: vec![
                RevisionReference::exact(r1.clone(), r2),
                RevisionReference::exact(r1.clone(), fork),
            ],
            patch: AssertionPatch {
                value: Change::Set("resolved".into()),
                ..Default::default()
            },
        })
        .unwrap()
        .id;
    assert_eq!(world.heads(&r1), vec![resolved]);
}

#[test]
fn a_second_write_into_a_held_slot_revises_and_an_equal_tag_is_already_there() {
    let mut world = World::new();
    let title = |text: &str| UserWrite::Assert {
        assertions: vec![statement(AssertionKind::Title, None, text.into())],
    };
    let first = world.write(title("first title")).unwrap();
    let second = world.write(title("second title")).unwrap();
    assert!(!second.repeat);
    assert_ne!(first.id, second.id);
    assert_eq!(world.heads(&first.id), vec![second.id.clone()]);
    let root: String = world
        .archive_db()
        .query_row(
            "SELECT root FROM assertion WHERE id=?1",
            [&second.id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(root, first.id);
    let tag = |value: serde_json::Value| UserWrite::Assert {
        assertions: vec![statement(
            AssertionKind::Tag,
            Some("flaky"),
            serde_json::json!({ "value": value }),
        )],
    };
    let tagged = world.write(tag(1.into())).unwrap();
    let before = world.count("assertion");
    let again = world.write(tag(serde_json::json!(1.0))).unwrap();
    assert!(again.repeat);
    assert_eq!(again.id, tagged.id);
    assert_eq!(world.count("assertion"), before);
    let other = world.write(tag(2.into())).unwrap();
    assert!(!other.repeat);
    assert_eq!(world.heads(&other.id), vec![other.id.clone()]);
    assert_eq!(world.heads(&tagged.id), vec![tagged.id.clone()]);
    // After its withdrawal the slot is free again.
    world.withdraw(&format!("assertion:{}", tagged.id));
    let rewritten = world.write(tag(1.into())).unwrap();
    assert!(!rewritten.repeat);
    assert_eq!(world.heads(&rewritten.id), vec![rewritten.id.clone()]);
}

#[test]
fn a_single_valued_mark_over_several_rows_is_refused_and_writes_nothing() {
    let mut world = World::new();
    let outcome = |resolved: Vec<Reference>| UserWrite::AssertQuery {
        query: "sessions".into(),
        assertion: NewAssertion {
            kind: AssertionKind::Outcome,
            key: None,
            value: Some("solved".into()),
            valid_until: None,
            links: Vec::new(),
        },
        resolved,
    };
    let before = world.count("assertion");
    match world.write(outcome(vec![r("session:codex:a"), r("session:codex:b")])) {
        Err(StoreError::Invalid(message)) => {
            assert!(message.starts_with("single_valued_bulk"), "{message}")
        }
        other => panic!("expected single_valued_bulk, got {other:?}"),
    }
    assert_eq!(world.count("assertion"), before);
    let first = world.write(outcome(vec![r("session:codex:a")])).unwrap();
    let second = world.write(outcome(vec![r("session:codex:a")])).unwrap();
    assert_eq!(world.heads(&first.id), vec![second.id]);
}

#[test]
fn a_request_commits_all_of_its_writes_or_none_and_numbers_them_from_one() {
    let mut world = World::new();
    let base = world.note("base");
    let newer = world.revise(&base, &base, "newer").unwrap().id;
    let before = world.count("assertion");
    let mut stale = statement(AssertionKind::Note, None, "third".into());
    stale.links.push(NewAssertionLink {
        role: LinkRole::Revises,
        reference: r(&format!("assertion:{base}@{base}")),
    });
    let refused = world.write(UserWrite::Assert {
        assertions: vec![
            statement(AssertionKind::Note, None, "one".into()),
            statement(AssertionKind::Note, None, "two".into()),
            stale,
        ],
    });
    assert!(
        matches!(refused, Err(StoreError::Stale { .. })),
        "{refused:?}"
    );
    assert_eq!(world.count("assertion"), before);
    assert_eq!(world.heads(&base), vec![newer]);
    let id = world.id();
    let outcome = world
        .archive
        .writer
        .user(operator_request(
            id.clone(),
            UserWrite::Assert {
                assertions: vec![
                    statement(AssertionKind::Note, None, "one".into()),
                    statement(AssertionKind::Note, None, "two".into()),
                    statement(AssertionKind::Note, None, "three".into()),
                ],
            },
        ))
        .unwrap();
    assert_eq!(outcome.id, format!("{id}.1"));
    let written: Vec<String> = world
        .archive_db()
        .prepare("SELECT id FROM assertion WHERE id LIKE ?1 ORDER BY id")
        .unwrap()
        .query_map([format!("{id}.%")], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(
        written,
        vec![format!("{id}.1"), format!("{id}.2"), format!("{id}.3")]
    );
}

#[test]
fn a_relation_also_points_about_the_same_revision() {
    let mut world = World::new();
    let claim = world.note("a claim");
    let challenge = world
        .write(UserWrite::Assert {
            assertions: vec![NewAssertion {
                kind: AssertionKind::Note,
                key: None,
                value: Some("it does not hold".into()),
                valid_until: None,
                links: vec![NewAssertionLink {
                    role: LinkRole::Challenges,
                    reference: r(&format!("assertion:{claim}@{claim}")),
                }],
            }],
        })
        .unwrap()
        .id;
    let roles: Vec<(String, String)> = world
        .archive_db()
        .prepare("SELECT role,ref FROM link WHERE record=?1 ORDER BY role")
        .unwrap()
        .query_map([&challenge], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let target = format!("assertion:{claim}@{claim}");
    assert_eq!(
        roles,
        vec![
            ("about".into(), target.clone()),
            ("challenges".into(), target)
        ]
    );
}

#[test]
fn every_stored_assertion_has_author_channel_machine_and_time() {
    let mut world = World::new();
    world.note("noted");
    let db = world.archive_db();
    for (column, row) in [
        ("author", "NULL,'cli','host',1"),
        ("via", "'operator',NULL,'host',1"),
        ("machine", "'operator','cli',NULL,1"),
        ("created_at", "'operator','cli','host',NULL"),
    ] {
        let refused = db.execute(
            &format!(
                "INSERT INTO assertion(id,root,kind,author,via,machine,created_at,commit_seq) VALUES('x','x','note',{row},1)"
            ),
            [],
        );
        assert!(refused.is_err(), "a null {column} was stored");
    }
}

#[test]
fn reading_current_assertions_about_a_target_uses_indexes_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("archive.db");
    crate::schema::ensure_archive(&Connection::open(&path).unwrap()).unwrap();
    let conn = Connection::open_in_memory().unwrap();
    conn.execute("ATTACH DATABASE ?1 AS a", [path.to_str().unwrap()])
        .unwrap();
    crate::source::views(&conn).unwrap();
    let sql = format!(
        "EXPLAIN QUERY PLAN SELECT n.id FROM source_link l JOIN source_assertion n ON n.id=l.record \
         WHERE l.ref='session:codex:s' AND l.role='about' AND n.kind='note' AND {}",
        crate::assertions::sql::current_assertion("n")
    );
    let plan: Vec<String> = conn
        .prepare(&sql)
        .unwrap()
        .query_map([], |row| row.get::<_, String>(3))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert!(
        plan.iter().all(|step| !step.starts_with("SCAN")),
        "{plan:?}"
    );
}

fn fact(header: &Header) -> AssertionFact {
    let withdrawal = header.kind == authored_generator::WITHDRAWAL;
    let mut links = Vec::new();
    if withdrawal {
        for target in &header.withdraws {
            links.push(AssertionLink {
                role: LinkRole::Withdraws,
                reference: r(&match target {
                    Target::Root(root) => format!("assertion:{root}"),
                    Target::Revision(revision) => {
                        format!("assertion:{}@{}", revision.root, revision.rev)
                    }
                }),
                seen_hash: None,
                quote: None,
            });
        }
    } else {
        links.push(AssertionLink {
            role: LinkRole::About,
            reference: r("session:codex:subject"),
            seen_hash: None,
            quote: None,
        });
        for base in &header.revises {
            links.push(AssertionLink {
                role: LinkRole::Revises,
                reference: r(&format!("assertion:{}@{}", base.root, base.rev)),
                seen_hash: None,
                quote: None,
            });
        }
    }
    AssertionFact {
        id: header.rev.clone(),
        root: header.root.clone(),
        kind: if withdrawal {
            AssertionKind::Withdrawal
        } else {
            AssertionKind::Note
        },
        key: None,
        value: (!withdrawal).then(|| format!("text {}", header.rev).into()),
        author: Author::Operator,
        via: Channel::Cli,
        machine: "synthetic".into(),
        created_at: Timestamp::from_unix_ms(header.at).unwrap(),
        valid_until: None,
        links,
    }
}

fn stored(conn: &Connection) -> Vec<(String, String, bool, bool)> {
    conn.prepare("SELECT root,id,current,withdrawn FROM revision_state ORDER BY root,id")
        .unwrap()
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

fn expected(history: &mut History) -> Vec<(String, String, bool, bool)> {
    history
        .states()
        .unwrap()
        .into_iter()
        .flat_map(|state| {
            state.revisions.into_iter().map(|revision| {
                let Revision { root, rev } = revision.revision;
                (root, rev, revision.current, revision.withdrawn)
            })
        })
        .collect()
}

fn archive() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::ensure_archive(&conn).unwrap();
    conn
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn the_stored_index_equals_the_rule_after_every_write(
        generated in authored_generator::history(24),
    ) {
        let conn = archive();
        let mut prefix = History::new();
        for (counter, header) in generated.records().iter().enumerate() {
            crate::assertions::apply(&conn, &[fact(header)], counter as i64 + 1).unwrap();
            prefix.push(header.clone(), Vec::new());
            prop_assert_eq!(stored(&conn), expected(&mut prefix));
        }
    }

    #[test]
    fn stored_currency_ignores_arrival_order_counters_and_clocks(
        generated in authored_generator::history(24),
        order in proptest::collection::vec(any::<u32>(), 24),
        clocks in proptest::collection::vec(0i64..4_000_000_000_000, 24),
    ) {
        let mut full = generated.clone();
        let mut records: Vec<(u32, Header)> = generated
            .records()
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, mut header)| {
                header.at = clocks[index % clocks.len()];
                (order[index % order.len()], header)
            })
            .collect();
        records.sort_by_key(|(key, header)| (*key, header.rev.clone()));
        let conn = archive();
        for (counter, (_, header)) in records.iter().enumerate() {
            crate::assertions::apply(&conn, &[fact(header)], counter as i64 + 1).unwrap();
        }
        prop_assert_eq!(stored(&conn), expected(&mut full));
    }
}
