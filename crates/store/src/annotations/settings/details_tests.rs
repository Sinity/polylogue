use super::*;
use crate::fixture::writer::WriterArchive;

fn revision(
    id: &str,
    root: &str,
    key: &str,
    machine: Option<&str>,
    value: settings::Value,
    at: Option<i64>,
    bases: &[(&str, &str)],
) -> AnnotationFact {
    AnnotationFact {
        id: id.into(),
        root: root.into(),
        kind: AnnotationKind::Setting,
        key: Some(key.into()),
        value: Some(
            serde_json::to_value(values::Setting {
                machine: machine.map(str::to_owned),
                value,
            })
            .unwrap(),
        ),
        author: Author::Unattributed,
        via: Channel::Import,
        machine: "writing-host".into(),
        created_at: at.map(|at| Timestamp::from_unix_ms(at).unwrap()),
        valid_until: None,
        links: bases
            .iter()
            .map(|(root, id)| AnnotationLink {
                role: LinkRole::Revises,
                reference: Reference::Annotation(RevisionReference::exact(
                    (*root).into(),
                    (*id).into(),
                )),
                seen_hash: None,
                quote: None,
            })
            .collect(),
    }
}

fn ids(details: &settings::Details) -> Vec<&str> {
    details
        .history
        .iter()
        .map(|entry| {
            let Reference::Annotation(reference) = &entry.authored.reference else {
                panic!("setting reference");
            };
            reference.revision.as_deref().unwrap()
        })
        .collect()
}

#[test]
fn setting_details_follow_declared_choices_and_keep_unknown_time_and_withdrawn_revisions() {
    let archive = WriterArchive::new();
    let boolean = settings::Value::Boolean;
    for fact in [
        revision(
            "z-old",
            "setting",
            "browser.actions",
            None,
            boolean(false),
            Some(100),
            &[],
        ),
        revision(
            "a-recent",
            "setting",
            "browser.actions",
            None,
            boolean(true),
            Some(200),
            &[("setting", "z-old")],
        ),
        revision(
            "b-tie",
            "setting",
            "browser.actions",
            None,
            boolean(false),
            Some(200),
            &[("setting", "a-recent")],
        ),
        revision(
            "unknown",
            "setting",
            "browser.actions",
            None,
            boolean(true),
            None,
            &[("setting", "b-tie")],
        ),
        revision(
            "unrelated",
            "other",
            "embed.calls",
            None,
            boolean(true),
            Some(300),
            &[],
        ),
    ] {
        archive.writer.restore_annotation(fact, None).unwrap();
    }
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let details = reader
        .setting_details("browser.actions", "reading-host")
        .unwrap();
    assert_eq!(details.allowed_values, ["on", "off"]);
    assert_eq!(details.machine, None);
    assert_eq!(
        details.current.value,
        Some(boolean(true)),
        "an undated descendant still owns the value"
    );
    assert_eq!(ids(&details), ["b-tie", "a-recent", "z-old", "unknown"]);
    let undated = details.history.last().unwrap();
    assert_eq!(undated.authored.at, None);
    assert_eq!(undated.authored.author, Author::Unattributed);
    assert_eq!(undated.authored.via, Channel::Import);
    assert!(undated.current && !undated.withdrawn);
    drop(reader);
    archive
        .writer
        .restore_annotation(
            AnnotationFact {
                id: "withdraw".into(),
                root: "withdraw".into(),
                kind: AnnotationKind::Withdrawal,
                key: None,
                value: None,
                author: Author::Operator,
                via: Channel::Cli,
                machine: "writing-host".into(),
                created_at: Timestamp::from_unix_ms(400),
                valid_until: None,
                links: vec![AnnotationLink {
                    role: LinkRole::Withdraws,
                    reference: Reference::Annotation(RevisionReference::exact(
                        "setting".into(),
                        "unknown".into(),
                    )),
                    seen_hash: None,
                    quote: None,
                }],
            },
            None,
        )
        .unwrap();
    let details = pool
        .reader()
        .unwrap()
        .setting_details("browser.actions", "reading-host")
        .unwrap();
    assert_eq!(ids(&details), ["b-tie", "a-recent", "z-old", "unknown"]);
    assert_eq!(details.current.value, Some(boolean(false)));
    let withdrawn = details.history.last().unwrap();
    assert!(!withdrawn.current && withdrawn.withdrawn);
}

#[test]
fn setting_details_select_the_target_host_not_the_recording_host() {
    let archive = WriterArchive::new();
    for (id, target, value, writer) in [
        ("remote", "host-b", 7, "writing-host"),
        ("other", "host-a", 3, "host-b"),
    ] {
        let mut fact = revision(
            id,
            id,
            "signal.interrupts_per_hour",
            Some(target),
            settings::Value::InterruptBudget(value),
            Some(100),
            &[],
        );
        fact.machine = writer.into();
        archive.writer.restore_annotation(fact, None).unwrap();
    }
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let details = reader
        .setting_details("signal.interrupts_per_hour", "host-b")
        .unwrap();
    assert_eq!(details.machine.as_deref(), Some("host-b"));
    assert_eq!(details.current.scope, settings::Scope::Host);
    assert_eq!(
        details.current.value,
        Some(settings::Value::InterruptBudget(7))
    );
    assert_eq!(ids(&details), ["remote"]);
    assert_eq!(details.history[0].authored.machine, "writing-host");
    assert!(details.allowed_values.is_empty());
    let untouched = reader
        .setting_details("signal.interrupts_per_hour", "host-c")
        .unwrap();
    assert_eq!(
        untouched.current.default,
        Some(settings::Value::InterruptBudget(4))
    );
    assert_eq!(untouched.current.value, untouched.current.default);
    assert!(untouched.history.is_empty());
}

#[test]
fn setting_history_order_never_picks_a_clock_winner_for_a_fork() {
    let archive = WriterArchive::new();
    for fact in [
        revision(
            "parent",
            "setting",
            "browser.actions",
            None,
            settings::Value::Boolean(true),
            Some(300),
            &[],
        ),
        revision(
            "left",
            "setting",
            "browser.actions",
            None,
            settings::Value::Boolean(false),
            Some(100),
            &[("setting", "parent")],
        ),
        revision(
            "right",
            "setting",
            "browser.actions",
            None,
            settings::Value::Boolean(true),
            Some(500),
            &[("setting", "parent")],
        ),
    ] {
        archive.writer.restore_annotation(fact, None).unwrap();
    }
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let details = pool
        .reader()
        .unwrap()
        .setting_details("browser.actions", "reading-host")
        .unwrap();
    assert_eq!(ids(&details), ["right", "parent", "left"]);
    assert_eq!(details.current.value, None);
    assert_eq!(
        details.current.conflicts,
        [
            Reference::Annotation(RevisionReference::exact("setting".into(), "left".into())),
            Reference::Annotation(RevisionReference::exact("setting".into(), "right".into())),
        ]
    );
    assert!(details.history[0].current && details.history[2].current);
    assert!(!details.history[1].current);
}
