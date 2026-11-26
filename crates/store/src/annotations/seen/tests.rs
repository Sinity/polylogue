use super::*;
use crate::fixture::{
    authored::operator_request,
    writer::{WriterArchive, message},
};
use polylogue_model::{
    Author, Role,
    user::{SeenMark, UserWrite},
};

fn request(id: &str, mark: SeenMark) -> polylogue_model::user::UserRequest {
    operator_request(id.into(), UserWrite::MarkSeen { marks: vec![mark] })
}

#[test]
fn seen_mark_written_only_when_it_advances() {
    let mut archive = WriterArchive::new();
    // Native IDs deliberately sort against the actual source reading order.
    let first = message("z-first", "reader", Role::User, 1, vec![]);
    let last = message("a-last", "reader", Role::Assistant, 2, vec![]);
    let first_item = Reference::Message(first.id.clone()).to_string();
    let last_item = Reference::Message(last.id.clone()).to_string();
    archive.write("reader", vec![first, last]);
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let target = Reference::parse("session:claude-code:reader").unwrap();
    let position = pool.reader().unwrap().change_position().unwrap();
    let mark = SeenMark {
        target: target.to_string(),
        item: Some(first_item),
        at: position,
    };
    let first_request = request("seen-first", mark.clone());
    let receipt = archive
        .writer
        .archive()
        .submit(first_request.clone(), None)
        .map(|committed| committed.outcome)
        .unwrap();
    assert!(!receipt.outcome.repeat);
    assert!(
        receipt.written.is_empty(),
        "reading marks supply no undo targets"
    );
    let after_first = pool.reader().unwrap().change_position().unwrap();
    assert!(
        archive
            .writer
            .archive()
            .user(request("seen-same", mark.clone()))
            .unwrap()
            .repeat
    );
    assert_eq!(
        pool.reader().unwrap().change_position().unwrap(),
        after_first
    );
    assert!(
        pool.reader()
            .unwrap()
            .user_request("seen-same")
            .unwrap()
            .is_none()
    );

    let later = SeenMark {
        item: Some(last_item),
        ..mark.clone()
    };
    assert!(
        !archive
            .writer
            .archive()
            .user(request("seen-later", later.clone()))
            .unwrap()
            .repeat
    );
    assert!(
        archive
            .writer
            .archive()
            .user(request("seen-backward", mark.clone()))
            .unwrap()
            .repeat
    );
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .seen_marks("operator", std::slice::from_ref(&target))
            .unwrap(),
        vec![later.clone()]
    );
    assert!(
        reader
            .seen_marks("agent", std::slice::from_ref(&target))
            .unwrap()
            .is_empty()
    );
    assert!(reader.user_request("seen-backward").unwrap().is_none());
    drop(reader);
    drop(pool);
    archive.restart();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    assert!(archive.writer.archive().user(first_request).unwrap().repeat);
    assert_eq!(
        pool.reader()
            .unwrap()
            .seen_marks("operator", std::slice::from_ref(&target))
            .unwrap(),
        vec![later]
    );
    let mut other = request("seen-other-author", mark.clone());
    other.authorship.author = Author::Agent(None);
    assert!(!archive.writer.archive().user(other).unwrap().repeat);
    assert_eq!(
        pool.reader()
            .unwrap()
            .seen_marks("agent", &[target])
            .unwrap(),
        vec![mark]
    );
}

#[test]
fn seen_mark_on_a_session_refuses_an_item_that_is_not_a_message_or_block() {
    let mut archive = WriterArchive::new();
    archive.write(
        "reader",
        vec![message("only", "reader", Role::User, 1, vec![])],
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let target = Reference::parse("session:claude-code:reader").unwrap();
    let position = pool.reader().unwrap().change_position().unwrap();
    let mark = SeenMark {
        target: target.to_string(),
        item: Some(target.to_string()),
        at: position,
    };
    assert!(matches!(
        archive
            .writer
            .archive()
            .user(request("seen-session-item", mark)),
        Err(StoreError::Refused(
            polylogue_model::Reason::SeenMarkNeedsItem {}
        ))
    ));
    assert!(
        pool.reader()
            .unwrap()
            .seen_marks("operator", &[target])
            .unwrap()
            .is_empty()
    );
}

#[test]
fn seen_marks_can_move_to_the_current_generation_without_changing_archive_position() {
    let archive = WriterArchive::new();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let target = Reference::parse("session:claude-code:reader").unwrap();
    let old = SeenMark {
        target: target.to_string(),
        item: None,
        at: pool.reader().unwrap().change_position().unwrap(),
    };
    archive
        .writer
        .archive()
        .user(request("seen-before", old.clone()))
        .unwrap();
    let before_reset = pool.reader().unwrap().change_position().unwrap();
    drop(pool);
    archive.archive.advance_generation_for_test().unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let new_position = pool.reader().unwrap().change_position().unwrap();
    assert_ne!(new_position.generation, before_reset.generation);
    assert_eq!(new_position.archive, before_reset.archive);
    let new = SeenMark {
        at: new_position,
        ..old.clone()
    };
    archive
        .writer
        .archive()
        .user(request("seen-after", new.clone()))
        .unwrap();
    assert_eq!(
        pool.reader()
            .unwrap()
            .seen_marks("operator", &[target])
            .unwrap(),
        vec![new]
    );
    assert!(matches!(
        archive.writer.archive().user(request("seen-stale", old)),
        Err(StoreError::Refused(
            polylogue_model::Reason::SeenFromAnotherGeneration {}
        ))
    ));
    assert!(
        pool.reader()
            .unwrap()
            .user_request("seen-stale")
            .unwrap()
            .is_none()
    );
}

fn relation(
    parent: &str,
    child: &str,
    branch: Option<&polylogue_model::MessageId>,
) -> polylogue_model::Link {
    polylogue_model::Link {
        kind: if branch.is_some() {
            polylogue_model::LinkKind::Fork
        } else {
            polylogue_model::LinkKind::Resume
        },
        copies_prefix: branch.is_none(),
        how: None,
        parent: crate::fixture::writer::sid(parent),
        child: crate::fixture::writer::sid(child),
        branch_point: branch.cloned(),
        call_id: None,
        evidence: "synthetic recorded relation".into(),
    }
}

#[test]
fn seen_session_items_follow_the_readers_branch_prefix_before_own_messages() {
    let mut archive = WriterArchive::new();
    let branch = message("z-branch", "parent", Role::User, 200, vec![]);
    let excluded = message("a-tail", "parent", Role::User, 0, vec![]);
    archive.write("parent", vec![branch.clone(), excluded.clone()]);
    let own = message("a-own", "child", Role::User, -1, vec![]);
    archive
        .links
        .push(relation("parent", "child", Some(&branch.id)));
    archive.write("child", vec![own.clone()]);
    let unrelated = message("other", "unrelated", Role::User, 999, vec![]);
    archive.write("unrelated", vec![unrelated.clone()]);
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let target = Reference::Session(crate::fixture::writer::sid("child"));
    let position = pool.reader().unwrap().change_position().unwrap();
    let mark = |message: &polylogue_model::Message| SeenMark {
        target: target.to_string(),
        item: Some(Reference::Message(message.id.clone()).to_string()),
        at: position.clone(),
    };
    {
        let reader = pool.reader().unwrap();
        let selection = reader
            .select_messages(&crate::fixture::writer::sid("child"), true)
            .unwrap();
        selection.require_complete().unwrap();
        assert_eq!(
            selection.locate(Some(&branch.id)).unwrap().unwrap().message,
            0
        );
        assert_eq!(selection.locate(Some(&own.id)).unwrap().unwrap().message, 1);
        assert!(selection.locate(Some(&excluded.id)).unwrap().is_none());
        assert!(selection.locate(Some(&unrelated.id)).unwrap().is_none());
    }
    assert!(
        !archive
            .writer
            .archive()
            .user(request("seen-inherited", mark(&branch)))
            .unwrap()
            .repeat
    );
    assert!(
        !archive
            .writer
            .archive()
            .user(request("seen-child", mark(&own)))
            .unwrap()
            .repeat
    );
    assert!(
        archive
            .writer
            .archive()
            .user(request("seen-reverse", mark(&branch)))
            .unwrap()
            .repeat
    );
    for (id, item) in [
        ("seen-excluded-tail", &excluded),
        ("seen-unrelated", &unrelated),
    ] {
        assert!(matches!(
            archive.writer.archive().user(request(id, mark(item))),
            Err(StoreError::Refused(polylogue_model::Reason::NotInSession { item: ref named, ref session }))
                if *named == format!("message:{}", item.id)
                    && *session == format!("session:{}", crate::fixture::writer::sid("child"))
        ));
        assert!(pool.reader().unwrap().user_request(id).unwrap().is_none());
    }
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .seen_marks("operator", std::slice::from_ref(&target))
            .unwrap(),
        vec![mark(&own)]
    );
    assert!(reader.user_request("seen-reverse").unwrap().is_none());
}

#[test]
fn seen_session_items_use_recorded_copy_membership_and_share_readers_ambiguity() {
    let mut archive = WriterArchive::new();
    let copied = message("z-copied", "parent", Role::User, 100, vec![]);
    let excluded = message("a-not-copied", "parent", Role::User, -100, vec![]);
    archive.write("parent", vec![copied.clone(), excluded.clone()]);
    let own = message("a-own", "child", Role::User, -200, vec![]);
    archive.links.push(relation("parent", "child", None));
    archive.write_from(
        "copy-child",
        polylogue_model::Session::new(
            crate::fixture::writer::sid("child"),
            polylogue_model::SessionKind::Main,
        ),
        vec![copied.clone(), own.clone()],
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let target = Reference::Session(crate::fixture::writer::sid("child"));
    let position = pool.reader().unwrap().change_position().unwrap();
    let mark = |message: &polylogue_model::Message| SeenMark {
        target: target.to_string(),
        item: Some(Reference::Message(message.id.clone()).to_string()),
        at: position.clone(),
    };
    {
        let reader = pool.reader().unwrap();
        let selection = reader
            .select_messages(&crate::fixture::writer::sid("child"), true)
            .unwrap();
        selection.require_complete().unwrap();
        assert_eq!(
            selection.locate(Some(&copied.id)).unwrap().unwrap().message,
            0
        );
        assert_eq!(selection.locate(Some(&own.id)).unwrap().unwrap().message, 1);
        assert!(selection.locate(Some(&excluded.id)).unwrap().is_none());
    }
    assert!(
        !archive
            .writer
            .archive()
            .user(request("seen-copy", mark(&copied)))
            .unwrap()
            .repeat
    );
    assert!(
        !archive
            .writer
            .archive()
            .user(request("seen-copy-own", mark(&own)))
            .unwrap()
            .repeat
    );
    assert!(
        archive
            .writer
            .archive()
            .user(request("seen-copy-reverse", mark(&copied)))
            .unwrap()
            .repeat
    );
    assert!(matches!(
        archive.writer.archive().user(request("seen-copy-excluded", mark(&excluded))),
        Err(StoreError::Refused(polylogue_model::Reason::NotInSession { item: ref named, ref session }))
            if *named == format!("message:{}", excluded.id)
                && *session == format!("session:{}", crate::fixture::writer::sid("child"))
    ));
    drop(pool);

    // Two original copied readings disagree. The reader exposes neither
    // invented order, and neither can become a seen checkpoint.
    let conflict_own = message("own", "conflict", Role::User, 0, vec![]);
    archive.links.push(relation("parent", "conflict", None));
    archive.write_from(
        "copy-left",
        polylogue_model::Session::new(
            crate::fixture::writer::sid("conflict"),
            polylogue_model::SessionKind::Main,
        ),
        vec![copied.clone(), excluded.clone(), conflict_own.clone()],
    );
    archive.links.push(relation("parent", "conflict", None));
    archive.write_from(
        "copy-right",
        polylogue_model::Session::new(
            crate::fixture::writer::sid("conflict"),
            polylogue_model::SessionKind::Main,
        ),
        vec![excluded, copied.clone(), conflict_own],
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let selection = reader
        .select_messages(&crate::fixture::writer::sid("conflict"), true)
        .unwrap();
    assert!(matches!(
        selection.parents().issues.as_slice(),
        [crate::read::ParentIssue::ConflictingOrder { .. }]
    ));
    assert!(selection.locate(Some(&copied.id)).unwrap().is_none());
    let mark = SeenMark {
        target: "session:claude-code:conflict".into(),
        item: Some(Reference::Message(copied.id).to_string()),
        at: reader.change_position().unwrap(),
    };
    drop(selection);
    drop(reader);
    assert!(matches!(
        archive
            .writer
            .archive()
            .user(request("seen-copy-conflict", mark)),
        Err(StoreError::IncompleteParents(_))
    ));
    assert!(
        pool.reader()
            .unwrap()
            .user_request("seen-copy-conflict")
            .unwrap()
            .is_none()
    );
}
