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
    let receipt = archive.writer.user_receipt(first_request.clone()).unwrap();
    assert!(!receipt.outcome.repeat);
    assert!(
        receipt.written.is_empty(),
        "reading marks supply no undo targets"
    );
    let after_first = pool.reader().unwrap().change_position().unwrap();
    assert!(
        archive
            .writer
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
            .user(request("seen-later", later.clone()))
            .unwrap()
            .repeat
    );
    assert!(
        archive
            .writer
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
    assert!(archive.writer.user(first_request).unwrap().repeat);
    assert_eq!(
        pool.reader()
            .unwrap()
            .seen_marks("operator", std::slice::from_ref(&target))
            .unwrap(),
        vec![later]
    );
    let mut other = request("seen-other-author", mark.clone());
    other.authorship.author = Author::Agent(None);
    assert!(!archive.writer.user(other).unwrap().repeat);
    assert_eq!(
        pool.reader()
            .unwrap()
            .seen_marks("agent", &[target])
            .unwrap(),
        vec![mark]
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
        .user(request("seen-after", new.clone()))
        .unwrap();
    assert_eq!(
        pool.reader()
            .unwrap()
            .seen_marks("operator", &[target])
            .unwrap(),
        vec![new]
    );
    assert!(archive.writer.user(request("seen-stale", old)).is_err());
    assert!(
        pool.reader()
            .unwrap()
            .user_request("seen-stale")
            .unwrap()
            .is_none()
    );
}
