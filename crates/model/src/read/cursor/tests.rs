use super::*;

fn cursor() -> PageCursor {
    PageCursor {
        target: "session:test:one".into(),
        options: options_digest(&("transcript", 4000)),
        versions: PageCursor::version(&[SessionVersion {
            session: crate::SessionId::parse("test:one").unwrap(),
            rewrite_version: 3,
            attachments: None,
        }]),
        position: PagePosition::Entry {
            key: EntryKey::Message {
                message: crate::MessageId::parse("test:one:m2").unwrap(),
                block: Some(1),
            },
            inside: None,
        },
        direction: ReadDirection::After,
    }
}

fn seal(key: u8, generation: &str) -> CursorSeal {
    CursorSeal {
        key: [key; 32],
        generation: generation.into(),
    }
}

#[test]
fn sealed_cursor_opens_to_the_same_value() {
    let key = seal(7, "g1");
    let sealed = cursor().seal(&key);
    assert_eq!(PageCursor::open(&sealed, &key).unwrap(), cursor());
}

#[test]
fn a_changed_byte_or_another_key_is_refused() {
    let key = seal(7, "g1");
    let sealed = cursor().seal(&key);
    assert_eq!(
        PageCursor::open(&sealed, &seal(8, "g1")),
        Err(CursorError::Invalid)
    );
    let mut forged = sealed.clone().into_bytes();
    forged[4] = if forged[4] == b'0' { b'1' } else { b'0' };
    assert_eq!(
        PageCursor::open(&String::from_utf8(forged).unwrap(), &key),
        Err(CursorError::Invalid)
    );
    assert_eq!(
        PageCursor::open("not a cursor", &key),
        Err(CursorError::Invalid)
    );
}

#[test]
fn a_cursor_names_its_own_reading() {
    let cursor = cursor();
    assert!(cursor.check("session:test:one", &cursor.options).is_ok());
    assert_eq!(
        cursor.check("session:test:two", &cursor.options),
        Err(CursorError::OtherReading("the target differs"))
    );
    assert_eq!(
        cursor.check("session:test:one", "other"),
        Err(CursorError::OtherReading("the reading options differ"))
    );
}

/// law: pl.read.page-rewrite-stale
#[test]
fn only_a_moved_rewrite_version_stales() {
    let cursor = cursor();
    assert!(!cursor.rewritten(&cursor.versions));
    let mut versions = vec![SessionVersion {
        session: crate::SessionId::parse("test:one").unwrap(),
        rewrite_version: 4,
        attachments: None,
    }];
    assert!(cursor.rewritten(&PageCursor::version(&versions)));
    versions[0].rewrite_version = 3;
    assert!(!cursor.rewritten(&PageCursor::version(&versions)));
    versions.push(SessionVersion {
        session: crate::SessionId::parse("test:parent").unwrap(),
        rewrite_version: 1,
        attachments: None,
    });
    assert!(cursor.rewritten(&PageCursor::version(&versions)));
}

#[test]
fn entry_keys_name_their_reference() {
    let message = crate::MessageId::parse("test:one:m2").unwrap();
    assert_eq!(
        EntryKey::header(&message).reference(),
        "message:test:one:m2"
    );
    assert_eq!(
        EntryKey::Message {
            message,
            block: Some(3)
        }
        .reference(),
        "block:test:one:m2#3"
    );
    assert_eq!(
        EntryKey::Event { event: "e1".into() }.reference(),
        "event:e1"
    );
}

#[test]
fn a_cursor_of_an_earlier_generation_is_stale_not_broken() {
    let sealed = cursor().seal(&seal(7, "g1"));
    assert_eq!(
        PageCursor::open(&sealed, &seal(9, "g2")),
        Err(CursorError::Stale)
    );
}

#[test]
fn a_cursor_sealed_with_the_same_key_for_another_generation_is_stale() {
    let sealed = cursor().seal(&seal(7, "g1"));
    assert_eq!(
        PageCursor::open(&sealed, &seal(7, "g2")),
        Err(CursorError::Stale)
    );
}
