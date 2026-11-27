use super::*;

fn cursor() -> PageCursor {
    PageCursor {
        session_read: None,
        alternates: vec![],
        target: "session:test:one".into(),
        digest: options_digest(&("transcript", 4000)),
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

fn seal(key: u8, index_instance: &str) -> CursorSeal {
    CursorSeal {
        key: [key; 32],
        index_instance: index_instance.into(),
    }
}

/// The sealed text with one letter of the target changed inside the payload
/// and the original MAC kept, so the payload still decodes.
fn with_changed_target(sealed: &str) -> String {
    let (payload, mac) = sealed.split_once('.').unwrap();
    let mut payload = unhex(payload).unwrap();
    let at = payload
        .windows(b"session:test:one".len())
        .position(|window| window == b"session:test:one")
        .unwrap();
    payload[at + "session:test:on".len()] = b'f';
    format!("{}.{mac}", hex(&payload))
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
    assert_eq!(
        PageCursor::open(&with_changed_target(&sealed), &key),
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
    assert!(cursor.check("session:test:one", &cursor.digest).is_ok());
    assert_eq!(
        cursor.check("session:test:two", &cursor.digest),
        Err(CursorError::OtherReading("the target differs"))
    );
    assert_eq!(
        cursor.check("session:test:one", "other"),
        Err(CursorError::OtherReading("the reading options differ"))
    );
    // A digest outside the current domain was written by an earlier build,
    // which read its options another way.
    let earlier = PageCursor {
        digest: blake3::hash(b"{}").to_hex().to_string(),
        ..cursor.clone()
    };
    assert_eq!(
        earlier.check("session:test:one", &cursor.digest),
        Err(CursorError::EarlierBuild)
    );
}

#[test]
fn a_change_to_any_composed_version_field_stales() {
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
    versions[0].attachments = Some(crate::BlobHash::of(b"attachment matching"));
    assert!(cursor.rewritten(&PageCursor::version(&versions)));
    let matched = PageCursor::version(&versions);
    versions[0].attachments = Some(crate::BlobHash::of(b"other matching"));
    assert_ne!(matched, PageCursor::version(&versions));
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
fn a_cursor_of_an_earlier_index_instance_is_stale_not_broken() {
    let sealed = cursor().seal(&seal(7, "g1"));
    assert_eq!(
        PageCursor::open(&sealed, &seal(9, "g2")),
        Err(CursorError::Stale)
    );
}

#[test]
fn a_cursor_sealed_with_the_same_key_for_another_index_instance_is_stale() {
    let sealed = cursor().seal(&seal(7, "g1"));
    assert_eq!(
        PageCursor::open(&sealed, &seal(7, "g2")),
        Err(CursorError::Stale)
    );
}

#[test]
fn a_session_lower_bound_round_trips_and_an_ordinary_cursor_stores_none() {
    let key = seal(7, "g1");
    let ordinary = cursor();
    let stored = serde_json::to_value(&ordinary).unwrap();
    let mut names: Vec<&str> = stored
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "alternates",
            "digest",
            "direction",
            "position",
            "target",
            "versions"
        ]
    );
    assert_eq!(
        serde_json::from_value::<PageCursor>(stored).unwrap(),
        ordinary
    );
    let mut bound = ordinary;
    bound.session_read = Some(SessionRead {
        session: crate::SessionId::parse("test:one").unwrap(),
        from_seen: true,
        logical: false,
        parts: vec![],
        after: Some("block:test:one:m1#0".into()),
    });
    assert_eq!(PageCursor::open(&bound.seal(&key), &key).unwrap(), bound);
    // The lower bound is inside the sealed payload: changing it is refused.
    let sealed = bound.seal(&key);
    let (payload, mac) = sealed.split_once('.').unwrap();
    let mut payload = unhex(payload).unwrap();
    let at = payload
        .windows(b"block:test:one:m1#0".len())
        .position(|window| window == b"block:test:one:m1#0")
        .unwrap();
    payload[at + "block:test:one:m".len()] = b'2';
    assert_eq!(
        PageCursor::open(&format!("{}.{mac}", hex(&payload)), &key),
        Err(CursorError::Invalid)
    );
}

#[test]
fn composed_session_bounds_survive_sealing_without_changing_physical_defaults() {
    let physical = SessionRead {
        session: crate::SessionId::parse("test:root").unwrap(),
        from_seen: true,
        after: None,
        logical: false,
        parts: vec![],
    };
    let value = serde_json::to_value(&physical).unwrap();
    assert!(value.get("logical").is_none());
    assert!(value.get("parts").is_none());
    assert_eq!(
        serde_json::from_value::<SessionRead>(value).unwrap(),
        physical
    );
    let mut bound = cursor();
    bound.session_read = Some(SessionRead {
        logical: true,
        parts: vec![
            SessionSeen {
                session: physical.session.clone(),
                after: None,
            },
            SessionSeen {
                session: crate::SessionId::parse("test:continuation").unwrap(),
                after: Some("block:test:continued#0".into()),
            },
        ],
        ..physical
    });
    let key = seal(7, "g1");
    assert_eq!(PageCursor::open(&bound.seal(&key), &key).unwrap(), bound);
}
