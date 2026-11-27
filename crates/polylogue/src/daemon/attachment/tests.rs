use super::*;
use crate::test_daemon;
use polylogue_model::{BlobHash, MessageId};
use std::io::{Read, Seek, SeekFrom, Write};

fn inline(daemon: &test_daemon::TestDaemon, bytes: &[u8]) -> MessageId {
    let record = serde_json::json!({"type":"user","uuid":"inline-message","sessionId":"inline-session","timestamp":"2026-03-01T10:00:00Z","message":{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":base64::engine::general_purpose::STANDARD.encode(bytes)}}]}});
    std::fs::write(
        daemon.home.join(".claude/projects/-p/inline.jsonl"),
        format!("{record}\n"),
    )
    .unwrap();
    daemon.wait_for(|status| status["store"]["messages"] == 2);
    MessageId::parse("claude-code:inline-message").unwrap()
}
fn request(
    message: &MessageId,
    input: Option<String>,
    cursor: Option<String>,
    length: usize,
) -> Range {
    Range {
        message: message.clone(),
        block: 0,
        attachment: 0,
        input,
        length,
        cursor,
    }
}
fn page(response: command_surface::Answer<BytePage>) -> BytePage {
    crate::test_answers::ok(response)
}
fn seal(daemon: &test_daemon::TestDaemon) -> CursorSeal {
    let reader = daemon.daemon.reads.reader().unwrap();
    CursorSeal {
        key: reader.cursor_key().unwrap(),
        index_instance: reader.change_position().unwrap().index_instance,
    }
}
fn refused_with(response: command_surface::Answer<BytePage>, code: &str) {
    assert!(
        matches!(response, command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some(code)),
        "{response:?}"
    );
}

#[test]
fn selected_inline_cas_pages_keep_bytes_without_inventing_an_input() {
    for bytes in [Vec::new(), vec![b'Q'; PAGE_BYTES + 17]] {
        let daemon = test_daemon::start("synthetic inline attachment");
        let message = inline(&daemon, &bytes);
        let mut cursor = None;
        let mut actual = Vec::new();
        loop {
            let first = cursor.is_none();
            let page = page(read(
                request(&message, None, cursor, usize::MAX),
                &daemon.daemon,
            ));
            assert!(page.input.is_none());
            assert_eq!(page.blob, BlobHash::of(&bytes));
            assert_eq!(page.size, bytes.len() as u64);
            assert_eq!(page.offset, actual.len() as u64);
            let part = base64::engine::general_purpose::STANDARD
                .decode(page.content_base64)
                .unwrap();
            assert!(part.len() <= PAGE_BYTES);
            actual.extend(part);
            if first {
                let descriptor = page.descriptor.unwrap();
                assert_eq!(
                    descriptor.state,
                    polylogue_store::assets::AttachmentState::Available
                );
                assert!(descriptor.matches.is_empty());
                assert_eq!(descriptor.source.blob, Some(BlobHash::of(&bytes)));
            } else {
                assert!(page.descriptor.is_none());
            }
            cursor = page.next;
            if cursor.is_none() {
                break;
            }
        }
        assert_eq!(actual, bytes);
    }
}

#[test]
fn explicit_inputs_and_inline_cursor_sources_never_fall_back_to_other_bytes() {
    let daemon = test_daemon::start("synthetic inline selectors");
    let message = inline(&daemon, &vec![b'R'; PAGE_BYTES + 1]);
    let refused = read(
        request(&message, Some("unrelated-receiving-input".into()), None, 1),
        &daemon.daemon,
    );
    assert!(
        matches!(refused,command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("attachment_unavailable")),
        "{refused:?}"
    );
    let first = page(read(request(&message, None, None, 1), &daemon.daemon));
    let cursor = first.next.unwrap();
    let refused = read(
        request(
            &message,
            Some("unrelated-receiving-input".into()),
            Some(cursor.clone()),
            1,
        ),
        &daemon.daemon,
    );
    assert!(
        matches!(refused,command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("bad_request")),
        "{refused:?}"
    );
    let seal = seal(&daemon);
    let mut changed: Cursor = seal.open(SEAL_DOMAIN, &cursor).unwrap();
    changed.source = Some(CursorSource::Inline {
        message: message.clone(),
        block: 1,
        attachment: 0,
    });
    let changed = seal.seal(SEAL_DOMAIN, &changed);
    let refused = read(request(&message, None, Some(changed), 1), &daemon.daemon);
    assert!(
        matches!(refused,command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("stale_cursor")),
        "{refused:?}"
    );
}

#[test]
fn receiving_member_cursors_keep_their_input_selection_and_original_wire_shape() {
    let daemon = test_daemon::start("synthetic receiving attachment");
    let expected = b"receiving bytes";
    let message = MessageId::parse(test_daemon::assets::seed_bytes(&daemon, expected)).unwrap();
    let mut cursor = None;
    let mut selected = None;
    let mut actual = Vec::new();
    loop {
        let page = page(read(request(&message, None, cursor, 3), &daemon.daemon));
        let input = page
            .input
            .clone()
            .expect("a receiving member keeps its input");
        if let Some(selected) = &selected {
            assert_eq!(&input, selected);
        } else {
            selected = Some(input.clone());
        }
        actual.extend(
            base64::engine::general_purpose::STANDARD
                .decode(page.content_base64)
                .unwrap(),
        );
        cursor = page.next;
        if let Some(cursor) = &cursor {
            let wire: serde_json::Value = seal(&daemon).open(SEAL_DOMAIN, cursor).unwrap();
            assert_eq!(wire["input"], input);
            assert!(
                wire.get("source").is_none(),
                "old receiving cursors need no inline type field"
            );
            if actual.len() == 3 {
                let refused = read(
                    request(
                        &message,
                        Some("unrelated-receiving-input".into()),
                        Some(cursor.clone()),
                        3,
                    ),
                    &daemon.daemon,
                );
                assert!(
                    matches!(refused,command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("attachment_unavailable")),
                    "{refused:?}"
                );
            }
        } else {
            break;
        }
    }
    assert_eq!(actual, expected);
}

#[test]
fn a_forged_attachment_cursor_is_refused() {
    let daemon = test_daemon::start("synthetic forged attachment cursor");
    let bytes = vec![b'F'; PAGE_BYTES + 5];
    let message = inline(&daemon, &bytes);
    let first = page(read(request(&message, None, None, 2), &daemon.daemon));
    let issued = first.next.unwrap();
    let seal = seal(&daemon);
    let position: Cursor = seal.open(SEAL_DOMAIN, &issued).unwrap();
    assert_eq!(position.offset, 2);
    let moved = Cursor {
        offset: 1,
        ..position
    };
    // The same position under another key, under the page cursors' domain,
    // as plain JSON, and with one payload byte changed.
    let other_key = CursorSeal {
        key: [9; 32],
        index_instance: seal.index_instance.clone(),
    };
    let (payload, mac) = issued.split_once('.').unwrap();
    let mut changed = payload.as_bytes().to_vec();
    changed[2] = if changed[2] == b'0' { b'1' } else { b'0' };
    for forged in [
        other_key.seal(SEAL_DOMAIN, &moved),
        seal.seal(b"polylogue.page-cursor\0", &moved),
        base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&moved).unwrap()),
        format!("{}.{mac}", String::from_utf8(changed.clone()).unwrap()),
    ] {
        refused_with(
            read(request(&message, None, Some(forged), 2), &daemon.daemon),
            "bad_request",
        );
    }
    // The issued cursor still continues where it said.
    let next = page(read(
        request(&message, None, Some(issued), 2),
        &daemon.daemon,
    ));
    assert_eq!(next.offset, 2);
}

#[test]
fn missing_or_corrupt_inline_frames_refuse_even_a_zero_byte_page() {
    for missing in [false, true] {
        let daemon = test_daemon::start("synthetic inline empty body");
        let message = inline(&daemon, b"");
        let (root, extent) = {
            let reader = daemon.daemon.reads.reader().unwrap();
            let blobs = reader.blob_reader().unwrap();
            (
                blobs.root().to_path_buf(),
                blobs
                    .extent(&hashed_blobs::Hash(BlobHash::of(b"").0))
                    .unwrap(),
            )
        };
        assert!(extent.len > 0);
        let path = root.join(&extent.segment);
        let held = path.with_extension("held-inline-empty");
        let mut original = [0];
        let offset = extent.offset + extent.len - 1;
        if missing {
            std::fs::rename(&path, &held).unwrap();
        } else {
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.read_exact(&mut original).unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&[original[0] ^ 1]).unwrap();
            file.sync_all().unwrap();
        }
        let refused = read(request(&message, None, None, 1), &daemon.daemon);
        if missing {
            std::fs::rename(&held, &path).unwrap();
        } else {
            let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&original).unwrap();
            file.sync_all().unwrap();
        }
        assert!(
            matches!(
                refused,
                command_surface::Answer {
                    outcome: command_surface::Outcome::Error(_),
                    ..
                }
            ),
            "{refused:?}"
        );
        assert_eq!(
            page(read(request(&message, None, None, 1), &daemon.daemon)).size,
            0
        );
    }
}
