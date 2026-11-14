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

#[test]
fn selected_inline_cas_pages_keep_exact_bytes_without_inventing_an_input() {
    for bytes in [Vec::new(), vec![b'Q'; PAGE_BYTES + 17]] {
        let daemon = test_daemon::start("synthetic canonical inline attachment");
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
    let mut changed: Cursor = serde_json::from_slice(
        &base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(cursor)
            .unwrap(),
    )
    .unwrap();
    changed.source = Some(CursorSource::Inline {
        message: message.clone(),
        block: 1,
        attachment: 0,
    });
    let changed = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&changed).unwrap());
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
            .expect("a receiving member retains its actual input");
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
            let wire: serde_json::Value = serde_json::from_slice(
                &base64::engine::general_purpose::URL_SAFE_NO_PAD
                    .decode(cursor)
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(wire["input"], input);
            assert!(
                wire.get("source").is_none(),
                "old receiving cursors need no inline discriminator"
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
fn missing_or_corrupt_inline_frames_refuse_even_a_zero_byte_page() {
    for missing in [false, true] {
        let daemon = test_daemon::start("synthetic physical inline empty body");
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
