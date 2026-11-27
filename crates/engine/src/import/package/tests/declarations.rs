//! Import recognition from the origin declarations: every origin's golden
//! input (`examples/sources/<origin>/<case>/`), and the cases where the
//! engine's earlier rules and the parsers' rules disagreed.
use super::*;
use polylogue_sources::origin::{Alone, Package};

fn examples() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/sources")
}

/// Stages `files` (relative paths and bytes) under a fresh directory and
/// imports `entry` among them; each member's origin by name.
fn routes(
    files: &[(PathBuf, Vec<u8>)],
    entry: &Path,
    named: Option<&str>,
) -> std::io::Result<Vec<(String, Option<&'static str>)>> {
    let scratch = tempfile::tempdir().unwrap();
    for (path, bytes) in files {
        let at = scratch.path().join(path);
        std::fs::create_dir_all(at.parent().unwrap()).unwrap();
        std::fs::write(at, bytes).unwrap();
    }
    let archive = Archive::at(&scratch.path().join("encoded"));
    let members = stage(&scratch.path().join(entry), &archive, named)?;
    Ok(members
        .iter()
        .filter(|member| !member.container)
        .map(|member| {
            let path = member
                .path
                .strip_prefix(scratch.path())
                .unwrap_or(&member.path);
            (path.display().to_string(), member.origin.map(|o| o.name))
        })
        .collect())
}

/// Each golden input is recognized as its origin by the origin's one rule:
/// on its own when its rule is a structure, when its path keeps one of its
/// locations, only when named for a named origin, and inside its export for
/// an origin that arrives only in one. A tool's file outside its layout is
/// refused.
#[test]
fn every_origin_s_golden_input_is_recognized_as_that_origin() {
    let mut checked = 0;
    for origin in ORIGINS {
        let Ok(cases) = std::fs::read_dir(examples().join(origin.name)) else {
            continue;
        };
        for case in cases {
            let case = case.unwrap().path();
            let recorded = std::fs::read_to_string(case.join("path")).unwrap();
            let recorded = Path::new(recorded.trim().trim_start_matches('/'));
            let input = std::fs::read_dir(&case)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .find(|path| path.file_stem().is_some_and(|stem| stem == "input"))
                .unwrap();
            let bytes = std::fs::read(&input).unwrap();
            let file = |path: &Path| vec![(path.to_path_buf(), bytes.clone())];
            let (files, entry, named, expected) = match &origin.import.alone {
                _ if origin.name == polylogue_sources::polylogue_bundle::ORIGIN => {
                    // The bundle golden starts with the current manifest.
                    (
                        file(recorded),
                        recorded.to_path_buf(),
                        None,
                        Some(origin.name),
                    )
                }
                Alone::Structure(_) => (
                    file(recorded),
                    recorded.to_path_buf(),
                    None,
                    Some(origin.name),
                ),
                Alone::Named => (
                    file(recorded),
                    recorded.to_path_buf(),
                    Some(origin.name),
                    Some(origin.name),
                ),
                Alone::Layout => {
                    let at_root = Path::new(origin.root.default).join(recorded);
                    let kept = [recorded.to_path_buf(), at_root]
                        .into_iter()
                        .find(|path| origin.locations.iter().any(|p| collect::ends_in(path, p)));
                    match kept {
                        Some(path) => (file(&path), path, None, Some(origin.name)),
                        None => (file(recorded), recorded.to_path_buf(), None, None),
                    }
                }
                Alone::No => match origin.import.member.as_ref().map(|m| m.package) {
                    Some(Package::Takeout) => {
                        let path = Path::new(Package::TAKEOUT).join(recorded);
                        (
                            file(&path),
                            PathBuf::from(Package::TAKEOUT),
                            None,
                            Some(origin.name),
                        )
                    }
                    Some(Package::AiderPair) => {
                        let dir = recorded.parent().unwrap();
                        let companion = dir.join(polylogue_sources::aider::COMPANION);
                        let mut files = file(recorded);
                        files.push((companion, b"# aider chat started\n".to_vec()));
                        (files, dir.to_path_buf(), None, Some(origin.name))
                    }
                    _ => (file(recorded), recorded.to_path_buf(), None, None),
                },
            };
            let routed = routes(&files, &entry, named).unwrap();
            let got = routed
                .iter()
                .find(|(path, _)| Path::new(path) == files[0].0)
                .map(|(_, origin)| *origin);
            assert_eq!(
                got,
                Some(expected),
                "{} case {}: {routed:?}",
                origin.name,
                case.display()
            );
            checked += 1;
        }
    }
    assert!(checked > 60, "{checked} golden cases");
}

/// One ChatGPT conversation alone, in the conversation API's shape, is
/// recognized whatever the file is named, also when a long value comes
/// before its identifying keys; the parser's own rule knew only the
/// export's conversation array.
#[test]
fn a_single_chatgpt_conversation_is_recognized_by_its_shape() {
    let body = format!(
        r#"{{"title":"{}","mapping":{{}},"current_node":"root","conversation_id":"one"}}"#,
        "long title ".repeat(20)
    );
    let path = PathBuf::from("saved/one-chat.json");
    assert_eq!(
        routes(&[(path.clone(), body.into_bytes())], &path, None).unwrap(),
        [("saved/one-chat.json".to_owned(), Some("chatgpt"))]
    );
}

/// The Grok account export is its own file name holding a `conversations`
/// array, whatever comes first in it; the same document under another name
/// is refused (docs/capture.md, Import). The parser's own rule accepted any
/// name whose first key was `conversations`.
#[test]
fn a_grok_export_is_its_file_name_and_its_conversations_array() {
    let body = format!(
        r#"{{"account_metadata":"{}","conversations":[]}}"#,
        "synthetic ".repeat(12)
    );
    for (name, expected) in [
        ("prod-grok-backend.json", Some("grok")),
        ("renamed.json", None),
    ] {
        let path = PathBuf::from(name);
        assert_eq!(
            routes(&[(path.clone(), body.clone().into_bytes())], &path, None).unwrap(),
            [(name.to_owned(), expected)]
        );
    }
    let first = r#"{"conversations":[]}"#;
    let renamed = PathBuf::from("renamed.json");
    assert_eq!(
        routes(
            &[(renamed.clone(), first.as_bytes().to_vec())],
            &renamed,
            None
        )
        .unwrap(),
        [("renamed.json".to_owned(), None)]
    );
}

/// An OpenRouter room export is recognized by its shape (a `version`, a
/// `messages` object, and the room's `id` or `characters`) under any name;
/// the parser's own rule required the `OpenRouter Chat` file name.
#[test]
fn an_openrouter_room_is_recognized_by_its_shape() {
    let room = r#"{"characters":{},"version":"1","messages":{}}"#;
    for name in ["OpenRouter Chat 2026-10-27.json", "room.json"] {
        let path = PathBuf::from(name);
        assert_eq!(
            routes(&[(path.clone(), room.as_bytes().to_vec())], &path, None).unwrap(),
            [(name.to_owned(), Some("openrouter"))]
        );
    }
    let not_a_room = r#"{"version":"1","messages":[]}"#;
    let path = PathBuf::from("room.json");
    assert_eq!(
        routes(
            &[(path.clone(), not_a_room.as_bytes().to_vec())],
            &path,
            None
        )
        .unwrap(),
        [("room.json".to_owned(), None)]
    );
}

/// A file named as a ZIP archive whose bytes are not one is refused, never
/// decoded as another container.
#[test]
fn a_zip_named_file_without_zip_bytes_is_refused() {
    let mut tar = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(CHAT.len() as u64);
    header.set_cksum();
    tar.append_data(&mut header, "conversations.json", CHAT.as_bytes())
        .unwrap();
    let path = PathBuf::from("export.zip");
    let error = routes(&[(path.clone(), tar.into_inner().unwrap())], &path, None).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        error.to_string().contains("named as a ZIP archive"),
        "{error}"
    );
}
