use super::*;
#[test]
fn row_number_ranges_and_hits_resolve() {
    let shown = ShownRows {
        question: "clock".into(),
        shown_at: 0,
        rows: vec![
            Row {
                reference: "session:codex:first".into(),
                hits: vec![
                    "message:codex:first#1".into(),
                    "message:codex:first#2".into(),
                ],
            },
            Row {
                reference: "session:codex:second".into(),
                hits: vec![],
            },
            Row {
                reference: "session:codex:third".into(),
                hits: vec![],
            },
        ],
    };
    assert_eq!(
        select("@1-2,3", &shown).unwrap(),
        [
            "session:codex:first",
            "session:codex:second",
            "session:codex:third"
        ]
    );
    assert_eq!(select("@1.2", &shown).unwrap(), ["message:codex:first#2"]);
    assert_eq!(
        select("@all", &shown).unwrap(),
        [
            "session:codex:first",
            "session:codex:second",
            "session:codex:third"
        ]
    );
    assert_eq!(
        select("@1,", &shown).unwrap_err(),
        "@1,: a row number is missing between commas"
    );
    assert_eq!(
        select("@2.1", &shown).unwrap_err(),
        "row @2 has no hit 1; use @2"
    );
    assert_eq!(select("@0", &shown).unwrap_err(), "@0: rows start at 1");
    assert_eq!(
        select("@1-2.1", &shown).unwrap_err(),
        "@1-2.1: a row range has no hit; use @1-2 or @2.1"
    );
    assert_eq!(
        select("@3-2", &shown).unwrap_err(),
        "@3-2 is not a row range"
    );
    assert_eq!(
        select("@2-4", &shown).unwrap_err(),
        "@2-4: the last list has 3 rows"
    );
}

#[test]
fn row_selection_words_are_recognised() {
    for word in ["@all", "@1", "@12", "@1,3", "@1-3", "@2.1", "@1-2,4.1"] {
        assert!(is_selection(word), "{word}");
    }
    for word in [
        "@", "@a", "all", "@0x", "@,1", "@-1", "1", "@all1", "user@1",
    ] {
        assert!(!is_selection(word), "{word}");
    }
}

#[test]
fn all_of_an_empty_list_is_refused() {
    let shown = ShownRows {
        question: "clock".into(),
        shown_at: 0,
        rows: vec![],
    };
    assert_eq!(
        select("@all", &shown).unwrap_err(),
        "@all: the last list has 0 rows"
    );
}

#[test]
fn failed_row_write_leaves_no_stale_listing() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("rows-pts-3");
    let listing = |reference: &str| ShownRows {
        question: "clock".into(),
        shown_at: 0,
        rows: vec![Row {
            reference: reference.into(),
            hits: vec![],
        }],
    };
    replace_listing(&path, &listing("session:codex:older"), |path, bytes| {
        polylogue_model::atomic_write::write_atomic(path, bytes, None)
    });
    let stored: ShownRows = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(
        select("@1", &stored).unwrap(),
        ["session:codex:older".to_owned()]
    );
    replace_listing(&path, &listing("session:codex:newer"), |_, _| {
        Err(std::io::Error::from_raw_os_error(28))
    });
    assert_eq!(
        fs::read(&path).unwrap_err().kind(),
        std::io::ErrorKind::NotFound
    );
}
