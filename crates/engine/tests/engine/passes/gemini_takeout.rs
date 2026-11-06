use super::*;

fn cell(id: &str, body: &str) -> String {
    format!(
        r#"<div class="outer-cell mdl-cell"><div class="mdl-typography--body-1">{body}</div><a href="https://gemini.google.com/app/{id}">conversation</a></div>"#
    )
}

#[test]
fn takeout_cells_keep_interleaved_parents_and_events_after_a_rebuild() {
    let w = ClaudeHome::new();
    let path = w
        .dir
        .path()
        .join("Takeout/My Activity/Gemini Apps/MyActivity.html");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let cells = [
        cell(
            "alpha",
            "Prompted&nbsp;alpha later<br>Jan 1, 2026, 10:03:00 AM UTC<br><p>alpha reply2</p>",
        ),
        cell(
            "bravo",
            "Prompted&nbsp;bravo later<br>Jan 1, 2026, 10:02:00 AM UTC<br><p>bravo reply2</p>",
        ),
        cell(
            "alpha",
            "Used&nbsp;a Gem<br>Jan 1, 2026, 10:01:30 AM UTC<br>",
        ),
        cell(
            "bravo",
            "Prompted&nbsp;bravo first<br>Jan 1, 2026, 10:01:00 AM UTC<br><p>bravo reply1</p>",
        ),
        cell(
            "alpha",
            "Prompted&nbsp;alpha first<br>Jan 1, 2026, 10:00:00 AM UTC<br><p>alpha reply1</p>",
        ),
        r#"<div class="outer-cell mdl-cell">missing body</div>"#.into(),
    ];
    let bytes = format!("<html><body>{}</body></html>", cells.concat());
    fs::write(&path, &bytes).unwrap();
    let expected;
    {
        let (mut e, _lock) = w.engine(64);
        let report = e.import(std::slice::from_ref(&path)).unwrap();
        assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
        for name in ["alpha", "bravo"] {
            let rows = messages_of(&w.root, &format!("gemini-app:{name}"));
            assert_eq!(
                rows.iter().map(Message::text).collect::<Vec<_>>(),
                [
                    format!("{name} first"),
                    format!("{name} reply1"),
                    format!("{name} later"),
                    format!("{name} reply2")
                ]
            );
            assert_eq!(rows[2].parent, Some(rows[1].id.clone()));
        }
        let reader = w.reads();
        let r = reader.reader().unwrap();
        for row in r
            .query("SELECT source_start,source_end FROM message_source", &[])
            .unwrap()
        {
            let a = row["source_start"].as_u64().unwrap() as usize;
            let b = row["source_end"].as_u64().unwrap() as usize;
            assert!(
                cells
                    .iter()
                    .any(|cell| cell.as_bytes() == &bytes.as_bytes()[a..b])
            );
        }
        assert_eq!(w.count("SELECT SUM(records) AS n FROM parse_state"), 6);
        let alpha = w.export("gemini-app:alpha");
        let activity = alpha.events.iter().find(|e| e.kind == "activity").unwrap();
        let messages = messages_of(&w.root, "gemini-app:alpha");
        assert_eq!(activity.after, Some(messages[1].id.clone()));
        let (a, b) = activity.range.unwrap();
        assert_eq!(&bytes[a as usize..b as usize], cells[2]);
        expected = [alpha, w.export("gemini-app:bravo")];
    }
    fs::remove_file(path).unwrap();
    delete_index(&w.root);
    let (mut e, _lock) = w.engine(64);
    let rebuilt = e.run_pass().unwrap();
    assert!(rebuilt.errors.is_empty(), "{:?}", rebuilt.errors);
    assert_eq!(
        [w.export("gemini-app:alpha"), w.export("gemini-app:bravo")],
        expected
    );
    assert_eq!(w.count("SELECT SUM(records) AS n FROM parse_state"), 6);
    assert_eq!(e.run_pass().unwrap().bytes_read, 0);
}

/// A page capture of the conversation: the extension's turns, its text
/// collapsed to one line, no times.
fn page_capture(turns: &[(&str, &str)]) -> Vec<u8> {
    let turns: Vec<serde_json::Value> = turns
        .iter()
        .enumerate()
        .map(|(i, (role, text))| {
            serde_json::json!({"provider_turn_id": format!("chat1:turn:{i}"), "role": role,
                               "text": text, "ordinal": i, "timestamp": null})
        })
        .collect();
    serde_json::to_vec(&serde_json::json!({
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "source": "browser-extension",
        "provenance": {"captured_at": "2026-01-02T00:00:00Z"},
        "provider_meta": {},
        "raw_provider_payload": {},
        "session": {"provider": "gemini", "provider_session_id": "chat1", "turns": turns}
    }))
    .unwrap()
}

/// The texts the session shows, sorted, and how many messages it holds.
fn shown(w: &ClaudeHome) -> (Vec<String>, i64) {
    let mut texts: Vec<String> = messages_of(&w.root, "gemini-app:chat1")
        .iter()
        .map(Message::text)
        .collect();
    texts.sort();
    (texts, w.count("SELECT count(*) AS n FROM message"))
}

#[test]
fn a_turn_in_a_takeout_and_a_page_capture_is_shown_once() {
    let takeout = format!(
        "<html><body>{}{}{}</body></html>",
        cell(
            "chat1",
            "Prompted&nbsp;continue<br>Jan 1, 2026, 10:02:00 AM UTC<br><p>more</p>",
        ),
        cell(
            "chat1",
            "Prompted&nbsp;second<br>Jan 1, 2026, 10:01:00 AM UTC<br><p>reply two</p>",
        ),
        cell(
            "chat1",
            "Prompted&nbsp;first\n  question<br>Jan 1, 2026, 10:00:00 AM UTC<br><p>reply</p><p>one</p>",
        ),
    );
    let capture = page_capture(&[
        ("user", "first question"),
        ("assistant", "reply one"),
        ("user", "second"),
        ("assistant", "reply two"),
        ("user", "continue"),
        ("assistant", "more"),
        ("user", "continue"),
        ("assistant", "even more"),
        ("user", "third"),
        ("assistant", "reply three"),
    ]);
    let mut results = Vec::new();
    for capture_first in [false, true] {
        let w = ClaudeHome::new();
        let export = w
            .dir
            .path()
            .join("Takeout/My Activity/Gemini Apps/MyActivity.html");
        fs::create_dir_all(export.parent().unwrap()).unwrap();
        fs::write(&export, &takeout).unwrap();
        let spool = w.home.join(".local/state/polylogue/browser-capture/gemini");
        fs::create_dir_all(&spool).unwrap();
        let (mut e, _lock) = w.engine(64 << 20);
        if capture_first {
            fs::write(spool.join("chat1.json"), &capture).unwrap();
            assert!(e.run_pass().unwrap().errors.is_empty());
            let report = e.import(std::slice::from_ref(&export)).unwrap();
            assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
        } else {
            let report = e.import(std::slice::from_ref(&export)).unwrap();
            assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
            fs::write(spool.join("chat1.json"), &capture).unwrap();
            assert!(e.run_pass().unwrap().errors.is_empty());
        }
        let (texts, stored) = shown(&w);
        // Both arrivals stay stored: six Takeout messages, ten captured.
        assert_eq!(stored, 16);
        // The two paired turns show from the Takeout; "continue" occurs
        // twice in the capture, so neither of its turns is matched.
        assert_eq!(
            texts,
            [
                "continue",
                "continue",
                "continue",
                "even more",
                "first\n  question",
                "more",
                "more",
                "reply\none",
                "reply three",
                "reply two",
                "second",
                "third",
            ]
        );
        assert_eq!(
            w.count(
                "SELECT count(*) AS n FROM message WHERE duplicate_of LIKE 'gemini-app:chat1#%'
                 AND duplicate_reason IS NOT NULL"
            ),
            4
        );
        results.push(texts);
    }
    assert_eq!(results[0], results[1]);
}
