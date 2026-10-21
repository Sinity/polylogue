use super::*;

fn cell(id: &str, body: &str) -> String {
    format!(
        r#"<div class="outer-cell mdl-cell"><div class="mdl-typography--body-1">{body}</div><a href="https://gemini.google.com/app/{id}">conversation</a></div>"#
    )
}

#[test]
fn takeout_cells_keep_interleaved_parents_and_events_after_retained_rebuild() {
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
        assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 6);
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
    let rebuilt = e.converge().unwrap();
    assert!(rebuilt.errors.is_empty(), "{:?}", rebuilt.errors);
    assert_eq!(
        [w.export("gemini-app:alpha"), w.export("gemini-app:bravo")],
        expected
    );
    assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 6);
    assert_eq!(e.converge().unwrap().bytes_read, 0);
}
