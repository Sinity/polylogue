use super::*;

fn cell(body: &str, app: Option<&str>) -> String {
    let caption = match app {
        Some(id) => format!(
            "Products:<br>&emsp;Gemini Apps<br>Details:<br>&emsp;{APP}{id}: <a href=\"{APP}{id}\">{APP}{id}</a><br>"
        ),
        None => "Products:<br>&emsp;Gemini Apps<br>".into(),
    };
    format!(
        "<div class=\"outer-cell mdl-cell\"><div class=\"mdl-grid\"><div class=\"header-cell\"><p>Gemini Apps<br></p></div>\
         <div class=\"content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1\">{body}</div>\
         <div class=\"content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1 mdl-typography--text-right\"></div>\
         <div class=\"content-cell mdl-cell mdl-cell--12-col mdl-typography--caption\">{caption}</div></div></div>"
    )
}

#[test]
fn prompts_group_into_their_conversation_oldest_first() {
    let html = format!(
        "<html><body>{}{}{}</body></html>",
        cell(
            "Prompted\u{a0}and then?<br>Jul 27, 2026, 11:00:00\u{202f}PM CEST<br><p>Then <strong>this</strong> &amp; that.</p>",
            Some("abc123")
        ),
        cell(
            "Prompted\u{a0}first question<br>Attached 1 file.<br>-&nbsp;<a href=\"notes.txt\">notes.txt</a><br>Jul 27, 2026, 10:59:30\u{202f}PM CEST<br><ul><li>one</li><li>two</li></ul>",
            Some("abc123")
        ),
        cell(
            "Used\u{a0}a Gem<br>Jul 26, 2026, 9:00:00\u{202f}AM CEST<br>",
            None
        ),
    );
    let step = crate::export::fixture::parse(
        &mut open(Path::new("MyActivity.html")).unwrap(),
        html.as_bytes(),
    )
    .unwrap();
    let msgs: Vec<&Message> = step.output.messages.iter().map(|r| &r.message).collect();
    assert_eq!(msgs.len(), 4);
    assert_eq!(msgs[0].text(), "first question");
    assert!(matches!(&msgs[0].blocks[1], Block::File(a) if a.name.as_deref() == Some("notes.txt")));
    assert_eq!(msgs[1].text(), "- one\n- two");
    assert_eq!(msgs[2].text(), "and then?");
    assert_eq!(msgs[2].parent, Some(msgs[1].id.clone()));
    assert_eq!(msgs[3].text(), "Then this & that.");
    assert_eq!(msgs[0].session.native().as_str(), "abc123");
    // 2026-07-27 20:59:30 UTC.
    assert_eq!(msgs[0].at.unwrap().unix_ms(), 1_785_185_970_000);
    let activity = step
        .output
        .events
        .iter()
        .find(|e| e.kind == "activity")
        .unwrap();
    assert_eq!(activity.data["verb"], "Used");
    assert_eq!(step.accounting.framed, 3);
}

#[test]
fn times_parse_and_unknown_zones_stay_unknown() {
    assert_eq!(
        parse_time("Jan 1, 2026, 12:00:00\u{202f}AM UTC")
            .unwrap()
            .unix_ms(),
        1_767_225_600_000
    );
    assert!(parse_time("Jan 1, 2026, 12:00:00 AM XYZ").is_none());
    assert!(recognize(
        Path::new("/t.zip!/Takeout/My Activity/Gemini Apps/MyActivity.html"),
        b"<html>"
    ));
}

#[test]
fn malformed_civil_dates_stay_unknown_without_panics() {
    for text in [
        "Feb 31, 2026, 12:00:00 AM UTC",
        "Jan 0, 2026, 12:00:00 AM UTC",
        "Jan 1, 9223372036854775807, 12:00:00 AM UTC",
        "Jan 1, -9223372036854775808, 12:00:00 AM UTC",
        "Jan 1, 2026, 9223372036854775807:00:00 PM UTC",
        "Jan 1, 2026, 0:00:00 AM UTC",
        "Jan 1, 2026, 12:60:00 AM UTC",
    ] {
        assert!(
            matches!(std::panic::catch_unwind(|| parse_time(text)), Ok(None)),
            "{text}"
        );
    }
}

#[test]
fn unknown_and_equal_times_keep_stable_order_and_conversation_links() {
    let html = format!(
        "<html>{}{}{}{}</html>",
        cell(
            "Prompted&nbsp;known first<br>Jan 1, 2026, 12:00:00 AM UTC<br>reply",
            Some("a")
        ),
        cell(
            "Prompted&nbsp;unknown<br>uncertain zone<br>reply",
            Some("unknown")
        ),
        cell(
            "Prompted&nbsp;known second<br>Jan 1, 2026, 12:00:00 AM UTC<br>reply",
            Some("b")
        ),
        format_args!("{CELL}\">no body</div>")
    );
    let step = crate::export::fixture::parse(
        &mut open(Path::new("MyActivity.html")).unwrap(),
        html.as_bytes(),
    )
    .unwrap();
    let prompts = step
        .output
        .messages
        .iter()
        .filter(|r| r.message.role == Role::User)
        .collect::<Vec<_>>();
    assert_eq!(
        prompts.iter().map(|r| r.message.text()).collect::<Vec<_>>(),
        ["unknown", "known first", "known second"]
    );
    assert!(prompts[0].message.at.is_none());
    assert_eq!(step.accounting.framed, 4);
    assert_eq!(step.accounting.refused.values().sum::<u64>(), 1);
}
