use super::*;

#[test]
fn a_recorded_title_beats_a_file_name_in_either_arrival_order() {
    for recorded_first in [false, true] {
        let mut s = Session::new(SessionId::parse("ai-studio:p").unwrap(), SessionKind::Main);
        let recorded = ("Electrons", TitleSource::Generated);
        let file = ("drive-id", TitleSource::FileName);
        for (text, source) in if recorded_first {
            [recorded, file]
        } else {
            [file, recorded]
        } {
            s.offer_title(text, source);
        }
        assert_eq!(s.title.unwrap().text, "Electrons");
    }
}

#[test]
fn merge_is_idempotent_and_keeps_the_earliest_start() {
    let mut a = session();
    a.offer_cwd("/a");
    a.offer_start(Timestamp::from_unix_ms(50).unwrap());
    let mut b = session();
    b.offer_cwd("/a");
    b.offer_cwd("/b");
    b.offer_title("t", TitleSource::Generated);
    b.offer_start(Timestamp::from_unix_ms(10).unwrap());
    a.merge(&b);
    let once = a.clone();
    a.merge(&b);
    assert_eq!(a, once);
    assert_eq!(a.cwd, ["/a", "/b"]);
    assert_eq!(a.started, Timestamp::from_unix_ms(10));
    assert_eq!(a.title.unwrap().text, "t");
}

#[test]
fn merge_of_unrelated_directories_appends_new_ones() {
    let mut a = session();
    a.offer_cwd("/x");
    let mut b = session();
    b.offer_cwd("/y");
    b.offer_cwd("/x");
    a.merge(&b);
    assert_eq!(a.cwd, ["/x", "/y"]);
}

fn session() -> Session {
    Session::new(
        SessionId::parse("claude-code:s").unwrap(),
        SessionKind::Main,
    )
}

#[test]
fn custom_title_beats_generated_and_summary_in_any_order() {
    let mut s = session();
    s.offer_title("summary", TitleSource::Summary);
    s.offer_title("custom", TitleSource::Custom);
    s.offer_title("generated", TitleSource::Generated);
    assert_eq!(s.title.unwrap().text, "custom");
}

#[test]
fn later_title_of_same_source_wins() {
    let mut s = session();
    s.offer_title("first", TitleSource::Generated);
    s.offer_title("second", TitleSource::Generated);
    assert_eq!(s.title.unwrap().text, "second");
}

#[test]
fn cwd_keeps_order_without_repeats_in_a_row() {
    let mut s = session();
    for d in ["/a", "/a", "/b", "/a"] {
        s.offer_cwd(d);
    }
    assert_eq!(s.cwd, ["/a", "/b", "/a"]);
}
