use super::*;
use polylogue_model::Timestamp;

const T: i64 = 1_792_332_000_000; // 14:00 local in these tests

fn hour(at: i64) -> u8 {
    (14 + (at - T).div_euclid(HOUR)).rem_euclid(24) as u8
}

fn at(h: i64, m: i64) -> i64 {
    T + (h - 14) * HOUR + m * MIN
}

fn base(id: &str, query: &str) -> Rule {
    Rule {
        id: id.into(),
        name: id.into(),
        query: query.into(),
        for_ms: 0,
        interrupt: false,
        quiet_hours: None,
        rate_limit: None,
        operator: true,
    }
}

fn m(rule: &str, rows: &[&str]) -> HashMap<String, Vec<String>> {
    HashMap::from([(
        rule.to_owned(),
        rows.iter().map(|r| (*r).to_owned()).collect(),
    )])
}

fn fired(h: &[Handoff]) -> Vec<(String, How)> {
    h.iter().map(|h| (h.row.clone(), h.how.clone())).collect()
}

#[test]
fn a_duration_fires_at_its_deadline_with_no_new_data() {
    // `sessions where live:waiting` with `for: 20m`: V1 waits from
    // 14:41 and fires at 15:01; a wait answered before 20 minutes fires
    // nothing.
    let rule = Rule {
        for_ms: 20 * MIN,
        ..base("waiting", "sessions where live:waiting")
    };
    let mut e = Evaluator::new(&[]);
    assert!(
        e.step(
            at(14, 30),
            std::slice::from_ref(&rule),
            &m("waiting", &[]),
            hour
        )
        .is_empty()
    );
    assert!(
        e.step(
            at(14, 41),
            std::slice::from_ref(&rule),
            &m("waiting", &["v1", "v2"]),
            hour
        )
        .is_empty()
    );
    assert_eq!(e.next_due(), Some(at(15, 1)));
    // V2's wait is answered at 14:50.
    assert!(
        e.step(
            at(14, 50),
            std::slice::from_ref(&rule),
            &m("waiting", &["v1"]),
            hour
        )
        .is_empty()
    );
    let h = e.step(
        at(15, 1),
        std::slice::from_ref(&rule),
        &m("waiting", &["v1"]),
        hour,
    );
    assert_eq!(fired(&h), vec![("v1".to_owned(), How::Notice)]);
    assert_eq!(h[0].matched, at(14, 41));
    // Still waiting: no second notice for the same wait.
    assert!(
        e.step(at(15, 30), &[rule], &m("waiting", &["v1"]), hour)
            .is_empty()
    );
}

#[test]
fn on_new_fires_each_row_once_and_the_first_reading_is_a_baseline() {
    let rule = base("failing", "tool_calls where outcome:error");
    let mut e = Evaluator::new(&[]);
    assert!(
        e.step(
            at(14, 0),
            std::slice::from_ref(&rule),
            &m("failing", &["old"]),
            hour
        )
        .is_empty()
    );
    let h = e.step(
        at(14, 5),
        std::slice::from_ref(&rule),
        &m("failing", &["old", "a", "b"]),
        hour,
    );
    assert_eq!(
        fired(&h),
        vec![("a".into(), How::Notice), ("b".into(), How::Notice)]
    );
    assert!(
        e.step(at(14, 6), &[rule], &m("failing", &["old", "a", "b"]), hour)
            .is_empty()
    );
}

#[test]
fn past_the_rate_limit_a_notice_is_kept_for_the_briefs() {
    let rule = Rule {
        rate_limit: Some((1, HOUR)),
        ..base("failing", "q")
    };
    let mut e = Evaluator::new(&[]);
    e.step(
        at(14, 0),
        std::slice::from_ref(&rule),
        &m("failing", &[]),
        hour,
    );
    let h = e.step(
        at(14, 5),
        std::slice::from_ref(&rule),
        &m("failing", &["a", "b"]),
        hour,
    );
    assert_eq!(
        fired(&h),
        vec![("a".into(), How::Notice), ("b".into(), How::Deferred)]
    );
    // An hour later the limit allows one again.
    let h = e.step(at(15, 6), &[rule], &m("failing", &["a", "b", "c"]), hour);
    assert_eq!(fired(&h), vec![("c".into(), How::Notice)]);
}

#[test]
fn quiet_hours_hold_an_interrupt_but_not_a_passive_notice() {
    let loud = Rule {
        interrupt: true,
        quiet_hours: Some((14, 16)),
        ..base("loud", "q")
    };
    let calm = Rule {
        quiet_hours: Some((14, 16)),
        ..base("calm", "q")
    };
    let rules = [loud, calm];
    let mut e = Evaluator::new(&[]);
    let none: HashMap<String, Vec<String>> =
        HashMap::from([("loud".into(), vec![]), ("calm".into(), vec![])]);
    e.step(at(14, 0), &rules, &none, hour);
    let both: HashMap<String, Vec<String>> = HashMap::from([
        ("loud".into(), vec!["x".into()]),
        ("calm".into(), vec!["x".into()]),
    ]);
    let h = e.step(at(14, 30), &rules, &both, hour);
    assert_eq!(
        h.iter().map(|h| h.rule.as_str()).collect::<Vec<_>>(),
        ["calm"]
    );
    let h = e.step(at(16, 0), &rules, &both, hour);
    assert_eq!(fired(&h), vec![("x".into(), How::Interrupt)]);
}

#[test]
fn agents_share_one_interrupt_budget_and_the_operator_is_outside_it() {
    let agent = |id: &str| Rule {
        interrupt: true,
        operator: false,
        ..base(id, "q")
    };
    let operator = Rule {
        interrupt: true,
        ..base("op", "q")
    };
    let rules = [agent("a1"), agent("a2"), agent("a3"), operator];
    let mut e = Evaluator {
        budget: Some(2),
        ..Evaluator::default()
    };
    let empty: HashMap<String, Vec<String>> =
        rules.iter().map(|r| (r.id.clone(), vec![])).collect();
    e.step(at(14, 0), &rules, &empty, hour);
    let all: HashMap<String, Vec<String>> = rules
        .iter()
        .map(|r| (r.id.clone(), vec!["x".to_owned()]))
        .collect();
    let h = e.step(at(14, 1), &rules, &all, hour);
    let by: HashMap<&str, (&How, bool)> = h
        .iter()
        .map(|h| (h.rule.as_str(), (&h.how, h.why.is_some())))
        .collect();
    let interrupts = ["a1", "a2", "a3"]
        .iter()
        .filter(|r| *by[*r].0 == How::Interrupt)
        .count();
    assert_eq!(interrupts, 2, "{by:?}");
    assert!(by.values().any(|(how, why)| **how == How::Notice && *why));
    assert_eq!(by["op"], (&How::Interrupt, false));
}

#[test]
fn a_withdrawn_rule_delivers_nothing_more() {
    let rule = base("r", "q");
    let mut e = Evaluator::new(&[]);
    e.step(at(14, 0), std::slice::from_ref(&rule), &m("r", &[]), hour);
    assert_eq!(e.step(at(14, 1), &[rule], &m("r", &["a"]), hour).len(), 1);
    // Withdrawn: the rule is no longer among the current ones.
    assert!(
        e.step(at(14, 2), &[], &m("r", &["a", "b"]), hour)
            .is_empty()
    );
}

#[test]
fn a_restart_neither_repeats_nor_forgets() {
    let rule = base("r", "q");
    let history = [Sent {
        rule: "r".into(),
        row: "a".into(),
        at: at(14, 1),
        interrupt_by_agent: false,
    }];
    let mut e = Evaluator::new(&history);
    let h = e.step(at(14, 5), &[rule], &m("r", &["a", "b"]), hour);
    assert_eq!(fired(&h), vec![("b".into(), How::Notice)]);
}

#[test]
fn rule_notes_are_read_or_refused_by_name() {
    let note = |value: &str, author: Author| Note {
        id: "n2".into(),
        first_note: "n1".into(),
        kind: NoteKind::Signal,
        key: None,
        value: Some(value.into()),
        author,
        created_at: Timestamp::from_unix_ms(T).unwrap(),
        valid_until: None,
        targets: vec![],
        relations: vec![],
    };
    let r = rule(&note(
        r#"{"name":"waiting","query":"sessions where live:waiting","for":"20m","interrupt":true,"quiet_hours":[22,7],"rate_limit":{"count":1,"window":"1h"}}"#,
        Author::Agent(None),
    ))
    .unwrap();
    assert_eq!(r.id, "n1", "a revision keeps the rule's identity");
    assert_eq!(r.for_ms, 20 * MIN);
    assert_eq!(r.quiet_hours, Some((22, 7)));
    assert_eq!(r.rate_limit, Some((1, HOUR)));
    assert!(!r.operator);
    let e = rule(&note(
        r#"{"query":"q","deliver":"action"}"#,
        Author::Operator,
    ))
    .unwrap_err();
    assert!(e.reason.contains("not built"), "{}", e.reason);
    assert!(rule(&note("not json", Author::Operator)).is_err());
    // The operator's rules reach the desktop unless they say otherwise;
    // agents' rules only when they ask.
    let plain = r#"{"query":"q"}"#;
    assert!(rule(&note(plain, Author::Operator)).unwrap().interrupt);
    assert!(!rule(&note(plain, Author::Agent(None))).unwrap().interrupt);
    let script = rule(&note(plain, Author::Script("example-unit".into()))).unwrap();
    assert!(!script.operator);
    assert!(!script.interrupt);
    let explicit_script = rule(&note(
        r#"{"query":"q","interrupt":true}"#,
        Author::Script("example-unit".into()),
    ))
    .unwrap();
    assert!(!explicit_script.operator);
    assert!(explicit_script.interrupt);
    let quiet = r#"{"query":"q","interrupt":false}"#;
    assert!(!rule(&note(quiet, Author::Operator)).unwrap().interrupt);
}

#[test]
fn an_interrupt_runs_the_notify_command_and_reports_its_failure() {
    use crate::daemon::{Request, Response, service::handle};
    use std::os::unix::fs::PermissionsExt;
    let d = crate::test_daemon::start("hello");
    let out = d.home.join("notified");
    let script = d.home.join("notify");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\nprintf '%s|%s\\n' \"$1\" \"$2\" >> {}\n",
            out.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let set = |value: &str| {
        let r = handle(
            Request::Setting {
                request_id: Some(polylogue_store::user_writes::new_request_id()),
                key: "notify.command".into(),
                value: Some(value.into()),
                machine: false,
            },
            &d.daemon,
        );
        assert!(matches!(r, Response::Ok { .. }), "{r:?}");
    };
    let rule = Rule {
        name: "waiting".into(),
        ..base("r1", "q")
    };
    let h = Handoff {
        rule: "r1".into(),
        row: "claude-code:s1".into(),
        matched: 0,
        at: 0,
        how: How::Interrupt,
        why: None,
    };
    set(&script.to_string_lossy());
    notify(&d.daemon, &rule, &h).unwrap();
    assert_eq!(
        std::fs::read_to_string(&out).unwrap(),
        "Polylogue: waiting|waiting: claude-code:s1\n"
    );
    set("/nonexistent/notifier");
    let e = notify(&d.daemon, &rule, &h).unwrap_err();
    assert!(e.contains("/nonexistent/notifier"), "{e}");
}

#[test]
fn a_rule_not_read_exactly_is_held_as_it_was() {
    let rule = base("errors", "tool_calls where outcome:error");
    let rules = std::slice::from_ref(&rule);
    let mut e = Evaluator::new(&[]);
    // While the index catches up the rule is not read: nothing is a
    // baseline yet.
    assert!(e.step(at(14, 0), rules, &HashMap::new(), hour).is_empty());
    // The first exact reading is the baseline.
    assert!(
        e.step(at(14, 1), rules, &m("errors", &["a"]), hour)
            .is_empty()
    );
    // An incomplete reading that misses `a` neither drops it nor fires.
    assert!(e.step(at(14, 2), rules, &HashMap::new(), hour).is_empty());
    assert!(
        e.step(at(14, 3), rules, &m("errors", &["a"]), hour)
            .is_empty()
    );
    // A row that is new in an exact reading fires once.
    let h = e.step(at(14, 4), rules, &m("errors", &["a", "b"]), hour);
    assert_eq!(fired(&h), [("b".to_owned(), How::Notice)]);
}
