use super::*;
use crate::protocol::line::{SaveReport, StepAnswer};
use polylogue_model::{
    Author, Origin, RevisionReference,
    authorship::{Authorship, Channel},
};

fn agent(d: &TestDaemon, id: &str, arguments: Value) -> command_surface::Answer<Value> {
    Call::new("rows", arguments)
        .id(id)
        .authorship(Authorship {
            author: Author::AgentOrigin(Origin::new("codex").unwrap()),
            via: Channel::Mcp,
        })
        .answer(&d.daemon)
}

#[track_caller]
fn rows(d: &TestDaemon, id: &str, selection: &str) -> Value {
    let answer = crate::test_answers::listing(agent(d, id, json!({"selection":selection})));
    answer
        .value()
        .unwrap_or_else(|| panic!("{answer:?}"))
        .clone()
}

#[track_caller]
fn save(d: &TestDaemon, id: &str, selection: &str, name: Option<&str>, watch: bool) -> SaveReport {
    match crate::test_answers::step(agent(
        d,
        id,
        json!({
            "selection":selection,
            "steps":[{"verb":"save","name":name,"watch":watch}]
        }),
    )) {
        StepAnswer::Saved(report) => report,
        other => panic!("expected accepted Save: {other:?}"),
    }
}

fn session_ids(value: &Value) -> std::collections::BTreeSet<String> {
    value["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn shipped_query_resolves_and_withdrawing_an_authored_override_restores_it() {
    let d = two_sessions();
    let original = rows(&d, "shipped-recent", "query:recent");
    assert_eq!(
        session_ids(&original),
        ["claude-code:s1".into(), "claude-code:s2".into()].into()
    );
    let overridden = save(
        &d,
        "override-recent",
        "sessions where id:claude-code:s1",
        Some("recent"),
        false,
    );
    let definition = overridden.saved.as_ref().unwrap();
    assert!(definition.new_revision);
    assert_eq!(definition.revision, 1);
    let current = rows(&d, "authored-recent", "query:recent");
    assert_eq!(session_ids(&current), ["claude-code:s1".into()].into());
    assert_eq!(current["definitions"][0]["revision"], definition.id);
    let reader = d.daemon.reads.reader().unwrap();
    let fact = reader
        .annotation(&RevisionReference::exact(
            definition.id.clone(),
            definition.id.clone(),
        ))
        .unwrap()
        .unwrap()
        .fact;
    assert_eq!(
        fact.author,
        Author::AgentOrigin(Origin::new("codex").unwrap())
    );
    assert_eq!(fact.via, Channel::Mcp);
    drop(reader);
    let undone = crate::test_answers::step(agent(
        &d,
        "withdraw-recent",
        json!({
            "references":[format!("annotation:{}@{}", definition.id, definition.id)],
            "steps":[{"verb":"withdraw"}]
        }),
    ));
    assert!(matches!(undone, StepAnswer::Wrote(_)), "{undone:?}");
    assert_eq!(
        session_ids(&rows(&d, "restored-recent", "query:recent")),
        session_ids(&original)
    );
    assert_eq!(
        session_ids(&rows(&d, "override-kept", &overridden.saved_rows.result)),
        ["claude-code:s1".into()].into()
    );
}

#[test]
fn resaving_the_same_question_keeps_new_rows_without_a_definition_revision() {
    let d = two_sessions();
    let question = "sessions where id:claude-code:s1";
    let first = save(&d, "same-question-first", question, Some("focused"), false);
    let second = save(&d, "same-question-second", question, Some("focused"), false);
    let first_saved = first.saved.as_ref().unwrap();
    let second_saved = second.saved.as_ref().unwrap();
    assert!(first_saved.new_revision);
    assert!(!second_saved.new_revision);
    assert_eq!(second_saved.id, first_saved.id);
    assert_eq!((first_saved.revision, second_saved.revision), (1, 1));
    assert!(!first.repeat && !second.repeat);
    assert_ne!(first.saved_rows.result, second.saved_rows.result);
    assert_eq!((first.saved_rows.rows, second.saved_rows.rows), (1, 1));
    assert_eq!(
        rows(&d, "same-first-kept", &first.saved_rows.result)["rows"],
        rows(&d, "same-second-kept", &second.saved_rows.result)["rows"]
    );
    let reader = d.daemon.reads.reader().unwrap();
    let history = reader.saved_query_record("focused").unwrap().unwrap();
    assert_eq!(history.revisions.len(), 1);
    assert_eq!(
        history.revisions[0].reference,
        format!("annotation:{}@{}", first_saved.id, first_saved.id)
    );
    assert_eq!(
        history
            .saved_rows
            .iter()
            .map(|part| part.result.clone())
            .collect::<std::collections::BTreeSet<_>>(),
        [first.saved_rows.result, second.saved_rows.result].into()
    );
}

#[test]
fn changing_a_saved_question_moves_its_existing_watch_to_the_exact_new_revision() {
    let d = two_sessions();
    let first = save(
        &d,
        "watched-first",
        "sessions where id:claude-code:s1",
        Some("watched"),
        true,
    );
    let second = save(
        &d,
        "watched-second",
        "sessions where id:claude-code:s2",
        Some("watched"),
        false,
    );
    let first_watch = first.watch.as_ref().unwrap();
    let second_watch = second.watch.as_ref().unwrap();
    assert!(first_watch.new_rule && second_watch.new_rule);
    assert_ne!(first_watch.id, second_watch.id);
    assert!(second.saved.as_ref().unwrap().new_revision);
    assert_eq!(second.saved.as_ref().unwrap().revision, 2);
    let reader = d.daemon.reads.reader().unwrap();
    for (id, expected) in [
        (&first_watch.id, "query:watched@1"),
        (&second_watch.id, "query:watched@2"),
    ] {
        let fact = reader
            .annotation(&RevisionReference::exact(
                first_watch.id.clone(),
                id.clone(),
            ))
            .unwrap()
            .unwrap()
            .fact;
        assert_eq!(fact.value.as_ref().unwrap()["query"], expected);
    }
    drop(reader);
    assert_eq!(
        session_ids(&rows(&d, "watched-old", "query:watched@1")),
        ["claude-code:s1".into()].into()
    );
    assert_eq!(
        session_ids(&rows(&d, "watched-new", "query:watched@2")),
        ["claude-code:s2".into()].into()
    );
    assert_eq!(
        session_ids(&rows(&d, "watched-current", "query:watched")),
        ["claude-code:s2".into()].into()
    );
    let root = &first.saved.as_ref().unwrap().id;
    let withdrawal = crate::test_answers::step(agent(
        &d,
        "withdraw-watched",
        json!({
            "references":[format!("annotation:{root}")],
            "steps":[{"verb":"withdraw"}]
        }),
    ));
    assert!(matches!(withdrawal, StepAnswer::Wrote(_)), "{withdrawal:?}");
    assert_eq!(count(&d, "annotations where kind:(saved_query|signal)"), 0);
    assert_eq!(count(&d, "results"), 2);
}

#[test]
fn a_saved_result_name_stays_fixed_when_the_newest_named_result_changes() {
    let d = two_sessions();
    let save_at = |id: &str, selection: &str, milliseconds: i64| {
        let authorship = Authorship {
            author: Author::AgentOrigin(Origin::new("codex").unwrap()),
            via: Channel::Mcp,
        };
        let answer = Call::new(
            "rows",
            json!({"selection":selection,"steps":[{"verb":"save","name":"population"}]}),
        )
        .id(id)
        .authorship(authorship)
        .at(polylogue_model::Timestamp::from_unix_ms(milliseconds).unwrap())
        .answer(&d.daemon);
        match crate::test_answers::step(answer) {
            StepAnswer::Saved(report) => report,
            other => panic!("expected accepted Save: {other:?}"),
        }
    };
    // Caller IDs sort opposite to the original run times.
    let first = save_at(
        "z-population-first",
        "sessions where id:claude-code:s1",
        1000,
    );
    let frozen = save(
        &d,
        "freeze-population",
        "result:population",
        Some("frozen-population"),
        false,
    );
    let second = save_at(
        "a-population-second",
        "sessions where id:claude-code:s2",
        2000,
    );
    assert_ne!(first.saved_rows.result, second.saved_rows.result);
    let named = rows(&d, "named-newest", "result:population");
    assert_eq!(session_ids(&named), ["claude-code:s2".into()].into());
    assert_eq!(
        named["definitions"],
        json!([{
            "name":"result:population", "revision":second.saved_rows.result
        }])
    );
    for (id, selection) in [
        ("frozen-definition", "query:frozen-population"),
        ("first-original", first.saved_rows.result.as_str()),
        ("frozen-kept", frozen.saved_rows.result.as_str()),
    ] {
        assert_eq!(
            session_ids(&rows(&d, id, selection)),
            ["claude-code:s1".into()].into()
        );
    }
}

#[test]
fn saving_raw_feed_rows_keeps_the_exact_relation_instead_of_reselecting_entities() {
    let d = two_sessions();
    let selection = "feed where kind:session | fields kind, ref | sort by ref";
    let expected = rows(&d, "raw-feed-before", selection);
    assert!(expected["count"].as_u64().unwrap() > 0, "{expected}");
    let kept = save(&d, "raw-feed-save", selection, None, false);
    assert!(kept.saved.is_none() && kept.watch.is_none());
    assert_eq!(kept.saved_rows.rows, expected["count"].as_u64().unwrap());
    let actual = rows(&d, "raw-feed-kept", &kept.saved_rows.result);
    assert_eq!(actual["rows"], expected["rows"]);
    assert_eq!(actual["columns"], expected["columns"]);
}

#[test]
fn recreating_a_withdrawn_figure_keeps_its_watch_on_the_new_family_and_denominator() {
    let d = two_sessions();
    let figure = |id: &str, denominator: &str| match crate::test_answers::step(agent(
        &d,
        id,
        json!({
            "selection":"sessions | count", "steps":[{
                "verb":"save", "name":"restarted", "watch":true, "denominator":denominator
            }]
        }),
    )) {
        StepAnswer::Saved(report) => report,
        other => panic!("{other:?}"),
    };
    let first = figure("old-family", "sessions");
    let first_saved = first.saved.as_ref().unwrap();
    let old = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .annotation(&RevisionReference::exact(
            first_saved.id.clone(),
            first_saved.id.clone(),
        ))
        .unwrap()
        .unwrap()
        .fact;
    crate::test_answers::step(agent(
        &d,
        "withdraw-old-family",
        json!({
            "references":[format!("annotation:{}@{}",first_saved.id,first_saved.id)],
            "steps":[{"verb":"withdraw"}]
        }),
    ));
    let second = figure("new-family", "sessions where id:claude-code:s1");
    let second_saved = second.saved.as_ref().unwrap();
    assert_eq!(second_saved.revision, 1);
    assert_ne!(second_saved.id, first_saved.id);
    let watch = second.watch.as_ref().unwrap();
    let reader = d.daemon.reads.reader().unwrap();
    let new = reader
        .annotation(&RevisionReference::exact(
            second_saved.id.clone(),
            second_saved.id.clone(),
        ))
        .unwrap()
        .unwrap()
        .fact;
    let old_value: polylogue_model::annotation::values::SavedQuery =
        serde_json::from_value(old.value.clone().unwrap()).unwrap();
    let new_value: polylogue_model::annotation::values::SavedQuery =
        serde_json::from_value(new.value.clone().unwrap()).unwrap();
    assert_eq!(
        old_value.hash, new_value.hash,
        "only the denominator changed"
    );
    assert_ne!(old_value.denominator, new_value.denominator);
    let rule = reader
        .annotation(&RevisionReference::exact(
            watch.id.clone(),
            watch.id.clone(),
        ))
        .unwrap()
        .unwrap()
        .fact;
    let pinned = rule.value.unwrap()["query"].as_str().unwrap().to_owned();
    assert_eq!(pinned, "query:restarted@1");
    let bindings = reader.selection_definitions(&pinned, &[]).unwrap();
    let definition = &bindings.saved["restarted@1"][0];
    assert_eq!(definition.root, second_saved.id);
    assert_eq!(
        definition.denominator.map(|hash| format!("query:{hash}")),
        new_value.denominator
    );
    assert_eq!(
        reader
            .annotation(&RevisionReference::exact(
                first_saved.id.clone(),
                first_saved.id.clone()
            ))
            .unwrap()
            .unwrap()
            .fact,
        old
    );
    let history = reader.saved_query_record("restarted").unwrap().unwrap();
    assert_eq!(history.revisions.len(), 2);
    drop(reader);
    let expected = rows(&d, "recreated-expected", "sessions | count");
    assert_eq!(
        rows(&d, "recreated-pinned", &pinned)["rows"],
        expected["rows"]
    );
    let third = figure("same-new-family", "sessions");
    assert_eq!(third.saved.as_ref().unwrap().revision, 2);
    assert!(
        third.watch.as_ref().unwrap().new_rule,
        "denominator changes rebase the watch"
    );
    let reader = d.daemon.reads.reader().unwrap();
    let bindings = reader
        .selection_definitions("query:restarted@1", &[])
        .unwrap();
    assert_eq!(
        bindings.saved["restarted@1"][0]
            .denominator
            .map(|hash| format!("query:{hash}")),
        new_value.denominator
    );
    let newest = reader
        .selection_definitions("query:restarted@2", &[])
        .unwrap();
    assert_eq!(
        newest.saved["restarted@2"][0]
            .denominator
            .map(|hash| format!("query:{hash}")),
        old_value.denominator
    );
}
