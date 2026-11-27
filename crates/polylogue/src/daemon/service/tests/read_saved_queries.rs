use super::*;
use crate::test_daemon::{self, Call};
use polylogue_model::read::*;
use serde_json::json;

fn save(d: &Daemon, id: &str, name: &str, text: &str, watch: bool) {
    let answer = Call::new(
        "rows",
        json!({"selection":text,"steps":[{"verb":"save","name":name,"watch":watch}]}),
    )
    .id(id)
    .answer(d);
    assert!(matches!(
        crate::test_answers::step(answer),
        crate::protocol::line::StepAnswer::Saved(_)
    ));
}
fn read(d: &Daemon, selection: &str) -> ReadPage {
    let words = vec![
        selection.to_owned(),
        "read".to_owned(),
        "--budget".to_owned(),
        "100000".to_owned(),
    ];
    let parsed =
        crate::cli::line::parse(&words, &crate::cli::line::SavedNames::default(), false).unwrap();
    assert_eq!(
        parsed.selection,
        Some(crate::cli::line::SelectionWord::Text(selection.to_owned()))
    );
    let steps: Vec<_> = parsed.steps.into_iter().map(|step| step.step).collect();
    crate::test_answers::read(
        Call::new("rows", json!({"selection":selection,"steps":steps})).answer(d),
    )
}

fn header(page: &ReadPage) -> &SavedQueryHeader {
    let HeaderKind::SavedQuery(saved) = &page.document.header.kind else {
        panic!("saved query header: {:?}", page.document.header)
    };
    saved
}

fn history_items(page: &ReadPage) -> impl Iterator<Item = &SavedQueryItem> {
    page.document
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Card { card } => match card.as_ref() {
                Card::SavedQuery { item } => Some(item.as_ref()),
                _ => None,
            },
            _ => None,
        })
}
fn history(page: &ReadPage) -> polylogue_model::saved_queries::SavedQueryRecord {
    let mut record = polylogue_model::saved_queries::SavedQueryRecord {
        name: header(page).name.clone(),
        revisions: vec![],
        saved_rows: vec![],
        watch: None,
        dependents: vec![],
    };
    for item in history_items(page) {
        match item {
            SavedQueryItem::Revision(revision) => record.revisions.push(revision.clone()),
            SavedQueryItem::Kept(kept) => record.saved_rows.push(kept.clone()),
            SavedQueryItem::Watch { reference } => record.watch.clone_from(reference),
            SavedQueryItem::Dependent { reference, .. } => {
                record.dependents.push(reference.clone())
            }
        }
    }
    record
}
fn resaves(page: &ReadPage) -> Vec<&ReadyCall> {
    history_items(page)
        .flat_map(|item| match item {
            SavedQueryItem::Dependent { resave, .. } => resave.as_slice(),
            _ => &[],
        })
        .collect()
}
fn data_items(page: &ReadPage) -> Vec<ReadItem> {
    page.document.items.iter().filter(|item| !matches!(&item.kind, ItemKind::Card { card } if matches!(card.as_ref(), Card::SavedQuery { .. }))).cloned().collect()
}

#[test]
fn saved_query_read_lists_dependents_with_executable_resave_lines() {
    let mut d = test_daemon::start("saved question reader");
    d.pause_signals();
    save(&d.daemon, "focus-first", "focus", "sessions", true);
    assert!(matches!(
        crate::test_answers::step(
            Call::new(
                "rows",
                json!({
                    "selection":"query:focus | count",
                    "steps":[{"verb":"save","name":"summary","denominator":"sessions"}]
                })
            )
            .id("dependent-first")
            .answer(&d.daemon)
        ),
        crate::protocol::line::StepAnswer::Saved(_)
    ));
    save(&d.daemon, "focus-second", "focus", "messages", false);
    let before = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .saved_query_record("focus")
        .unwrap()
        .unwrap();
    let first = read(&d.daemon, "query:focus");
    let saved = header(&first);
    let saved_history = history(&first);
    assert_eq!(saved_history, before);
    assert_eq!(saved_history.revisions.len(), 2);
    assert_eq!(
        saved.selected_revision,
        saved_history.revisions[1].reference
    );
    assert_eq!(saved.query_id.to_string(), before.revisions[1].hash);
    assert_eq!(saved_history.saved_rows.len(), 2);
    assert!(saved_history.watch.is_some());
    assert_eq!(saved_history.dependents.len(), 1);
    let calls = resaves(&first);
    assert_eq!(calls.len(), 1);
    let refresh = &calls[0].request;
    assert_eq!(
        refresh.arguments,
        json!({"selection":"query:focus | count","steps":[{"verb":"save","name":"summary","denominator":before.revisions[0].hash}]})
    );
    let declarations = crate::commands::declarations();
    let command = declarations.cli_command(refresh).unwrap();
    assert!(command.contains("save summary"), "{command}");
    // The real CLI parser consumes the exact generated command.
    let (words, stdin) = crate::cli::line::tests::shell_arguments(&command);
    assert!(stdin.is_empty());
    let parsed =
        crate::cli::line::parse(&words, &crate::cli::line::SavedNames::default(), false).unwrap();
    assert_eq!(
        parsed.selection,
        Some(crate::cli::line::SelectionWord::Text(
            "query:focus | count".into()
        ))
    );
    assert_eq!(parsed.steps.len(), 1);
    assert_eq!(
        parsed.steps[0].step["denominator"],
        before.revisions[0].hash
    );
    for format in [Format::Terminal, Format::Markdown, Format::Html] {
        let mut bytes = Vec::new();
        polylogue_render::encode_document(
            &first.document,
            format,
            polylogue_render::EncodeOptions {
                commands: Some(declarations.as_ref()),
                ..Default::default()
            },
            &mut bytes,
        )
        .unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("Saved query: focus"), "{text}");
        assert!(text.contains("Revision 1:"));
        assert!(text.contains("Kept:"));
        assert!(text.contains("Watch:"));
        assert!(text.contains("Dependent:"));
        assert!(text.contains("save summary"));
    }
    d.daemon.writes.as_ref().unwrap().flush_rows().unwrap();
    // A read may queue its run. It never revises a definition or keeps rows.
    let after = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .saved_query_record("focus")
        .unwrap()
        .unwrap();
    assert_eq!(after.revisions, before.revisions);
    assert_eq!(after.saved_rows, before.saved_rows);
    assert_eq!(after.watch, before.watch);
    let old = read(&d.daemon, "query:focus@1");
    assert_eq!(
        header(&old).selected_revision,
        before.revisions[0].reference
    );
    assert_eq!(data_items(&old), read(&d.daemon, "sessions").document.items);
    assert_eq!(
        data_items(&first),
        read(&d.daemon, "messages").document.items
    );
    d.restart();
    d.pause_signals();
    let repeated = read(&d.daemon, "query:focus");
    let repeated_history = history(&repeated);
    assert_eq!(repeated_history.revisions, saved_history.revisions);
    assert_eq!(repeated_history.saved_rows, saved_history.saved_rows);
    assert_eq!(repeated_history.watch, saved_history.watch);
    assert_eq!(header(&repeated).selected_revision, saved.selected_revision);
    assert_eq!(data_items(&repeated), data_items(&first));
    let answer = crate::test_answers::step(Call::from_request(refresh).answer(&d.daemon));
    assert!(matches!(
        answer,
        crate::protocol::line::StepAnswer::Saved(_)
    ));
    let refreshed = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .saved_query_record("summary")
        .unwrap()
        .unwrap();
    assert_eq!(refreshed.revisions.len(), 2);
    assert_ne!(refreshed.revisions[0].hash, refreshed.revisions[1].hash);
    crate::test_answers::step(Call::new("rows", json!({"selection":"annotations where kind:saved_query key:summary","steps":[{"verb":"withdraw"}]})).answer(&d.daemon));
    // Withdrawing one exact revision reveals its predecessor.
    let previous_dependent = read(&d.daemon, "query:focus");
    assert_eq!(resaves(&previous_dependent).len(), 1);
    let polylogue_model::Reference::Annotation(mut family) =
        polylogue_model::Reference::parse(&refreshed.revisions[0].reference).unwrap()
    else {
        panic!("saved definition is an annotation")
    };
    family.revision = None;
    crate::test_answers::step(Call::new("rows", json!({"selection":polylogue_model::Reference::Annotation(family).to_string(),"steps":[{"verb":"withdraw"}]})).answer(&d.daemon));
    let without_dependent = read(&d.daemon, "query:focus");
    assert!(resaves(&without_dependent).is_empty());
    assert!(!history(&without_dependent).dependents.is_empty());
}

#[test]
fn saved_query_reads_keep_missing_withdrawn_and_historical_names_distinct() {
    let mut d = test_daemon::start("saved query states");
    d.pause_signals();
    save(&d.daemon, "saved-state", "focus", "sessions", false);
    crate::test_answers::step(Call::new("rows", json!({"selection":"annotations where kind:saved_query key:focus","steps":[{"verb":"withdraw"}]})).answer(&d.daemon));
    for (target, name) in [("query:focus", "focus"), ("query:absent", "absent")] {
        let failure = crate::test_answers::refusal(
            Call::new(
                "rows",
                json!({"selection":target,"steps":[{"verb":"read"}]}),
            )
            .answer(&d.daemon),
        );
        assert!(failure.message.contains(name), "{failure:?}");
    }
    assert_eq!(
        history(&read(&d.daemon, "query:focus@1")).revisions.len(),
        1
    );
    save(&d.daemon, "recent-override", "recent", "messages", false);
    assert!(matches!(
        read(&d.daemon, "query:recent").document.header.kind,
        HeaderKind::SavedQuery(_)
    ));
    crate::test_answers::step(Call::new("rows", json!({"selection":"annotations where kind:saved_query key:recent","steps":[{"verb":"withdraw"}]})).answer(&d.daemon));
    let shipped = read(&d.daemon, "query:recent");
    assert!(!matches!(
        shipped.document.header.kind,
        HeaderKind::SavedQuery(_)
    ));
}

mod pages;
