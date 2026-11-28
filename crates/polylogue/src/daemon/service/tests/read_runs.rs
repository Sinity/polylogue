use super::*;
use crate::test_daemon::{self, Call};
use polylogue_model::{
    Reference, RunId,
    query_objects::{QueryRun, QueryRunOutcome, RunBinding, RunOutcome},
    read::*,
};
use serde_json::json;

fn captured(d: &Daemon, selection: &str) -> (QueryRun, QueryRunOutcome) {
    crate::test_answers::read(
        Call::new(
            "rows",
            json!({
                "selection":selection,"steps":[{"verb":"read","budget":100_000}]
            }),
        )
        .answer(d),
    );
    d.writes.as_ref().unwrap().flush_rows().unwrap();
    let reply =
        crate::test_answers::listing(Call::new("rows", json!({"selection":"runs"})).answer(d));
    let value = reply
        .value()
        .unwrap_or_else(|| panic!("recorded runs: {reply:?}"));
    assert_eq!(value["count"], 1, "one actual query read: {value}");
    let Reference::Run(id) = Reference::parse(value["rows"][0]["id"].as_str().unwrap()).unwrap()
    else {
        panic!("recorded run")
    };
    let reader = d.reads.reader().unwrap();
    (
        reader.query_run(&id).unwrap().unwrap(),
        reader.query_run_outcome(&id).unwrap().unwrap(),
    )
}

fn call(run: &QueryRun, view: &str, budget: u64) -> Call {
    Call::new(
        "rows",
        json!({"selection":Reference::Run(run.id.clone()).to_string(),"steps":[{"verb":"read","view":view,"budget":budget}]}),
    )
}

fn only_card(page: &ReadPage) -> &Card {
    let [
        ReadItem {
            kind: ItemKind::Card { card },
            ..
        },
    ] = page.document.items.as_slice()
    else {
        panic!("one run card: {:?}", page.document.items)
    };
    card
}

#[test]
fn waiting_on_run_history_records_once_and_wakes_for_an_independent_run() {
    use polylogue_model::{
        Author,
        authorship::{Authorship, Channel},
    };
    use polylogue_store::{read::limit::ReadLimit, selection::SelectionError};
    use std::time::{Duration, Instant};

    let mut d = test_daemon::start("waiting on query history");
    d.pause_signals();
    let daemon = &d.daemon;
    let limits = std::array::from_fn::<_, 2, _>(|_| {
        ReadLimit::new(Duration::from_secs(20), Default::default())
    });
    let day = query_language::time::DaySetting::new(
        jiff::tz::TimeZone::UTC,
        jiff::civil::Time::constant(0, 0, 0, 0),
    )
    .unwrap();
    let options = polylogue_query::Options::fixture(
        polylogue_query::Clock::new(jiff::Timestamp::now(), day),
        vec![],
    );
    let program = polylogue_store::query::compile("runs", &options).unwrap();
    let number_of_runs = || {
        let reader = daemon.reads.reader().unwrap();
        let mut count = 0;
        reader
            .visit_query_values::<SelectionError>(&program, None, |_| {
                count += 1;
                Ok(std::ops::ControlFlow::Continue(()))
            })
            .unwrap();
        count
    };
    std::thread::scope(|scope| {
        let (send, received) = std::sync::mpsc::channel();
        for limit in &limits {
            let send = send.clone();
            let wait = Call::new(
                "rows",
                json!({"selection":"runs where author:operator AND outcome:ok","wait":"1m"}),
            )
            .within(limit);
            scope.spawn(move || send.send(wait.answer(daemon)).unwrap());
        }
        let ready = Instant::now() + Duration::from_secs(7);
        while limits
            .iter()
            .any(|limit| limit.stage() != Some("listing wait"))
            && Instant::now() < ready
        {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(
            limits
                .iter()
                .all(|limit| limit.stage() == Some("listing wait"))
        );
        assert_eq!(
            number_of_runs(),
            2,
            "each waiting request has one beginning"
        );

        let independent = crate::test_answers::listing(
            Call::new("rows", json!({"selection":"sessions"}))
                .authorship(Authorship {
                    author: Author::Operator,
                    via: Channel::Cli,
                })
                .answer(daemon),
        );
        let independent = independent.value().unwrap()["resolved"]["run"].clone();
        let mut waits = Vec::new();
        for _ in 0..2 {
            let reply = received.recv_timeout(Duration::from_secs(6));
            if reply.is_err() {
                for limit in &limits {
                    limit.cancellation.cancel();
                }
            }
            let wait =
                crate::test_answers::listing(reply.expect("the independent run wakes both waits"));
            let wait = wait.value().unwrap();
            assert_eq!(wait["rows"][0]["id"], independent);
            let Reference::Run(id) =
                Reference::parse(wait["resolved"]["run"].as_str().unwrap()).unwrap()
            else {
                panic!("a waiting query's run identity");
            };
            let reader = daemon.reads.reader().unwrap();
            let recorded = reader.query_run_outcome(&id).unwrap().unwrap();
            assert_eq!(recorded.outcome, RunOutcome::Ok);
            assert_eq!(recorded.count, Some(1));
            waits.push(id);
        }
        assert_ne!(waits[0], waits[1]);
        assert_eq!(number_of_runs(), 3, "rechecks have no run records");
    });
}

#[test]
fn run_card_reads_the_selected_record_and_offers_an_executable_question() {
    let mut d = test_daemon::start("run reader fixture");
    d.pause_signals();
    let (run, outcome) = captured(&d.daemon, "sessions");
    let request = call(&run, "transcript", 100_000).id("selected-run-read");
    let page = crate::test_answers::read(request.clone().answer(&d.daemon));
    assert_eq!(
        page.document.header.target,
        Reference::Run(run.id.clone()).to_string()
    );
    assert_eq!(
        page.document.items[0].source.reference,
        page.document.header.target
    );
    let Card::Run {
        query_id: question,
        read_position,
        author,
        count,
        exact,
        outcome: shown,
    } = only_card(&page)
    else {
        panic!("run card")
    };
    assert_eq!(
        question.as_ref().and_then(Reference::question_id),
        run.query_id
    );
    assert_eq!(read_position, &run.read_position);
    assert_eq!(author, &run.authorship.author);
    assert_eq!(*count, outcome.count);
    assert_eq!(*exact, Some(outcome.exact));
    assert_eq!(shown.as_ref(), Some(&outcome.outcome));
    let next = page.document.footer.next.as_ref().unwrap();
    assert_eq!(next.request.operation, "rows");
    assert_eq!(
        next.request.arguments,
        json!({"selection":format!("query:{}",run.query_id.unwrap())})
    );
    let rerun = crate::test_answers::listing(Call::from_request(&next.request).answer(&d.daemon));
    assert_eq!(rerun.value().unwrap()["count"], 1);
    assert_eq!(rerun.value().unwrap()["unit"], "sessions");
    let Reference::Run(read_id) = Reference::parse(&page.snapshot.query).unwrap() else {
        panic!("read's own run")
    };
    {
        let reader = d.daemon.reads.reader().unwrap();
        let read = reader.query_run(&read_id).unwrap().unwrap();
        assert_eq!(read.query_id, None);
        assert!(
            matches!(read.binding, RunBinding::OriginalObject {execution} if execution.reference == Reference::Run(run.id.clone()))
        );
        assert_eq!(reader.query_run(&run.id).unwrap(), Some(run.clone()));
        assert_eq!(
            reader.query_run_outcome(&run.id).unwrap(),
            Some(outcome.clone())
        );
    }
    let raw = crate::test_answers::read(call(&run, "raw", 100_000).answer(&d.daemon));
    let [
        ReadItem {
            kind: ItemKind::Other { type_name, data },
            ..
        },
    ] = raw.document.items.as_slice()
    else {
        panic!("raw run")
    };
    assert_eq!(type_name, "run");
    assert_eq!(*data, json!({"run":run,"outcome":outcome}));
    // The stored field names are the format, not whatever serde derives.
    assert_eq!(data["run"]["id"], run.id.to_string());
    assert_eq!(data["outcome"]["run"], data["run"]["id"]);
    assert!(raw.document.footer.next.is_none());
    d.restart();
    d.pause_signals();
    let repeated = crate::test_answers::read(request.answer(&d.daemon));
    assert_eq!(repeated.document.items, page.document.items);
}

#[test]
fn run_cards_keep_pending_missing_and_incomplete_states_distinct() {
    let mut d = test_daemon::start("run states");
    d.pause_signals();
    let (mut run, mut outcome) = captured(&d.daemon, "sessions");
    run.id = RunId::parse("018f0000-0000-7000-8000-000000000021").unwrap();
    let missing = crate::test_answers::refusal(call(&run, "transcript", 100_000).answer(&d.daemon));
    assert_eq!(missing.code.as_deref(), Some("not_found"));
    d.daemon
        .writes
        .as_ref()
        .unwrap()
        .archive()
        .begin_query_run(run.clone(), None)
        .unwrap();
    let pending = crate::test_answers::read(call(&run, "transcript", 100_000).answer(&d.daemon));
    assert!(matches!(
        only_card(&pending),
        Card::Run {
            count: None,
            exact: None,
            outcome: None,
            ..
        }
    ));
    run.id = RunId::parse("018f0000-0000-7000-8000-000000000022").unwrap();
    outcome.run = run.id.clone();
    outcome.run_hash = run.record_hash().unwrap();
    d.daemon
        .writes
        .as_ref()
        .unwrap()
        .archive()
        .finish_query_run(outcome)
        .unwrap()
        .recv()
        .unwrap()
        .unwrap();
    let incomplete =
        crate::test_answers::refusal(call(&run, "transcript", 100_000).answer(&d.daemon));
    assert_eq!(incomplete.code.as_deref(), Some("not_yet_available"));
    assert!(incomplete.message.contains("beginning is unavailable"));
}

#[test]
fn run_card_suppresses_an_unresolved_question_until_its_object_arrives() {
    let mut d = test_daemon::start("missing question object");
    d.pause_signals();
    let (mut run, _) = captured(&d.daemon, "sessions");
    let object = polylogue_query::plan::prepare("messages", &Default::default(), |_| Ok(()))
        .unwrap()
        .query_object()
        .unwrap();
    run.id = RunId::parse("018f0000-0000-7000-8000-000000000023").unwrap();
    run.query_id = Some(object.id);
    let RunBinding::Original { execution } = &mut run.binding else {
        panic!("query execution")
    };
    execution.replay.context.question = object.id;
    execution.replay.bound_ast = object.query.clone();
    execution.replay.source_outputs = object.source_outputs.clone();
    execution.replay.original_text = "messages".into();
    execution.replay.definitions.clear();
    d.daemon
        .writes
        .as_ref()
        .unwrap()
        .archive()
        .begin_query_run(run.clone(), None)
        .unwrap();
    let unavailable =
        crate::test_answers::read(call(&run, "transcript", 100_000).answer(&d.daemon));
    assert!(matches!(
        only_card(&unavailable),
        Card::Run {
            query_id: Some(_),
            ..
        }
    ));
    assert!(unavailable.document.footer.next.is_none());
    d.daemon
        .writes
        .as_ref()
        .unwrap()
        .archive()
        .query_object(object, None)
        .unwrap();
    let available = crate::test_answers::read(call(&run, "transcript", 100_000).answer(&d.daemon));
    let next = available.document.footer.next.as_ref().unwrap();
    let result = crate::test_answers::listing(Call::from_request(&next.request).answer(&d.daemon));
    assert_eq!(result.value().unwrap()["count"], 1);
}

#[test]
fn raw_run_pages_replay_after_restart_and_reject_changed_completion() {
    let mut d = test_daemon::start("raw run paging");
    d.pause_signals();
    let selection = format!(
        "sessions where title:{} OR kind:*",
        serde_json::to_string(&"neutral ".repeat(256)).unwrap()
    );
    let (mut run, _) = captured(&d.daemon, &selection);
    run.id = RunId::parse("018f0000-0000-7000-8000-000000000024").unwrap();
    d.daemon
        .writes
        .as_ref()
        .unwrap()
        .archive()
        .begin_query_run(run.clone(), None)
        .unwrap();
    let first = crate::test_answers::read(call(&run, "raw", 512).answer(&d.daemon));
    let next = first
        .document
        .footer
        .next
        .as_ref()
        .expect("raw object continues")
        .request
        .clone();
    let before = crate::test_answers::read(Call::from_request(&next).answer(&d.daemon));
    d.restart();
    d.pause_signals();
    let after = crate::test_answers::read(Call::from_request(&next).answer(&d.daemon));
    let normalize = |page: ReadPage| {
        page.document
            .items
            .into_iter()
            .map(|mut item| {
                if matches!(item.kind, ItemKind::Fragment { .. }) {
                    item.source = Source::record("read-operation");
                }
                item
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(normalize(before), normalize(after));
    let completion = QueryRunOutcome {
        run: run.id.clone(),
        binding_hash: run.binding_hash().unwrap(),
        run_hash: run.record_hash().unwrap(),
        at: run.at,
        outcome: RunOutcome::Ok,
        count: Some(0),
        exact: true,
        next: None,
    };
    d.daemon
        .writes
        .as_ref()
        .unwrap()
        .archive()
        .finish_query_run(completion)
        .unwrap()
        .recv()
        .unwrap()
        .unwrap();
    let changed = crate::test_answers::refusal(Call::from_request(&next).answer(&d.daemon));
    assert_eq!(changed.code.as_deref(), Some("stale_cursor"));
}

#[test]
fn object_read_runs_have_no_invented_question_or_followup() {
    use polylogue_model::{
        Author,
        annotation::{AnnotationKind, LinkRole, NewAnnotation, NewAnnotationLink},
        authorship::{Authorship, Channel},
        user::UserWrite,
    };
    let mut d = test_daemon::start("object read run");
    d.pause_signals();
    let actor = Authorship {
        author: Author::Operator,
        via: Channel::Cli,
    };
    let request = d.daemon.invocation(&actor).request(
        "no-query-object".into(),
        UserWrite::Assert {
            annotations: vec![NewAnnotation {
                kind: AnnotationKind::Note,
                key: None,
                value: Some(json!("synthetic note")),
                valid_until: None,
                links: vec![NewAnnotationLink {
                    role: LinkRole::About,
                    reference: Reference::parse("repo:synthetic").unwrap(),
                }],
            }],
        },
    );
    d.daemon
        .writes
        .as_ref()
        .unwrap()
        .archive()
        .user(request)
        .unwrap();
    let original = crate::test_answers::read(Call::new("rows",json!({"selection":"request:no-query-object","steps":[{"verb":"read","budget":100_000}]})).answer(&d.daemon));
    assert_eq!(original.document.header.target, original.snapshot.query);
    let Reference::Run(id) = Reference::parse(&original.snapshot.query).unwrap() else {
        panic!("object run")
    };
    let run = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .query_run(&id)
        .unwrap()
        .unwrap();
    assert_eq!(run.query_id, None);
    let page = crate::test_answers::read(call(&run, "transcript", 100_000).answer(&d.daemon));
    assert!(matches!(
        only_card(&page),
        Card::Run {
            query_id: None,
            count: Some(1),
            outcome: Some(RunOutcome::Ok),
            ..
        }
    ));
    assert!(page.document.footer.next.is_none());
}

#[test]
fn a_listing_continuation_reuses_its_durable_original_run() {
    let mut daemon = test_daemon::start_over_morrow(&|_| {});
    daemon.pause_signals();
    let save = |d: &Daemon, question: &str, request: &str| {
        crate::test_answers::step(
            Call::new(
                "rows",
                json!({"selection":question,"all":true,"steps":[{"verb":"save","name":"paging"}]}),
            )
            .id(request)
            .answer(d),
        )
    };
    save(
        &daemon.daemon,
        "sessions | sort by id",
        "save-paging-original",
    );
    let listing = |selection: &str, cursor: Option<&str>| {
        Call::new("rows", json!({"selection":selection, "page_size":1}))
            .authorship(polylogue_model::authorship::Authorship {
                author: polylogue_model::Author::Operator,
                via: polylogue_model::authorship::Channel::Cli,
            })
            .cursor(cursor)
            .answer(&daemon.daemon)
    };
    let first = crate::test_answers::listing(listing("query:paging", None));
    let first = first
        .value()
        .unwrap_or_else(|| panic!("first page: {first:?}"));
    let run_ref = first["resolved"]["run"].as_str().expect("run reference");
    let Reference::Run(id) = Reference::parse(run_ref).unwrap() else {
        panic!("expected a run reference");
    };
    let next = first["next"].as_str().expect("more than one session");
    save(
        &daemon.daemon,
        "messages | sort by id",
        "save-paging-changed",
    );
    let second = crate::test_answers::listing(listing("query:paging", Some(next)));
    let second = second
        .value()
        .unwrap_or_else(|| panic!("second page: {second:?}"));
    assert_eq!(second["resolved"]["run"], run_ref);
    assert_eq!(second["unit"], "sessions");
    assert_eq!(second["resolved"]["query"], first["resolved"]["query"]);
    assert_eq!(second["count"], first["count"]);
    assert_ne!(second["rows"], first["rows"]);
    let count_before_restart = {
        let read = daemon.daemon.reads.reader().unwrap();
        let run = read.query_run(&id).unwrap().unwrap();
        assert_eq!(run.authorship.author, polylogue_model::Author::Operator);
        assert_eq!(
            run.authorship.via,
            polylogue_model::authorship::Channel::Cli
        );
        assert_eq!(
            read.query_run_outcome(&id).unwrap().unwrap().outcome,
            RunOutcome::Ok
        );
        // Both pages use the same immutable run, with a completed first-page outcome.
        let completion = read.query_run_outcome(&id).unwrap().unwrap();
        assert_eq!(
            completion.next.unwrap().request.cursor.as_deref(),
            Some(next)
        );
        completion.count
    };
    assert_eq!(count_before_restart, first["count"].as_u64());
    daemon.restart();
    let read = daemon.daemon.reads.reader().unwrap();
    assert_eq!(
        read.query_run_outcome(&id).unwrap().unwrap().count,
        count_before_restart
    );
}

#[test]
fn saving_an_executed_query_records_one_run_and_copying_its_result_does_not() {
    let mut daemon = test_daemon::start("Save this session");
    daemon.pause_signals();
    let step = crate::test_answers::step(
        Call::new(
            "rows",
            json!({"selection":"sessions","steps":[{"verb":"save","name":"original"}]}),
        )
        .id("record-the-save")
        .answer(&daemon.daemon),
    );
    let crate::protocol::line::StepAnswer::Saved(receipt) = step else {
        panic!("saved query receipt");
    };
    let result = receipt.saved_rows.result;
    crate::test_answers::step(
        Call::new(
            "rows",
            json!({"selection":result,"steps":[{"verb":"save","name":"copy"}]}),
        )
        .id("copy-the-saved-rows")
        .answer(&daemon.daemon),
    );
    let runs = crate::test_answers::listing(
        Call::new("rows", json!({"selection":"runs"})).answer(&daemon.daemon),
    );
    let runs = runs
        .value()
        .unwrap_or_else(|| panic!("run history: {runs:?}"));
    assert_eq!(runs["count"], 1, "one evaluation of the original question");
    let Reference::Run(id) = Reference::parse(runs["rows"][0]["id"].as_str().unwrap()).unwrap()
    else {
        panic!("recorded Save evaluation");
    };
    let reader = daemon.daemon.reads.reader().unwrap();
    let run = reader.query_run(&id).unwrap().unwrap();
    let request = reader.user_request("record-the-save-1").unwrap().unwrap();
    let polylogue_model::user::UserWrite::Save(save) = request.write else {
        panic!("durable Save request");
    };
    let kept: serde_json::Value = serde_json::from_str(&save.rows_json).unwrap();
    assert_eq!(kept["run"], Reference::Run(id.clone()).to_string());
    let binding = run.binding.original().unwrap();
    assert_eq!(
        binding.accepted_request.as_ref().unwrap()["arguments"]["steps"][0]["verb"],
        "save"
    );
    assert_eq!(
        reader.query_run_outcome(&id).unwrap().unwrap().count,
        Some(1)
    );
}
