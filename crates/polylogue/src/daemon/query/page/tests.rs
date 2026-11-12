use super::*;
use serde_json::json;

fn options() -> Options {
    Options {
        clock: query_language::time::Clock::new(
            jiff::Timestamp::from_millisecond(1_791_367_200_000).unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::get("Europe/Warsaw").unwrap(),
                jiff::civil::Time::constant(4, 0, 0, 0),
            )
            .unwrap(),
        ),
        origins: vec!["claude-code".into()],
        page_size: Some(1),
        random_seed: Some(9),
        embeddings: false,
        self_session: None,
        saved: Default::default(),
        fields: vec![],
        kept: Default::default(),
        chat_pages: vec![],
    }
}
fn position() -> ChangePosition {
    ChangePosition {
        generation: "synthetic-generation".into(),
        index: 3,
        archive: 2,
    }
}

#[test]
fn continuation_authenticates_question_clock_seed_and_read_context() {
    let options = options();
    let text = "since:7d | sort by id";
    let program = polylogue_store::query::compile(text, &options).unwrap();
    assert_eq!(program.resolved_times.len(), 1);
    let question = identity(text, &options).unwrap();
    let seal = CursorSeal::new([7; 32]);
    let context = context(question, &position(), (1, 2), &options, &program, 9, false).unwrap();
    let row: serde_json::Map<String, Value> = program
        .order
        .iter()
        .enumerate()
        .map(|(i, key)| (key.column.clone(), json!(i)))
        .collect();
    let token = after_row(
        &program,
        &Value::Object(row),
        &Paging {
            seal: &seal,
            context,
        },
    )
    .unwrap();
    let opened = open(&token, &seal, question, &position(), (1, 2), &options).unwrap();
    assert_eq!(opened.context.clock, options.clock);
    assert_eq!(opened.context.random_seed, 9);
    assert!(opened.context.resolved_times.is_empty());
    polylogue_store::query::continuation::check_resolved_times(
        &opened.context,
        &program.resolved_times,
    )
    .unwrap();
    assert_eq!(
        check(&opened, &program, &options, false).unwrap().len(),
        program.order.len()
    );
    let other = identity("since:3d | sort by id", &options).unwrap();
    assert_eq!(
        open(&token, &seal, other, &position(), (1, 2), &options)
            .unwrap_err()
            .code(),
        "invalid_query"
    );
    let mut tampered = token.into_bytes();
    let last = tampered.len() - 1;
    tampered[last] = if tampered[last] == b'0' { b'1' } else { b'0' };
    assert_eq!(
        open(
            std::str::from_utf8(&tampered).unwrap(),
            &seal,
            question,
            &position(),
            (1, 2),
            &options
        )
        .unwrap_err()
        .code(),
        "invalid_query"
    );
}

#[test]
fn authenticated_retired_context_and_changed_definitions_refuse_continuation() {
    let options = options();
    let program = polylogue_store::query::compile("sessions", &options).unwrap();
    let question = identity("sessions", &options).unwrap();
    let seal = CursorSeal::new([8; 32]);
    let context = context(question, &position(), (1, 2), &options, &program, 9, false).unwrap();
    let continuation = Continuation {
        context,
        after: OrderPosition {
            values: vec![Value::Null; program.order.len()],
            row_identity: "row-a".into(),
        },
    };
    let token = seal.seal(&continuation).unwrap();
    let mut newer = position();
    newer.generation = "rebuilt-generation".into();
    assert_eq!(
        open(token.as_str(), &seal, question, &newer, (1, 2), &options)
            .unwrap_err()
            .code(),
        "stale_cursor"
    );
    // A rebuilt index has a new sealing key, so the earlier generation's
    // cursor no longer authenticates; it is still `stale_cursor`, not
    // `invalid_query` (docs/query.md, Paging, Refusals).
    let rebuilt_seal = CursorSeal::new([9; 32]);
    assert_eq!(
        open(
            token.as_str(),
            &rebuilt_seal,
            question,
            &newer,
            (1, 2),
            &options
        )
        .unwrap_err()
        .code(),
        "stale_cursor"
    );
    assert_eq!(
        open(
            token.as_str(),
            &rebuilt_seal,
            question,
            &position(),
            (1, 2),
            &options
        )
        .unwrap_err()
        .code(),
        "invalid_query",
        "a cursor naming the current generation that does not authenticate was never sealed here"
    );
    assert_eq!(
        open("broken", &rebuilt_seal, question, &newer, (1, 2), &options)
            .unwrap_err()
            .code(),
        "invalid_query"
    );
    for compiler in ["shared-query-v1", "shared-query-v2"] {
        let mut retired = continuation.clone();
        retired
            .context
            .versions
            .insert("compiler".into(), compiler.into());
        assert_eq!(
            check(&retired, &program, &options, false)
                .unwrap_err()
                .code(),
            "stale_cursor"
        );
    }
    let mut changed = program.clone();
    changed.definitions.push(polylogue_query::Definition {
        name: "saved".into(),
        revision: "new-revision".into(),
    });
    assert_eq!(
        check(&continuation, &changed, &options, false)
            .unwrap_err()
            .code(),
        "stale_cursor"
    );
    let mut changed = options.clone();
    changed.fields = vec!["title".into()];
    assert_eq!(
        check(&continuation, &program, &changed, false)
            .unwrap_err()
            .code(),
        "invalid_query"
    );
}

#[test]
fn definition_order_does_not_retire_the_same_semantic_question() {
    let options = options();
    let mut program = polylogue_store::query::compile("sessions", &options).unwrap();
    program.definitions = ["a", "b"]
        .map(|name| polylogue_query::Definition {
            name: name.into(),
            revision: format!("{name}-revision"),
        })
        .to_vec();
    let continuation = Continuation {
        context: context(
            identity("sessions", &options).unwrap(),
            &position(),
            (1, 2),
            &options,
            &program,
            9,
            false,
        )
        .unwrap(),
        after: OrderPosition {
            values: vec![Value::Null; program.order.len()],
            row_identity: "row-a".into(),
        },
    };
    program.definitions.reverse();
    check(&continuation, &program, &options, false).unwrap();
}

#[test]
fn resolution_vector_size_does_not_grow_the_authenticated_continuation() {
    let options = options();
    let mut program = polylogue_store::query::compile("since:7d | sort by id", &options).unwrap();
    let time = program.resolved_times[0].clone();
    program.resolved_times = vec![time; 2000];
    let question = identity("since:7d | sort by id", &options).unwrap();
    let context = context(question, &position(), (1, 2), &options, &program, 9, false).unwrap();
    let continuation = Continuation {
        context,
        after: OrderPosition {
            values: vec![Value::Null; program.order.len()],
            row_identity: "synthetic-row".into(),
        },
    };
    let seal = CursorSeal::new([7; 32]);
    let token = seal.seal(&continuation).unwrap();
    assert!(token.as_str().len() < 4096);
    check(&continuation, &program, &options, false).unwrap();
    program.resolved_times[1].expression = "another expression".into();
    assert_eq!(
        check(&continuation, &program, &options, false)
            .unwrap_err()
            .code(),
        "stale_cursor"
    );
}
