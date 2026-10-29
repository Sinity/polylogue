use super::*;
use polylogue_model::LogicalDay;
use serde_json::json;

fn options() -> Options {
    Options {
        clock: polylogue_query::time::clock(
            1_791_367_200_000,
            LogicalDay::named("Europe/Warsaw", 240).unwrap(),
        )
        .unwrap(),
        origins: vec!["claude-code".into()],
        command_words: vec![],
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
    let program = polylogue_query::compile(text, &options).unwrap();
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
    assert_eq!(opened.context.resolved_times, program.resolved_times);
    assert_eq!(
        check(&opened, &program, &options, false).unwrap().len(),
        program.order.len()
    );
    let other = identity("since:3d | sort by id", &options).unwrap();
    assert_eq!(
        open(&token, &seal, other, &position(), (1, 2), &options)
            .unwrap_err()
            .kind,
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
        .kind,
        "invalid_query"
    );
}

#[test]
fn authenticated_retired_context_and_changed_definitions_refuse_continuation() {
    let options = options();
    let program = polylogue_query::compile("sessions", &options).unwrap();
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
            .kind,
        "stale_cursor"
    );
    let mut changed = program.clone();
    changed.definitions.push(polylogue_query::Definition {
        name: "saved".into(),
        revision: "new-revision".into(),
    });
    assert_eq!(
        check(&continuation, &changed, &options, false)
            .unwrap_err()
            .kind,
        "stale_cursor"
    );
    let mut changed = options.clone();
    changed.fields = vec!["title".into()];
    assert_eq!(
        check(&continuation, &program, &changed, false)
            .unwrap_err()
            .kind,
        "invalid_query"
    );
}
