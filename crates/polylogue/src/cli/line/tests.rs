use super::*;
fn parsed(words: &[&str]) -> ParsedLine {
    parse(
        &words.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        &FirstWordCatalog::default(),
        true,
    )
    .unwrap()
}
#[test]
fn first_word_order_matches_design() {
    let catalog = FirstWordCatalog {
        saved: vec!["mine".into(), "status".into(), "read".into()],
        shipped: vec![],
    };
    for (words, rule) in [
        (vec![], 1),
        (vec!["-"], 2),
        (vec!["status"], 3),
        (vec!["read"], 4),
        (vec!["mine"], 5),
        (vec!["sessions"], 6),
        (vec!["latest"], 7),
        (vec!["word:"], 8),
        (vec!["chrono"], 9),
    ] {
        let args = words.into_iter().map(str::to_owned).collect::<Vec<_>>();
        assert_eq!(parse(&args, &catalog, true).unwrap().first.rule(), rule);
    }
}
#[test]
fn chain_parses_fixed_arity_left_to_right() {
    let line = parsed(&["sessions", "note", "read", "tag", "later", "read"]);
    assert_eq!(
        line.steps.iter().map(|s| s.verb).collect::<Vec<_>>(),
        ["note", "tag", "read"]
    );
    assert_eq!(line.steps[0].step["text"], "read");
}
#[test]
fn save_name_optional_before_verb() {
    let line = parsed(&["sessions", "save", "read"]);
    assert_eq!(line.steps.len(), 2);
    assert_eq!(line.steps[0].step["name"], Value::Null);
    assert_eq!(
        parsed(&["sessions", "save", "later", "read"]).steps[0].step["name"],
        "later"
    );
}
#[test]
fn leading_options_accept_the_equals_spelling() {
    let spaced = parsed(&[
        "--deadline",
        "5s",
        "--page-size",
        "10",
        "--fields",
        "id,at",
        "--why",
        "sessions",
    ]);
    let equals = parsed(&[
        "--deadline=5s",
        "--page-size=10",
        "--fields=id,at",
        "--why=true",
        "sessions",
    ]);
    assert_eq!(equals.line.deadline.as_deref(), Some("5s"));
    assert_eq!(equals.line.deadline, spaced.line.deadline);
    assert_eq!(equals.listing.page_size.map(|size| size.get()), Some(10));
    assert_eq!(equals.listing.fields, ["id", "at"]);
    assert!(equals.listing.why);
    assert!(!parsed(&["--why=false", "sessions"]).listing.why);
}

#[test]
fn options_belong_to_the_verb_before_them() {
    let line = parsed(&[
        "sessions", "read", "--budget", "200", "brief", "--budget", "100", "--json",
    ]);
    assert_eq!(line.steps[0].step["budget"], 200);
    assert_eq!(line.steps[1].step["budget"], 100);
    assert!(line.line.json);
}
#[test]
fn read_keeps_its_directory_and_format() {
    let line = parsed(&["sessions", "read", "--to", "out/", "--format", "jsonl"]);
    assert_eq!(line.steps[0].to, Some(Destination::Dir("out/".into())));
    let request = super::super::run_line::request(&line).unwrap();
    let rows: crate::protocol::line::Rows = serde_json::from_value(request.arguments).unwrap();
    let [crate::protocol::line::Step::Read(step)] = rows.steps.as_slice() else {
        panic!("one read")
    };
    assert_eq!(step.format.as_deref(), Some("jsonl"));
}

#[test]
fn undeclared_option_refused() {
    let error = parse(
        &[
            "sessions".into(),
            "tag".into(),
            "x".into(),
            "--follow".into(),
        ],
        &FirstWordCatalog::default(),
        true,
    )
    .unwrap_err();
    assert_eq!(error.message, "tag: unknown option --follow");
}
#[test]
fn selection_must_be_one_argument_with_corrected_line() {
    let error = parse(
        &["repo:sinex".into(), "since:7d".into(), "read".into()],
        &FirstWordCatalog::default(),
        true,
    )
    .unwrap_err();
    assert_eq!(
        error.try_line.as_deref(),
        Some("polylogue 'repo:sinex since:7d' read")
    );
    assert_eq!(error.code, Code::NotAVerb);
}
#[test]
fn literal_selection_keeps_its_marker_in_the_corrected_line() {
    let error = parse(
        &["--".into(), "foo".into(), "bar".into()],
        &FirstWordCatalog::default(),
        true,
    )
    .unwrap_err();
    assert_eq!(error.try_line.as_deref(), Some("polylogue -- 'foo bar'"));
    assert_eq!(error.code, Code::NotAVerb);
}
#[test]
fn row_numbers_refused_off_terminal() {
    let error = parse(
        &["@1".into(), "read".into()],
        &FirstWordCatalog::default(),
        false,
    )
    .unwrap_err();
    assert_eq!(error.code, Code::RowNumbersOffTerminal);
}
#[test]
fn stdin_is_read_once() {
    let error = parse(
        &["-".into(), "note".into(), "-".into()],
        &FirstWordCatalog::default(),
        false,
    )
    .unwrap_err();
    assert_eq!(error.message, "standard input can be read once per line");
}
#[test]
fn literal_verb_words_and_option_words_stay_text() {
    for text in ["read", "--hidden"] {
        let line = parsed(&["sessions", "edit", "--", text, "read"]);
        assert_eq!(line.steps[0].step["text"], text);
        assert_eq!(line.steps[1].verb, "read");
    }
}
#[test]
fn literal_selection_and_stdin_are_distinct() {
    for text in ["status", "read", "-", "--"] {
        let line = parsed(&["--", text, "read"]);
        assert_eq!(line.selection, Some(SelectionWord::Text(text.into())));
        assert_eq!(line.first, FirstWord::QueryText);
    }
    let line = parsed(&["-", "note", "--", "-", "read"]);
    assert_eq!(line.steps[0].step["text"], "-");
    assert_eq!(line.steps[0].stdin, None);
    let line = parsed(&["sessions", "edit", "--", "--", "read", "--budget", "100"]);
    assert_eq!(line.steps[0].step["text"], "--");
    assert_eq!(line.steps[1].step["budget"], 100);
}

#[test]
fn stdin_intent_is_consumed_once_and_literal_hyphens_do_not_read_it() {
    let mut line = parsed(&["-", "note", "--", "-"]);
    let mut input = std::io::Cursor::new(b"session:codex:first\nsession:codex:second\n".as_slice());
    super::super::run_line::read_stdin(&mut line, &mut input).unwrap();
    assert_eq!(
        line.selection,
        Some(SelectionWord::References(vec![
            "session:codex:first".into(),
            "session:codex:second".into()
        ]))
    );
    assert_eq!(line.steps[0].step["text"], "-");
    let position = input.position();
    super::super::run_line::read_stdin(&mut line, &mut input).unwrap();
    assert_eq!(input.position(), position);
    let mut literal = parsed(&["--", "-", "note", "--", "-"]);
    let mut untouched = std::io::Cursor::new(b"untouched".as_slice());
    super::super::run_line::read_stdin(&mut literal, &mut untouched).unwrap();
    assert_eq!(untouched.position(), 0);
    assert_eq!(literal.selection, Some(SelectionWord::Text("-".into())));
}

pub(crate) fn shell_arguments(command: &str) -> (Vec<String>, Vec<u8>) {
    let script = format!(
        r#"polylogue() {{ printf '%s\0' "$@"; printf '\0'; cat; }}
{command}"#
    );
    let output = std::process::Command::new("sh")
        .args(["-c", &script])
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let boundary = output
        .stdout
        .windows(2)
        .position(|bytes| bytes == [0, 0])
        .unwrap();
    let words = if boundary == 0 {
        vec![]
    } else {
        output.stdout[..boundary]
            .split(|byte| *byte == 0)
            .map(|word| String::from_utf8(word.to_vec()).unwrap())
            .collect()
    };
    (words, output.stdout[boundary + 2..].to_vec())
}

fn formatted(request: &command_surface::RunnableRequest) -> command_surface::RunnableRequest {
    let command = crate::commands::declarations()
        .cli_command(request)
        .unwrap();
    let (words, bytes) = shell_arguments(&command);
    let mut line = parse(
        &words,
        &FirstWordCatalog {
            saved: vec!["mine".into()],
            shipped: vec![],
        },
        false,
    )
    .unwrap();
    let mut input = std::io::Cursor::new(bytes.as_slice());
    super::super::run_line::read_stdin(&mut line, &mut input).unwrap();
    let actual = super::super::run_line::request(&line).unwrap();
    let expected_rows: crate::protocol::line::Rows =
        serde_json::from_value(request.arguments.clone()).unwrap();
    let actual_rows: crate::protocol::line::Rows =
        serde_json::from_value(actual.arguments.clone()).unwrap();
    assert_eq!(actual_rows, expected_rows, "{command}");
    let mut expected = request.clone();
    let mut envelope = actual.clone();
    expected.arguments = Value::Null;
    envelope.arguments = Value::Null;
    assert_eq!(envelope, expected, "{command}");
    actual
}

#[test]
fn declared_line_format_round_trips_options_literals_references_and_deadline_units() {
    use command_surface::{Deadline, Format, TokenBudget};
    for selection in ["status", "read", "mine", "-", "--", "a quoted ' word"] {
        let mut request = crate::client::request(
            "rows",
            serde_json::json!({
                "selection":selection,"steps":[{"verb":"edit","text":"--"},{"verb":"read","cursor":"step cursor","budget":137}],"all":true,
            }),
            "quoted ' request".into(),
        );
        request.deadline = Deadline::Milliseconds(500);
        request.budget_tokens = TokenBudget::Tokens(73);
        request.cursor = Some("listing cursor".into());
        request.fresh = true;
        request.format = Format::Json;
        formatted(&request);
    }
    let mut explicit = crate::client::request(
        "rows",
        serde_json::json!({
            "references":["session:codex:first","message:codex:second"], "steps":[{"verb":"note","text":"-"}],
        }),
        "explicit-references".into(),
    );
    explicit.deadline = Deadline::None;
    explicit.budget_tokens = TokenBudget::Unlimited;
    formatted(&explicit);
}

#[test]
fn read_budget_null_zero_and_absence_round_trip_the_real_line_boundary() {
    // `--budget N|none` on brief as on read (docs/interfaces.md, Option names).
    let brief = parse(
        &["sessions", "brief", "--budget", "none"].map(str::to_owned),
        &FirstWordCatalog::default(),
        false,
    )
    .unwrap();
    assert_eq!(brief.steps[0].step["budget"], Value::Null);
    for budget in [
        None,
        Some(Value::Null),
        Some(serde_json::json!(0)),
        Some(serde_json::json!(u64::MAX)),
    ] {
        let mut step = serde_json::json!({"verb":"read", "cursor":"accepted read cursor", "show":["tools:full"]});
        if let Some(budget) = &budget {
            step["budget"] = budget.clone();
        }
        let mut request = crate::client::request(
            "rows",
            serde_json::json!({"selection":"session:test:one", "steps":[step], "all":true}),
            "read-budget-round-trip".into(),
        );
        request.deadline = command_surface::Deadline::Milliseconds(500);
        request.budget_tokens = command_surface::TokenBudget::Tokens(333);
        request.format = command_surface::Format::Json;
        let command = crate::commands::declarations()
            .cli_command(&request)
            .unwrap();
        match budget {
            Some(Value::Null) => assert!(command.contains("--budget none"), "{command}"),
            Some(Value::Number(ref n)) => {
                assert!(command.contains(&format!("--budget {n}")), "{command}")
            }
            None => assert_eq!(
                command.matches("--budget").count(),
                1,
                "only the envelope budget: {command}"
            ),
            _ => unreachable!(),
        }
        formatted(&request);
    }
}

#[test]
fn line_formatter_diagnoses_unrepresentable_step_fields() {
    let request = crate::client::request(
        "rows",
        serde_json::json!({
            "selection":"sessions", "steps":[{"verb":"read","delivery":"stream"}],
        }),
        "unsupported-line-option".into(),
    );
    let error = crate::commands::declarations()
        .cli_command(&request)
        .unwrap_err();
    assert!(
        error
            .message
            .contains("read has options unavailable on the command line"),
        "{error:?}"
    );
}

#[test]
fn generated_cli_next_and_previous_requests_reach_the_real_reader() {
    let daemon = crate::test_daemon::start(&"Synthetic λ🙂 page. ".repeat(30));
    let first = parsed(&[
        "--budget",
        "333",
        "--all",
        "session:claude-code:s1",
        "read",
        "--budget",
        "1",
        "--deadline",
        "none",
        "--json",
    ]);
    let request = super::super::run_line::request(&first).unwrap();
    let mut unsupported = request.clone();
    unsupported.fresh = true;
    let refused = crate::client::call_ready(&daemon.daemon.archive, &unsupported).unwrap();
    assert_eq!(refused.exit_code(), 2);
    assert!(
        matches!(refused.outcome, command_surface::Outcome::Error(ref error) if error.message == "polylogue does not take fresh")
    );
    let read = |request: &command_surface::RunnableRequest| {
        let request = formatted(request);
        let answer = crate::client::call_ready(&daemon.daemon.archive, &request).unwrap();
        let line: crate::protocol::line::LineAnswer =
            serde_json::from_value(crate::test_answers::ok(answer)).unwrap();
        let crate::protocol::line::LineAnswer::Steps(mut steps) = line else {
            panic!("expected read steps")
        };
        assert_eq!(steps.steps.len(), 1);
        let crate::protocol::line::StepOutcome::Done {
            reply: crate::protocol::line::StepAnswer::Read(page),
            ..
        } = steps.steps.remove(0).outcome
        else {
            panic!("expected a finished read")
        };
        page
    };
    let first = read(&request);
    let next = first
        .document
        .footer
        .next
        .as_ref()
        .expect("one-token page must continue");
    assert_eq!(
        next.request.budget_tokens,
        command_surface::TokenBudget::Tokens(333)
    );
    assert_eq!(next.request.deadline, command_surface::Deadline::None);
    assert_eq!(next.request.format, command_surface::Format::Json);
    assert_eq!(next.request.fresh, request.fresh);
    assert_eq!(next.request.arguments["all"], true);
    let second = read(&next.request);
    let previous = second
        .document
        .footer
        .previous
        .as_ref()
        .expect("second page has a predecessor");
    assert_eq!(
        previous.request.budget_tokens,
        command_surface::TokenBudget::Tokens(333)
    );
    assert_eq!(previous.request.deadline, command_surface::Deadline::None);
    assert_eq!(previous.request.format, command_surface::Format::Json);
    assert_eq!(previous.request.fresh, request.fresh);
    assert_eq!(previous.request.arguments["all"], true);
    let restored = read(&previous.request);
    assert_eq!(restored.document.items, first.document.items);
    assert_eq!(restored.document.header, first.document.header);
}

#[test]
fn every_docs_example_parses() {
    let docs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs");
    let mut verbs = std::collections::BTreeSet::new();
    for file in std::fs::read_dir(docs).unwrap() {
        let path = file.unwrap().path();
        if path.extension().and_then(|value| value.to_str()) != Some("md") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let mut fenced = false;
        let mut executable = false;
        for (line_number, line) in text.lines().enumerate() {
            if let Some(language) = line.strip_prefix("```") {
                if fenced {
                    fenced = false;
                    executable = false;
                } else {
                    fenced = true;
                    executable = matches!(language.trim(), "" | "bash" | "sh" | "shell");
                }
                continue;
            }
            if !executable || !(line == "polylogue" || line.starts_with("polylogue ")) {
                continue;
            }
            let (words, bytes) = shell_arguments(line);
            let mut parsed =
                parse(&words, &FirstWordCatalog::default(), true).unwrap_or_else(|error| {
                    panic!(
                        "{}:{}: {}: {}",
                        path.display(),
                        line_number + 1,
                        line,
                        error.message
                    )
                });
            if parsed.first == FirstWord::Command {
                let argv: Vec<_> = std::iter::once("polylogue")
                    .chain(words.iter().map(String::as_str))
                    .collect();
                crate::cli::parse(&argv).unwrap_or_else(|error| {
                    panic!("{}:{}: {error}", path.display(), line_number + 1)
                });
            } else {
                super::super::run_line::read_stdin(&mut parsed, &mut std::io::Cursor::new(bytes))
                    .unwrap();
                super::super::run_line::request(&parsed).unwrap_or_else(|error| {
                    panic!("{}:{}: {}", path.display(), line_number + 1, error.message)
                });
                verbs.extend(parsed.steps.iter().map(|step| step.verb.to_owned()));
            }
        }
    }
    assert_eq!(
        verbs,
        crate::daemon::verbs::declarations()
            .verbs()
            .iter()
            .map(|verb| verb.verb().to_owned())
            .collect()
    );
}
