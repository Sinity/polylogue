use super::*;
use serde_json::{Map, json};

/// A value of the type a schema property declares, chosen so the handler
/// receives it: the first choice of a closed set, `seen` for `--from` and
/// a built document format for `--format`.
fn sample(name: &str, property: &Value, root: &Value) -> Value {
    match name {
        "from" => return json!("seen"),
        "format" => return json!("markdown"),
        "verdict" => return json!("accept"),
        "value" => return json!("allow"),
        "as" => return json!("finding"),
        _ => {}
    }
    let property = match property
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|reference| reference.strip_prefix("#/$defs/"))
    {
        Some(definition) => &root["$defs"][definition],
        None => property,
    };
    if let Some(first) = property["enum"]
        .as_array()
        .and_then(|values| values.first())
    {
        return first.clone();
    }
    for branches in ["anyOf", "oneOf"] {
        if let Some(branch) = property[branches]
            .as_array()
            .and_then(|branches| branches.iter().find(|branch| branch["type"] != "null"))
        {
            return sample("", branch, root);
        }
    }
    let types = match &property["type"] {
        Value::String(kind) => vec![kind.as_str()],
        Value::Array(kinds) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => vec![],
    };
    if types.contains(&"boolean") {
        json!(true)
    } else if types.contains(&"integer") || types.contains(&"number") {
        json!(1)
    } else if types.contains(&"array") {
        json!([sample("", &property["items"], root)])
    } else {
        json!("x")
    }
}

/// A verb's step with its required words filled in.
fn step(verb: &command_surface::StepDeclaration) -> Map<String, Value> {
    let schema = verb.input_schema();
    let mut step = Map::new();
    step.insert("verb".into(), json!(verb.verb()));
    for required in schema["required"].as_array().into_iter().flatten() {
        let name = required.as_str().unwrap();
        step.insert(
            name.into(),
            sample(name, &schema["properties"][name], schema),
        );
    }
    step
}

fn code(failure: &Failure) -> Option<&str> {
    failure.code.as_deref()
}

/// docs/interfaces.md, Operation table: a designed option that is not built
/// has no field, and every surface that meets it answers `not_built` naming
/// its page; an option nobody designed is the ordinary unknown option.
#[test]
fn unbuilt_marks_match_refusals() {
    let declarations = crate::commands::declarations();
    let verbs = crate::daemon::verbs::declarations();
    let rows = crate::commands::command("rows").unwrap();
    let mut options = 0;
    for row in UNBUILT {
        let failure = row.failure();
        assert_eq!(code(&failure), Some("not_built"), "{row:?}");
        assert!(
            failure.message.contains(&format!("docs/{}", row.page)),
            "{}",
            failure.message
        );
        let page = row.page.split_once('#').map_or(row.page, |(file, _)| file);
        assert!(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../docs")
                .join(page)
                .is_file(),
            "{page}"
        );
        if !row.is_option() {
            assert_eq!(refusal(row.command, row.word), failure, "{row:?}");
            continue;
        }
        options += 1;
        let field = row.word.trim_start_matches("--").replace('-', "_");
        if let Some(verb) = verbs.verbs().iter().find(|verb| verb.verb() == row.command) {
            // Building the option adds its field and deletes its row.
            assert!(
                verb.input_schema()["properties"].get(&field).is_none(),
                "{row:?} is declared"
            );
            // The line parser.
            let mut words = vec!["latest".to_owned(), row.command.to_owned()];
            for positional in verb
                .arguments()
                .iter()
                .filter(|argument| argument.form == command_surface::ArgumentForm::Positional)
                .filter(|argument| step(verb).contains_key(&argument.name))
            {
                words.push(step(verb)[&positional.name].as_str().unwrap().to_owned());
            }
            words.extend([row.word.to_owned(), "x".to_owned()]);
            let refused = crate::cli::line::parse(&words, &Default::default(), false).unwrap_err();
            assert_eq!(refused.code, Code::NotBuilt, "{words:?}");
            assert_eq!(refused.message, failure.message);
            // The argument decoder, as the daemon and the MCP proxy run it.
            let mut written = step(verb);
            written.insert(field.clone(), json!(true));
            let arguments = json!({"selection": "latest", "steps": [written]});
            let refused = crate::daemon::handler::policy(
                rows,
                &arguments,
                polylogue_model::authorship::Channel::Mcp,
            )
            .unwrap_err();
            assert_eq!(refused.error, failure, "{arguments}");
        } else {
            let command = crate::commands::command(row.command)
                .unwrap_or_else(|| panic!("{row:?} names no verb or command"));
            let declared = declarations.spec_named(command.name).unwrap();
            assert!(
                declared
                    .arguments
                    .iter()
                    .all(|argument| argument.name != field),
                "{row:?} is declared"
            );
            // The command parser.
            let mut argv = vec!["polylogue".to_owned()];
            argv.extend(command.cli_words());
            argv.push(row.word.to_owned());
            let error = crate::cli::command()
                .try_get_matches_from(&argv)
                .unwrap_err();
            assert_eq!(
                crate::cli::unbuilt_option(&error, &argv[1..]).as_ref(),
                Some(&failure),
                "{argv:?}"
            );
            // The argument decoder.
            let refused = crate::daemon::handler::policy(
                command,
                &json!({ field: true }),
                polylogue_model::authorship::Channel::Mcp,
            )
            .unwrap_err();
            assert_eq!(refused.error, failure);
        }
    }
    assert!(options > 0);
    // An option nobody designed is the ordinary refusal on every surface.
    let refused = crate::cli::line::parse(
        &["latest".into(), "read".into(), "--colour".into()],
        &Default::default(),
        false,
    )
    .unwrap_err();
    assert_eq!(
        (refused.code, refused.message.as_str()),
        (Code::BadRequest, "read: unknown option --colour")
    );
    let refused = crate::daemon::handler::policy(
        rows,
        &json!({"selection": "latest", "steps": [{"verb": "read", "colour": true}]}),
        polylogue_model::authorship::Channel::Mcp,
    )
    .unwrap_err();
    assert_eq!(code(&refused.error), Some("bad_request"));
    let argv = ["polylogue", "import", "--colour"];
    let error = crate::cli::command()
        .try_get_matches_from(argv)
        .unwrap_err();
    let words: Vec<String> = argv[1..].iter().map(|word| (*word).to_owned()).collect();
    assert_eq!(crate::cli::unbuilt_option(&error, &words), None);
    // A use the list does not name is a defect, not a refusal of the design.
    assert_eq!(code(&refusal("read", "--colour")), Some("internal"));
    // A listed destination is refused by the line parser itself.
    let refused = crate::cli::line::parse(
        &[
            "latest".into(),
            "read".into(),
            "--to".into(),
            "browser".into(),
        ],
        &Default::default(),
        false,
    )
    .unwrap_err();
    assert_eq!(refused.code, Code::NotBuilt);
    assert_eq!(
        refused.message,
        listed("polylogue", "--to browser")
            .unwrap()
            .failure()
            .message
    );
}

/// docs/terminal.md, Help: `polylogue help unbuilt` prints the list, here
/// and through the daemon's `help`.
#[test]
fn help_unbuilt_prints_the_list() {
    let expected = "\
Designed and not built yet, with the page that designs each:
  read --until: docs/reading.md#ranges
  read --first: docs/reading.md#ranges
  read --last: docs/reading.md#ranges
  read --from a message or a time: docs/reading.md#ranges
  read --beside: docs/reading.md#alternates
  read --blind: docs/reading.md#compare-and-blind
  read --seed: docs/reading.md#compare-and-blind
  read --align: docs/reading.md#compare-and-blind
  read --until-idle: docs/session-control.md#following
  read --format citation: docs/page-text.md#citation-lines
  pick Ctrl-o (open in the browser): docs/terminal.md#the-picker
  pick Ctrl-y (copy the reference): docs/terminal.md#the-picker
  brief of a page or other rows: docs/agent-context.md#brief
  note --parts: docs/alternates.md
  note --check: docs/annotations.md#checked-claims
  note --expect: docs/annotations.md#checked-claims
  note --page: docs/pages.md#writing
  note --section: docs/pages.md#writing
  note --level: docs/pages.md#writing
  note --template: docs/page-templates.md
  note --param: docs/page-templates.md
  note --layout: docs/pages.md#layouts
  note --about: docs/pages.md#writing
  note --role: docs/pages.md#members
  note --brief: docs/agent-context.md#pages-here
  note --rename: docs/annotations.md#note
  note --as translation:LANG, transliteration:LANG, simplification or summary: docs/alternates.md
  note --as heading: docs/pages.md#writing
  judge --on: docs/judging.md#the-better-of-several-rows
  judge --seed: docs/judging.md#blind
  judge defer: docs/judging.md#verdicts-on-annotations
  judge over several rows: docs/judging.md#the-better-of-several-rows
  save page:NAME: docs/saved-queries.md#saving-to-a-page
  save --section: docs/saved-queries.md#saving-to-a-page
  save --after: docs/saved-queries.md#saving-to-a-page
  save --fixed: docs/saved-queries.md#saving-to-a-page
  save --live: docs/saved-queries.md#saving-to-a-page
  save --kept: docs/saved-queries.md#saving-to-a-page
  save --template: docs/saved-queries.md#saving-to-a-page
  save --as: docs/pages.md#forms
  send --page: docs/session-control.md#send
  send --section: docs/session-control.md#send
  send --attach to a harness session: docs/session-control.md#send
  answer with text: docs/session-control.md#answer
  resume --fork: docs/session-control.md#resume
  resume --anyway: docs/session-control.md#resume
  help --to: docs/terminal.md#help
  help --format: docs/terminal.md#help
  help views: docs/terminal.md#help
  help kinds: docs/terminal.md#help
  help keys: docs/terminal.md#help
  help saved: docs/terminal.md#help
  help COMMAND: docs/terminal.md#help
  start --submit in a harness session: docs/session-control.md#start
  start --model in a harness session: docs/session-control.md#start
  start --effort in a harness session: docs/session-control.md#start
  start --project in a harness session: docs/session-control.md#start
  start --attach in a harness session: docs/session-control.md#start
  import --withdraw-missing: docs/page-working-copy.md#check-in
  install HARNESS --dry-run: docs/installing.md#installing
  polylogue --to browser: docs/reading.md#formats-and-destinations
  polylogue --to clipboard: docs/reading.md#formats-and-destinations
";
    assert_eq!(text(), expected);
    let w = crate::test_daemon::start("help unbuilt");
    let answer = crate::test_answers::ok(
        crate::test_daemon::Call::new("help", json!({"topic": "unbuilt"})).answer(&w.daemon),
    );
    assert_eq!(answer, json!({"manual": expected}));
}

/// The messages of every `not_built` failure anywhere in an answer.
fn collect_unbuilt(value: &Value, refusals: &mut Vec<String>) {
    match value {
        Value::Object(object) => {
            if object.get("code") == Some(&json!("not_built")) {
                refusals.push(object["message"].as_str().unwrap_or_default().to_owned());
            }
            object
                .values()
                .for_each(|value| collect_unbuilt(value, refusals));
        }
        Value::Array(values) => values
            .iter()
            .for_each(|value| collect_unbuilt(value, refusals)),
        _ => {}
    }
}

/// docs/agents.md, The skill file: `help --all` names every verb with its
/// options, and no option or verb it names answers `not_built`. Each verb
/// runs under a dry run on the Morrow archive with one declared option at a
/// time; each command of the read tool runs with one argument at a time.
#[test]
fn help_all_names_nothing_that_refuses_as_not_built() {
    use crate::daemon::service::ArchiveMode;
    let archive = polylogue_testkit::Morrow::build(polylogue_testkit::Set::Main);
    let mut runtime =
        crate::test_daemon::start_runtime(&archive.root, archive.config(), ArchiveMode::Standalone)
            .unwrap();
    let daemon = &runtime.daemon;
    let manual = crate::test_answers::ok(
        crate::test_daemon::Call::new("help", json!({"all": true})).answer(daemon),
    )["manual"]
        .as_str()
        .unwrap()
        .to_owned();
    // A declared option may meet a use that is not built yet (`send
    // --attach` to a harness session), refused from the list; any other
    // `not_built` would be an option help offers that its handler refuses.
    let uses: Vec<String> = UNBUILT
        .iter()
        .filter(|row| !row.is_option())
        .map(|row| row.failure().message)
        .collect();
    let refused_as_unbuilt = |answer: &command_surface::Answer<Value>| -> Option<String> {
        let mut refusals = Vec::new();
        collect_unbuilt(&serde_json::to_value(answer).unwrap(), &mut refusals);
        refusals.into_iter().find(|message| !uses.contains(message))
    };
    let mut ran = 0;
    let offered = crate::daemon::verbs::offered_over_mcp();
    for verb in crate::daemon::verbs::declarations()
        .verbs()
        .iter()
        .filter(|verb| offered.iter().any(|(name, _)| *name == verb.verb()))
    {
        assert!(manual.contains(&verb.signature()), "{}", verb.signature());
        let schema = verb.input_schema();
        let mut steps = vec![step(verb)];
        for argument in verb.arguments().iter().filter(|argument| argument.listed) {
            let mut one = step(verb);
            one.insert(
                argument.name.clone(),
                sample(
                    &argument.name,
                    &schema["properties"][&argument.name],
                    schema,
                ),
            );
            steps.push(one);
        }
        for one in steps {
            let arguments = json!({"selection": "latest", "steps": [one], "dry_run": true});
            let answer = crate::test_daemon::Call::new("rows", arguments.clone()).answer(daemon);
            assert_eq!(refused_as_unbuilt(&answer), None, "{arguments}");
            ran += 1;
        }
    }
    let declarations = crate::commands::declarations();
    for command in crate::mcp::operations(crate::commands::Tool::Read) {
        if command.name == "rows" {
            continue;
        }
        let description = declarations
            .descriptions()
            .iter()
            .find(|description| description.spec.name == command.name)
            .unwrap();
        let schema = &description.input_schema;
        let required: Map<String, Value> = schema["required"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(|name| {
                (
                    name.to_owned(),
                    sample(name, &schema["properties"][name], schema),
                )
            })
            .collect();
        for argument in description
            .spec
            .arguments
            .iter()
            .filter(|argument| argument.listed && !argument.calling_process)
        {
            assert!(
                manual.contains(&format!("`{}`", command.name)) && manual.contains(&argument.name),
                "{} {}",
                command.name,
                argument.name
            );
            let mut arguments = required.clone();
            arguments.insert(
                argument.name.clone(),
                sample(
                    &argument.name,
                    &schema["properties"][&argument.name],
                    schema,
                ),
            );
            let arguments = Value::Object(arguments);
            let answer =
                crate::test_daemon::Call::new(command.name, arguments.clone()).answer(daemon);
            assert_eq!(
                refused_as_unbuilt(&answer),
                None,
                "{} {arguments}",
                command.name
            );
            ran += 1;
        }
    }
    assert!(ran > 50, "{ran}");
    runtime.shutdown().unwrap();
}
