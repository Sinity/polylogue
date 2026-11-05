use super::*;
use serde_json::json;

#[test]
fn a_continuation_is_the_same_arguments_with_a_cursor() {
    let typed: Arguments = serde_json::from_value(json!({
        "target": "session:test:one",
        "cursor": "sealed-by-the-daemon",
    }))
    .unwrap();
    assert_eq!(typed.cursor.as_deref(), Some("sealed-by-the-daemon"));
    for value in [
        json!({"target":"sessions","budget":0}),
        json!({"target":"sessions","root_run":"019a0000-0000-7000-8000-000000000001"}),
        json!({"cursor":"no target"}),
    ] {
        assert!(serde_json::from_value::<Arguments>(value).is_err());
    }
}

#[test]
fn every_read_builder_keeps_the_cursor_with_its_reading_options() {
    let command = crate::commands::command("read").unwrap();
    let caller = crate::commands::Caller {
        surface: crate::commands::Surface::Http,
        cwd: None,
        json: true,
    };
    let build = command.build_typed.unwrap();
    let request = build(
        json!({"target":"session:test:one","view":"dialogue","cursor":"sealed"}),
        &caller,
        "continuation-witness",
    )
    .unwrap();
    let crate::daemon::Request::Read {
        options, cursor, ..
    } = &request
    else {
        panic!("read request");
    };
    assert_eq!(options.view, polylogue_model::read::View::Dialogue);
    assert_eq!(cursor.as_deref(), Some("sealed"));
    let round = Arguments::from_request(&request).unwrap();
    assert_eq!(round.cursor.as_deref(), Some("sealed"));
    let (chosen, values) = crate::cli::parse(&[
        "polylogue",
        "read",
        "session:test:one",
        "--view",
        "dialogue",
        "--cursor",
        "sealed",
    ])
    .unwrap();
    let crate::commands::Action::Stream { build, .. } = chosen.action else {
        panic!("read stream adapter");
    };
    let native = crate::commands::Caller {
        surface: crate::commands::Surface::Cli,
        ..caller
    };
    let crate::daemon::Request::Read { cursor, .. } = build(chosen, &values, &native).unwrap()
    else {
        panic!("read request");
    };
    assert_eq!(cursor.as_deref(), Some("sealed"));
}
