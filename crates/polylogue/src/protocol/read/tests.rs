use super::*;
use serde_json::json;

const ROOT: &str = "019a0000-0000-7000-8000-000000000001";
fn resume() -> serde_json::Value {
    json!({"root_run":ROOT,"cursor":"opaque-authenticated-by-owner"})
}

#[test]
fn resume_is_a_closed_form_and_rejects_even_default_valued_selector_overrides() {
    let typed: Arguments = serde_json::from_value(resume()).unwrap();
    let Arguments::Resume(typed) = typed else {
        panic!("resume form");
    };
    assert!(typed.budget.is_none());
    assert_eq!(typed.cursor.as_str(), "opaque-authenticated-by-owner");
    for (key, value) in [
        ("target", json!("sessions")),
        ("view", json!("transcript")),
        ("format", json!("json")),
        ("delivery", json!("page")),
        ("tools", json!("headers")),
        ("thinking", json!("headers")),
        ("harness", json!("headers")),
        ("output_lines", json!(100)),
        ("inherited", json!(false)),
        ("branches", json!(false)),
        ("turns", json!(false)),
        ("assertions", json!(true)),
        ("for_others", json!(false)),
        ("variant", json!([])),
        ("with_blobs", json!(false)),
        ("to", json!("synthetic-output")),
    ] {
        let mut invalid = resume();
        invalid[key] = value;
        assert!(
            serde_json::from_value::<Arguments>(invalid).is_err(),
            "{key}"
        );
    }
    for value in [
        json!({"target":"sessions","cursor":"old-cursor"}),
        json!({"target":"sessions","budget":0}),
        json!({"root_run":ROOT,"cursor":"opaque","budget":0}),
    ] {
        assert!(serde_json::from_value::<Arguments>(value).is_err());
    }
}

#[test]
fn all_read_builders_preserve_the_closed_resume_without_current_defaults() {
    let command = crate::commands::command("read").unwrap();
    let caller = crate::commands::Caller {
        surface: crate::commands::Surface::Http,
        cwd: None,
        json: true,
    };
    let build = command.build_typed.unwrap();
    let request = build(resume(), &caller, "resume-witness").unwrap();
    let crate::daemon::Request::ReadResume {
        root_run,
        cursor,
        budget,
    } = &request
    else {
        panic!("read resume request");
    };
    assert_eq!(root_run.to_string(), ROOT);
    assert_eq!(cursor.as_str(), "opaque-authenticated-by-owner");
    assert!(budget.is_none());
    assert_eq!(
        serde_json::to_value(Arguments::from_request(&request).unwrap()).unwrap(),
        json!({"root_run":ROOT,"cursor":"opaque-authenticated-by-owner","budget":null})
    );
    let pairs = crate::commands::Values::from_pairs(
        command,
        crate::commands::Surface::Http,
        vec![
            ("root_run".into(), ROOT.into()),
            ("cursor".into(), "opaque-authenticated-by-owner".into()),
        ],
    )
    .unwrap();
    assert_eq!(pairs.0.len(), 2);
    let (chosen, values) = crate::cli::parse(&[
        "polylogue",
        "read",
        "--root-run",
        ROOT,
        "--cursor",
        "opaque-authenticated-by-owner",
    ])
    .unwrap();
    let crate::commands::Action::Stream { build, .. } = chosen.action else {
        panic!("read stream adapter");
    };
    let native = crate::commands::Caller {
        surface: crate::commands::Surface::Cli,
        ..caller
    };
    assert!(matches!(
        build(chosen, &values, &native).unwrap(),
        crate::daemon::Request::ReadResume { budget: None, .. }
    ));
    let (chosen, values) = crate::cli::parse(&[
        "polylogue",
        "read",
        "--root-run",
        ROOT,
        "--cursor",
        "opaque",
        "--view",
        "transcript",
    ])
    .unwrap();
    let crate::commands::Action::Stream { build, .. } = chosen.action else {
        unreachable!()
    };
    assert!(build(chosen, &values, &native).is_err());
}
