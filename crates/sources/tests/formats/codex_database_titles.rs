//! Titles from Codex's state database, through the row dump parser.
use crate::support::{jsonl, parse};
use polylogue_model::{Session, TitleSource};
use serde_json::{Value, json};

const ID: &str = "019a0000-0000-7000-8000-000000000001";
const PATH: &str = "exports/codex-database/state_5.sqlite.jsonl";

fn thread(fields: Value) -> Session {
    let mut row = json!({"id":ID});
    row.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    let bytes = jsonl(&[json!({"t":"threads","k":[ID],"h":"synthetic","r":row})]);
    let parsed = parse("codex-database", PATH, &bytes);
    assert!(parsed["output"]["messages"].as_array().unwrap().is_empty());
    serde_json::from_value(parsed["output"]["sessions"][0].clone()).unwrap()
}

#[test]
fn native_title_is_an_exact_summary_fallback_behind_an_explicit_name() {
    let title = "  Native caption λ\nsecond line  ";
    let session = thread(
        json!({"title":title,"first_user_message":"original prompt","agent_nickname":"Arden"}),
    );
    assert_eq!(
        session.title.unwrap(),
        polylogue_model::Title {
            text: title.into(),
            source: TitleSource::Summary
        }
    );
    let session =
        thread(json!({"title":title,"name":"  Explicit name 🦀  ","agent_nickname":"Arden"}));
    assert_eq!(
        session.title.unwrap(),
        polylogue_model::Title {
            text: "  Explicit name 🦀  ".into(),
            source: TitleSource::Generated
        }
    );
}

#[test]
fn unavailable_title_and_name_values_are_not_coerced_into_captions() {
    let unavailable = [
        Value::Null,
        json!(true),
        json!(12),
        json!(12.5),
        json!([]),
        json!({}),
        json!(""),
        json!(" \n\t "),
    ];
    assert!(thread(json!({})).title.is_none());
    for value in &unavailable {
        assert!(thread(json!({"title":value,"name":value})).title.is_none());
        let session = thread(json!({"title":"native caption","name":value}));
        assert_eq!(session.title.unwrap().source, TitleSource::Summary);
        let session = thread(json!({"title":value,"name":"explicit name"}));
        assert_eq!(session.title.unwrap().source, TitleSource::Generated);
        let session = thread(json!({"title":value,"name":value,"agent_nickname":"Arden"}));
        assert_eq!(session.title.unwrap().source, TitleSource::AgentName);
    }
}

#[test]
fn index_rename_outranks_database_caption_in_either_merge_order() {
    let fallback = thread(json!({"title":"native first-prompt caption"}));
    let renamed = parse(
        "codex",
        ".codex/session_index.jsonl",
        &jsonl(&[json!({"id":ID,"thread_name":"explicit index rename"})]),
    );
    let renamed: Session =
        serde_json::from_value(renamed["output"]["sessions"][0].clone()).unwrap();
    let mut a = fallback.clone();
    a.merge(&renamed);
    let mut b = renamed.clone();
    b.merge(&fallback);
    assert_eq!(a.title, b.title);
    assert_eq!(
        a.title.unwrap(),
        polylogue_model::Title {
            text: "explicit index rename".into(),
            source: TitleSource::Generated
        }
    );
}
