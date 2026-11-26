//! Web chats as recipients of `send`, `answer` and `start`
//! (docs/extension.md, Chat actions; docs/session-control.md).
use crate::commands::Surface;
use crate::test_daemon::{self, Call, TestDaemon};
use polylogue_model::authorship::{Authorship, Channel};
use polylogue_model::{Author, Origin};
use serde_json::{Value, json};

fn operator() -> Authorship {
    Authorship {
        author: Author::Operator,
        via: Channel::Cli,
    }
}

fn agent() -> Authorship {
    Authorship {
        author: Author::AgentOrigin(Origin::new("codex").unwrap()),
        via: Channel::Mcp,
    }
}

fn daemon() -> TestDaemon {
    let d = test_daemon::start("web chats");
    d.set("browser.actions", "on");
    d
}

/// One act step on `selection`; its acted value.
#[track_caller]
fn step(
    d: &TestDaemon,
    selection: &str,
    step: Value,
    surface: Surface,
) -> Result<Value, command_surface::Failure> {
    let answer = Call::new("rows", json!({"selection": selection, "steps": [step]}))
        .authorship(operator())
        .surface(surface)
        .answer(&d.daemon);
    let report = crate::test_answers::ok(answer);
    let outcome = &report["reply"]["steps"][0]["outcome"];
    if outcome["status"] == "failed" {
        return Err(serde_json::from_value(outcome["error"].clone()).unwrap());
    }
    Ok(outcome["reply"]["reply"].clone())
}

fn send(d: &TestDaemon, selection: &str, text: &str) -> Value {
    let sent = step(
        d,
        selection,
        json!({"verb": "send", "text": text}),
        Surface::Cli,
    )
    .unwrap();
    sent["reached"][0]["action"].clone()
}

fn action(d: &TestDaemon, id: &str) -> polylogue_model::user::ChatAction {
    d.daemon
        .reads
        .reader()
        .unwrap()
        .chat_action(id)
        .unwrap()
        .unwrap()
}

fn claim(d: &TestDaemon, by: &str) -> Value {
    crate::test_answers::ok(Call::new("internal chat claim", json!({"by": by})).answer(&d.daemon))["action"].clone()
}

#[test]
fn send_to_web_chat_is_held_for_approval() {
    let d = test_daemon::start("web chats off");
    let refused = step(
        &d,
        "session:chatgpt:conv-9",
        json!({"verb": "send", "text": "and the same for the 2023 export"}),
        Surface::Cli,
    )
    .unwrap_err();
    assert_eq!(refused.code.as_deref(), Some("browser_actions_off"));
    assert_eq!(
        refused.message,
        "chat actions are off; turn them on with `polylogue set browser.actions on`"
    );
    d.set("browser.actions", "on");

    let held = send(
        &d,
        "session:chatgpt:conv-9",
        "and the same for the 2023 export",
    );
    assert_eq!(held["status"], "awaiting_approval", "{held}");
    assert_eq!(held["conversation"], "conv-9");
    assert_eq!(held["submit"], true);
    assert_eq!(held["needs_approval"], true);
    assert!(
        held.get("model").is_none(),
        "the conversation keeps its own model"
    );
    // Held turns are not handed to the extension.
    assert_eq!(claim(&d, "ext-1"), Value::Null);

    // A pasted conversation URL the archive does not hold names its session.
    let pasted = send(
        &d,
        "https://chatgpt.com/c/conv-10",
        "a first turn from a link",
    );
    assert_eq!(pasted["status"], "awaiting_approval", "{pasted}");
    assert_eq!(pasted["conversation"], "conv-10");
    assert_eq!(pasted["provider"], "chatgpt");
}

#[test]
fn answer_allow_releases_held_turn() {
    let d = daemon();
    let held = send(&d, "session:chatgpt:conv-9", "go on");
    let id = held["id"].as_str().unwrap().to_owned();
    let allowed = step(
        &d,
        "session:chatgpt:conv-9",
        json!({"verb": "answer", "value": "allow"}),
        Surface::Cli,
    )
    .unwrap();
    assert_eq!(allowed[0]["action"]["id"], id.as_str(), "{allowed}");
    assert_eq!(allowed[0]["action"]["status"], "queued");
    let decided = action(&d, &id);
    let event = decided
        .events
        .iter()
        .find(|e| e.kind == "approved")
        .unwrap();
    assert_eq!(event.by.as_deref(), Some("operator"));
    // Released: the extension takes it.
    assert_eq!(claim(&d, "ext-1")["id"], id.as_str());

    // The first answer wins; a late one is refused with what won.
    let late = step(
        &d,
        "session:chatgpt:conv-9",
        json!({"verb": "answer", "value": "deny"}),
        Surface::Cli,
    )
    .unwrap_err();
    assert_eq!(late.code.as_deref(), Some("conflict"));
    assert_eq!(
        late.message,
        format!("already_answered: the held turn {id} was allowed by operator")
    );

    // deny cancels a held turn.
    let other = send(&d, "session:chatgpt:conv-11", "drop this");
    step(
        &d,
        "session:chatgpt:conv-11",
        json!({"verb": "answer", "value": "deny", "reason": "not now"}),
        Surface::Cli,
    )
    .unwrap();
    let declined = action(&d, other["id"].as_str().unwrap());
    assert_eq!(
        declined.status(i64::MAX).0,
        polylogue_model::user::ChatStatus::Cancelled
    );
}

#[test]
fn mcp_answer_on_held_turn_refused() {
    let d = daemon();
    let held = send(&d, "session:chatgpt:conv-9", "post this");
    let refused = step(
        &d,
        "session:chatgpt:conv-9",
        json!({"verb": "answer", "value": "allow"}),
        Surface::Mcp,
    )
    .unwrap_err();
    assert_eq!(refused.code.as_deref(), Some("refused"));
    assert_eq!(
        refused.message,
        "a held web chat turn is the operator's to answer: approve it in the browser or terminal"
    );
    let still = action(&d, held["id"].as_str().unwrap());
    assert_eq!(
        still.status(0).0,
        polylogue_model::user::ChatStatus::AwaitingApproval
    );
}

fn start(d: &TestDaemon, arguments: Value, authorship: Authorship) -> Value {
    crate::test_answers::ok(
        Call::new("start", arguments)
            .authorship(authorship)
            .answer(&d.daemon),
    )["action"]
        .clone()
}

#[test]
fn start_in_chatgpt_leaves_draft() {
    let d = daemon();
    let draft = start(
        &d,
        json!({"text": "Summarise the attached notes", "in": "chatgpt", "model": "GPT X", "project": "g-p-morrow"}),
        operator(),
    );
    assert_eq!(draft["status"], "queued", "{draft}");
    assert_eq!(draft["submit"], false);
    assert_eq!(draft["conversation"], Value::Null);
    assert_eq!(draft["project"], "g-p-morrow");
    let id = draft["id"].as_str().unwrap().to_owned();
    assert_eq!(claim(&d, "ext-1")["id"], id.as_str());
    let report = |outcome: &str| {
        Call::new(
            "internal chat report",
            json!({"id": id, "update": {"by": "ext-1", "outcome": outcome, "phase": outcome}}),
        )
        .answer(&d.daemon)
    };
    // A draft cannot be submitted; it is left in the composer.
    assert!(matches!(
        report("submitted").outcome,
        command_surface::Outcome::Error(_)
    ));
    assert_eq!(
        crate::test_answers::ok(report("drafted"))["action"]["status"],
        "drafted"
    );
}

#[test]
fn start_submit_sends_new_conversation() {
    let d = daemon();
    let sent = start(
        &d,
        json!({"text": "compare these two retry strategies", "in": "chatgpt", "submit": true}),
        operator(),
    );
    assert_eq!(sent["status"], "queued", "{sent}");
    assert_eq!(sent["submit"], true);
    assert_eq!(sent["needs_approval"], false);
    assert_eq!(claim(&d, "ext-1")["id"], sent["id"]);

    // Anyone else's submit is held for the operator's answer.
    let held = start(
        &d,
        json!({"text": "an agent's new conversation", "in": "chatgpt", "submit": true}),
        agent(),
    );
    assert_eq!(held["status"], "awaiting_approval", "{held}");
    assert_eq!(held["needs_approval"], true);
}
