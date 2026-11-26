//! The units `import_jobs` and `chat_actions` list what the browser
//! extension was asked to do, each row standing where its events leave it at
//! the query's instant (docs/query.md, units and fields).
use super::*;
use crate::fixture::Fixture;
use polylogue_model::BlobHash;
use polylogue_model::user::{
    ImportJobCounts, ImportJobReport, NewChatAction, NewImportJob, UserWrite,
};
use serde_json::{Value, json};

/// The instant `opts` resolves queries at.
const NOW: i64 = 1_790_000_000_000;

fn rows(fixture: &Fixture, text: &str) -> Vec<Value> {
    let program = compile(text, &opts(&[])).unwrap_or_else(|error| panic!("{text}: {error}"));
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let mut rows = reader.query_rows(&program).unwrap();
    for row in &mut rows {
        crate::query::execute::present(&program, row).unwrap();
    }
    rows
}

/// Each row's values of `columns`, in the unit's order.
fn table(rows: &[Value], columns: &[&str]) -> Vec<Vec<Value>> {
    rows.iter()
        .map(|row| columns.iter().map(|c| row[*c].clone()).collect())
        .collect()
}

fn at(fixture: &mut Fixture, before_now: i64, write: UserWrite) {
    fixture
        .user_at(
            write,
            polylogue_model::Timestamp::from_unix_ms(NOW - before_now).unwrap(),
        )
        .unwrap();
}

fn job(id: &str, provider: &str) -> UserWrite {
    UserWrite::ImportJobStart(NewImportJob {
        id: id.into(),
        provider: provider.into(),
        since: None,
    })
}

fn claim_job(by: &str) -> UserWrite {
    UserWrite::ImportJobClaim { by: by.into() }
}

fn report_job(job: &str, by: &str, kind: &str, position: &str, captured: u64) -> UserWrite {
    UserWrite::ImportJobReport {
        job: job.into(),
        by: by.into(),
        report: ImportJobReport {
            kind: kind.into(),
            position: Some(position.into()),
            counts: ImportJobCounts {
                listed: captured + 2,
                captured,
                unchanged: 2,
                failed: 0,
            },
            detail: None,
        },
    }
}

#[test]
fn import_jobs_unit_lists_provider_imports() {
    let directory = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(directory.path()).unwrap();
    // Finished.
    at(&mut f, 950_000, job("job-finished", "chatgpt"));
    at(&mut f, 940_000, claim_job("ext-1"));
    at(
        &mut f,
        930_000,
        report_job("job-finished", "ext-1", "finished", "3:0", 40),
    );
    // Claimed long ago and quiet since: the claim has expired.
    at(&mut f, 900_000, job("job-quiet", "claude-ai"));
    at(&mut f, 800_000, claim_job("ext-2"));
    // Claimed and renewed by reports within each claim's lifetime.
    at(&mut f, 750_000, job("job-running", "grok"));
    at(&mut f, 700_000, claim_job("ext-3"));
    for (before, position) in [
        (560_000, "0:10"),
        (400_000, "0:20"),
        (240_000, "0:24"),
        (80_000, "0:28"),
    ] {
        at(
            &mut f,
            before,
            report_job("job-running", "ext-3", "progress", position, 26),
        );
    }
    let columns = [
        "id",
        "provider",
        "state",
        "listed",
        "captured",
        "position",
        "claimed_by",
    ];
    assert_eq!(
        table(&rows(&f, "import_jobs"), &columns),
        [
            vec![
                json!("job-running"),
                json!("grok"),
                json!("running"),
                json!(28),
                json!(26),
                json!("0:28"),
                json!("ext-3")
            ],
            vec![
                json!("job-quiet"),
                json!("claude-ai"),
                json!("waiting"),
                json!(0),
                json!(0),
                Value::Null,
                Value::Null
            ],
            vec![
                json!("job-finished"),
                json!("chatgpt"),
                json!("finished"),
                json!(42),
                json!(40),
                json!("3:0"),
                Value::Null
            ],
        ]
    );
    let ids = |text: &str| {
        rows(&f, text)
            .iter()
            .map(|row| row["id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(ids("import_jobs where state:waiting"), ["job-quiet"]);
    assert_eq!(
        ids("import_jobs where NOT state:(finished|failed|cancelled)"),
        ["job-running", "job-quiet"]
    );
    assert_eq!(ids("import_jobs where claimed_by:ext-3"), ["job-running"]);
}

fn action(id: &str, conversation: Option<&str>, submit: bool) -> UserWrite {
    let needs_approval = submit && conversation.is_some();
    UserWrite::ChatEnqueue(NewChatAction {
        id: id.into(),
        request_key: id.into(),
        request_hash: BlobHash::of(id.as_bytes()),
        request: json!({
            "provider": "chatgpt",
            "conversation": conversation,
            "text": format!("turn {id}"),
            "model": "synthetic-model",
            "submit": submit,
        }),
        stage_only: !submit,
        needs_approval,
        attachments: vec![],
    })
}

fn claim_action(by: &str) -> UserWrite {
    UserWrite::ChatClaim { by: by.into() }
}

fn report_action(id: &str, by: &str, outcome: &str, phase: &str) -> UserWrite {
    UserWrite::ChatReport {
        action: id.into(),
        by: by.into(),
        outcome: outcome.into(),
        phase: phase.into(),
        detail: None,
        receipt: None,
    }
}

#[test]
fn chat_actions_unit_lists_turns() {
    let directory = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(directory.path()).unwrap();
    at(&mut f, 950_000, action("drafted", None, false));
    at(&mut f, 940_000, claim_action("tab-a"));
    at(
        &mut f,
        930_000,
        report_action("drafted", "tab-a", "drafted", "drafted"),
    );
    // A submit into an existing conversation waits for the operator.
    at(&mut f, 900_000, action("held", Some("conv-9"), true));
    // Sending began, then the extension went quiet past its claim.
    at(&mut f, 800_000, action("sent", None, true));
    at(&mut f, 790_000, claim_action("tab-b"));
    at(
        &mut f,
        780_000,
        report_action("sent", "tab-b", "progress", "sending"),
    );
    // Claimed once the sending claim lapsed, before sending, and quiet
    // since: back in the queue. One action is out at a time, so a held claim
    // is its own fixture below.
    at(&mut f, 590_000, action("returned", None, false));
    at(&mut f, 580_000, claim_action("tab-d"));
    let columns = [
        "id",
        "kind",
        "state",
        "phase",
        "conversation",
        "needs_approval",
        "claimed_by",
    ];
    assert_eq!(
        table(&rows(&f, "chat_actions"), &columns),
        [
            vec![
                json!("returned"),
                json!("draft"),
                json!("queued"),
                json!("claimed"),
                Value::Null,
                json!(false),
                Value::Null
            ],
            vec![
                json!("sent"),
                json!("submit"),
                json!("outcome_unknown"),
                json!("sending"),
                Value::Null,
                json!(false),
                Value::Null
            ],
            vec![
                json!("held"),
                json!("submit"),
                json!("awaiting_approval"),
                json!("awaiting_approval"),
                json!("conv-9"),
                json!(true),
                Value::Null
            ],
            vec![
                json!("drafted"),
                json!("draft"),
                json!("drafted"),
                json!("drafted"),
                Value::Null,
                json!(false),
                Value::Null
            ],
        ]
    );
    let ids = |text: &str| {
        rows(&f, text)
            .iter()
            .map(|row| row["id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(ids("chat_actions where state:awaiting_approval"), ["held"]);
    assert_eq!(ids("chat_actions where state:queued"), ["returned"]);
    assert_eq!(ids("chat_actions where \"turn sent\""), ["sent"]);
}

#[test]
fn a_renewed_chat_action_claim_names_its_holder() {
    let directory = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(directory.path()).unwrap();
    at(&mut f, 650_000, action("claimed", None, false));
    at(&mut f, 640_000, claim_action("tab-c"));
    // Each renewal falls within the lifetime of the claim before it.
    for before in [500_000, 350_000, 200_000, 60_000] {
        at(
            &mut f,
            before,
            report_action("claimed", "tab-c", "progress", "typing"),
        );
    }
    assert_eq!(
        table(
            &rows(&f, "chat_actions"),
            &["id", "state", "phase", "claimed_by"]
        ),
        [vec![
            json!("claimed"),
            json!("claimed"),
            json!("typing"),
            json!("tab-c")
        ]]
    );
}
