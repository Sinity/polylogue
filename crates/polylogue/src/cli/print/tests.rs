use super::*;
use crate::browser::import_job::JobSummary;
use crate::resume::{Command, Plan, ResumableSession};
use polylogue_model::SessionId;
use serde_json::json;

const HOSTILE: &str = "\x1b]0;title\x07\x1b]2;title\x1b\\\x1b]52;c;YQ==\x07\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\x1b[2J\x1b\u{9b}2J\u{202e}";

/// HOSTILE with each control written as its visible escape: `\xNN` for C0,
/// DEL and C1 characters, `\u{NNNN}` for the direction override.
const HOSTILE_ESCAPED: &str = "\\x1b]0;title\\x07\\x1b]2;title\\x1b\\\\x1b]52;c;YQ==\\x07\\x1b]8;;https://example.test\\x1b\\link\\x1b]8;;\\x1b\\\\x7f\\x1b[2J\\x1b\\x9b2J\\u{202e}";

fn safe(text: &str) {
    assert!(
        !text
            .chars()
            .any(|c| (c.is_control() && c != '\n' && c != '\t') || c == '\u{202e}'),
        "{text:?}"
    );
}

fn escaped(text: &str) {
    safe(text);
    assert!(text.contains("\\x1b"), "{text:?}");
}

fn hostile_command() -> Command {
    Command {
        program: HOSTILE.into(),
        args: vec![HOSTILE.into()],
        cwd: Some(HOSTILE.into()),
    }
}

fn session() -> SessionId {
    SessionId::parse("codex:print-test").unwrap()
}

#[test]
fn terminal_printers_escape_archived_text() {
    let context_answer = crate::brief::BriefReply {
        delivery: None,
        prepared: crate::brief::Prepared {
            event: polylogue_model::DeliveryEvent::SessionBrief,
            text: HOSTILE.into(),
            refs: vec![],
            left_out: vec![],
            search_summary: None,
            empty: false,
            next: None,
        },
    };
    assert_eq!(brief(&context_answer, TextStyle::Plain), HOSTILE_ESCAPED);
    let plan_answer = protocol::resume::Answer::Plan(Plan {
        session: session(),
        harness_command: Some(hostile_command()),
        context_assisted: hostile_command(),
    });
    escaped(&resume(&plan_answer, TextStyle::Plain));
    let job = JobSummary {
        id: HOSTILE.into(),
        provider: HOSTILE.into(),
        since: None,
        status: polylogue_model::user::ImportJobStatus::Running,
        listed: 1,
        captured: 0,
        unchanged: 0,
        failed: 0,
        position: None,
        detail: Some(HOSTILE.into()),
        created_at: None,
        updated_at: Some(HOSTILE.into()),
        claimed_by: None,
        claim_expires_at: None,
    };
    escaped(&import_job_answer(
        &protocol::browser::ImportJobAnswer::Current(protocol::browser::ImportJobCurrent {
            job: Some(Box::new(job)),
        }),
        TextStyle::Plain,
    ));
    let page = json!({
        "format":"markdown",
        "snapshot":{"incomplete":[],"query":"query:synthetic","ran_at":"2026-03-01T12:00:00Z","read_at":{"generation":"synthetic","index":0,"archive":0},"freshness":{"local":{"parsing":{"state":"complete"},"derivation":{"state":"complete"},"search":{"state":"complete"},"confirmed_stream":{"state":"not_applicable"},"kept_chains":0,"raw_only_chains":0},"sinex":{"mode":"standalone"}}},
        "document":{
            "header":{"target":"event:synthetic","view":"transcript","title":{"state":"value","value":HOSTILE},"kind":{"kind":"event","event_type":"synthetic"}},
            "items":[{"source":{"ref":"event:synthetic"},"kind":{"kind":"text","text":HOSTILE}}],
            "footer":{"referenced_by":[],"shown":1,"count":{"state":"value","value":1}}
        }
    });
    let page: polylogue_model::read::ReadPage = serde_json::from_value(page).unwrap();
    escaped(&read(&page, TextStyle::Plain));
    let sessions = protocol::resume::Answer::Sessions(vec![ResumableSession {
        session: polylogue_store::NearbySession {
            id: session(),
            kind: polylogue_model::SessionKind::Main,
            title: Some(HOSTILE.into()),
            started: None,
            last_activity: None,
            messages: None,
            cwd: None,
        },
        open: polylogue_store::OpenItems {
            todo: Some(HOSTILE.into()),
            waiting_on: Some(HOSTILE.into()),
            ..Default::default()
        },
        follow_ups: vec![HOSTILE.into()],
    }]);
    assert_eq!(
        resume(&sessions, TextStyle::Plain),
        format!(
            "  codex:print-test  {HOSTILE_ESCAPED}\n    todo in progress: {HOSTILE_ESCAPED}; waiting on {HOSTILE_ESCAPED}; {HOSTILE_ESCAPED}\n"
        )
    );
    let status = json!({"notices":[HOSTILE], "errors":[HOSTILE]});
    let text = super::json(&status, TextStyle::Plain);
    safe(&text);
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), status);
}

#[test]
fn continue_plan_prints_the_harness_command_line() {
    let answer = protocol::resume::Answer::Plan(Plan {
        session: session(),
        harness_command: Some(Command {
            program: "codex".into(),
            args: vec!["resume".into(), "print-test".into()],
            cwd: Some("/work/project".into()),
        }),
        context_assisted: Command {
            program: "polylogue".into(),
            args: vec!["codex:print-test".into(), "brief".into()],
            cwd: None,
        },
    });
    assert_eq!(
        resume(&answer, TextStyle::Plain),
        "harness command:    cd /work/project && codex resume print-test\n\
         with context:       polylogue codex:print-test brief\n"
    );
}

#[test]
fn a_harness_resume_without_a_directory_is_not_printed_as_runnable() {
    let answer = protocol::resume::Answer::Plan(Plan {
        session: session(),
        harness_command: Some(Command {
            program: "codex".into(),
            args: vec!["resume".into(), "print-test".into()],
            cwd: None,
        }),
        context_assisted: Command {
            program: "polylogue".into(),
            args: vec!["codex:print-test".into(), "brief".into()],
            cwd: None,
        },
    });
    assert_eq!(
        resume(&answer, TextStyle::Plain),
        "harness command:    directory_unknown: the session recorded no directory to resume in\n\
         with context:       polylogue codex:print-test brief\n"
    );
}

#[test]
fn tag_reports_show_skips_and_escape_row_references() {
    use crate::protocol::line::{LineAnswer, StepAnswer, StepOutcome, StepReport, StepsAnswer};
    let report = polylogue_model::user::AssertQueryOutcome::all_tagged(vec![
        polylogue_model::Reference::Url(HOSTILE.into()),
    ]);
    let answer = LineAnswer::Steps(StepsAnswer {
        resolved: None,
        steps: vec![StepReport {
            verb: "tag".into(),
            request_id: None,
            outcome: StepOutcome::Done {
                reply: StepAnswer::WroteEach(report),
                undo: vec![],
            },
        }],
    });
    let text = line(&answer, TextStyle::Plain);
    assert_eq!(
        text,
        format!("tagged 0 rows\nalready tagged: url:{HOSTILE_ESCAPED}\n")
    );
}

#[test]
fn a_day_zone_this_machine_cannot_load_is_named_not_replaced() {
    let data = json!({"by": "day", "rows": [{"day": "2026-01-01", "sessions": 2}]});
    let text = measurement("time", &data, None, "Nowhere/Unlisted");
    let (shown, note) = text.trim_end().rsplit_once('\n').unwrap();
    assert_eq!(serde_json::from_str::<Value>(shown).unwrap(), data);
    assert!(
        note.starts_with("the day zone Nowhere/Unlisted cannot be loaded here (")
            && note.ends_with("); the measure is shown as its data"),
        "{note}"
    );
}
