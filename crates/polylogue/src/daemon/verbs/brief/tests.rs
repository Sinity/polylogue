use crate::test_daemon::{self, Call};
use serde_json::{Value, json};

#[test]
fn brief_routes_by_selection_form() {
    let d = test_daemon::start("a clock discussion");
    let brief = |selection: Option<&str>| {
        let mut rows = json!({"steps": [{"verb": "brief", "cwd": d.home}]});
        if let Some(selection) = selection {
            rows["selection"] = Value::from(selection);
        }
        crate::test_answers::ok(Call::new("rows", rows).answer(&d.daemon))
    };
    let directory = brief(None);
    assert_eq!(
        directory["reply"]["steps"][0]["outcome"]["reply"]["reply"]["prepared"]["event"],
        "session_start",
        "{directory}"
    );
    let session = brief(Some("session:claude-code:s1"));
    assert_eq!(
        session["reply"]["steps"][0]["outcome"]["reply"]["reply"]["prepared"]["event"],
        "session_brief",
        "{session}"
    );
    let words = brief(Some("clock"));
    assert_eq!(
        words["reply"]["steps"][0]["outcome"]["reply"]["reply"]["prepared"]["event"],
        "search_brief",
        "{words}"
    );
    let rows = brief(Some("sessions"));
    assert_eq!(
        rows["reply"]["steps"][0]["outcome"]["error"]["code"], "not_built",
        "{rows}"
    );
}

#[test]
fn brief_keeps_envelope_budget_until_step_overrides_it() {
    use crate::daemon::handler::Operation;
    use command_surface::TokenBudget;
    let d = test_daemon::start("a clock discussion");
    let authorship = polylogue_reading::operation::Context::worker_authorship("brief-test");
    let parts = test_daemon::CallParts::new(&d.daemon, authorship, "budget-brief")
        .cwd(d.home.clone())
        .budget(TokenBudget::Tokens(0));
    let call = parts.call(&d.daemon);
    let brief = |step: Value| {
        let rows = serde_json::from_value(json!({"steps": [step]})).unwrap();
        let answer = crate::daemon::service::line::Line::run(rows, &call).unwrap();
        serde_json::to_value(answer.value().unwrap()).unwrap()
    };
    let too_small = brief(json!({"verb": "brief"}));
    assert_eq!(
        too_small["reply"]["steps"][0]["outcome"]["status"], "failed",
        "{too_small}"
    );
    let overridden = brief(json!({"verb": "brief", "budget": 10000}));
    assert_eq!(
        overridden["reply"]["steps"][0]["outcome"]["status"], "done",
        "{overridden}"
    );
}
