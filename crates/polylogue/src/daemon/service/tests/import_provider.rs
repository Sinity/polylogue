//! `import PROVIDER`: the browser extension's import job for a web chat,
//! started, stopped and waited for through the `import` row
//! (docs/extension.md, Importing a provider).
use crate::test_daemon::{self, Call, TestDaemon};
use serde_json::{Value, json};

fn import(d: &TestDaemon, arguments: Value) -> command_surface::Answer<Value> {
    Call::new("import", arguments).answer(&d.daemon)
}

fn claim(d: &TestDaemon, by: &str) -> Value {
    crate::test_answers::ok(Call::new("internal import claim", json!({"by": by})).answer(&d.daemon))
}

#[test]
fn import_provider_cancel_stops_job() {
    let d = test_daemon::start("import provider");
    let started = crate::test_answers::ok(import(&d, json!({"paths": ["chatgpt"]})));
    let id = started["job"]["id"].as_str().unwrap().to_owned();
    assert_eq!(started["job"]["status"], "waiting");
    // Starting again while it runs shows the same job.
    let again = crate::test_answers::ok(import(&d, json!({"paths": ["chatgpt"]})));
    assert_eq!(again["job"]["id"], id.as_str());
    assert_eq!(claim(&d, "ext-1")["job"]["id"], id.as_str());

    let cancelled =
        crate::test_answers::ok(import(&d, json!({"paths": ["chatgpt"], "cancel": true})));
    assert_eq!(cancelled["job"]["id"], id.as_str());
    assert_eq!(cancelled["job"]["status"], "cancelled");
    // Nothing is left to hand out or to cancel.
    assert_eq!(claim(&d, "ext-1")["job"], Value::Null);
    let refused =
        crate::test_answers::refusal(import(&d, json!({"paths": ["chatgpt"], "cancel": true})));
    assert_eq!(refused.code.as_deref(), Some("not_found"));
    assert_eq!(
        refused.message,
        "no import of chatgpt is running; start one with `polylogue import chatgpt`"
    );
}

#[test]
fn import_dot_slash_name_reads_file() {
    let d = test_daemon::start("import a path named like a provider");
    let dir = tempfile::tempdir().unwrap();
    // A ChatGPT export unpacked into a directory that happens to be called
    // `chatgpt`.
    let conversation = json!([{"id": "s9", "title": "A directory named chatgpt", "mapping": {"node": {
        "id": "node", "parent": null, "children": [], "message": {"id": "m9",
        "author": {"role": "user"}, "create_time": 1700000000,
        "content": {"content_type": "text", "parts": ["a directory named chatgpt"]}, "metadata": {}}}},
        "current_node": "node"}]);
    std::fs::create_dir(dir.path().join("chatgpt")).unwrap();
    std::fs::write(
        dir.path().join("chatgpt/conversations.json"),
        serde_json::to_vec(&conversation).unwrap(),
    )
    .unwrap();
    let call = Call::new("import", json!({"paths": ["./chatgpt"]})).cwd(dir.path().to_owned());
    let report = crate::test_answers::ok(call.answer(&d.daemon));
    assert!(report.get("job").is_none(), "{report}");
    assert_eq!(report["refused"], json!([]), "{report}");
    assert_eq!(report["packages"][0]["sessions"], 1, "{report}");
    assert!(
        d.daemon
            .reads
            .reader()
            .unwrap()
            .import_jobs()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn import_provider_wait_returns_when_the_job_finishes() {
    let d = test_daemon::start("import provider wait");
    // A wait that runs out answers empty with the job and the same line next.
    let short = import(&d, json!({"paths": ["grok"], "wait": "1s"}));
    let next = short.envelope.next.first().expect("a next call").clone();
    let waited = crate::test_answers::empty(short);
    assert_eq!(waited["job"]["status"], "waiting");
    let id = waited["job"]["id"].as_str().unwrap().to_owned();
    assert_eq!(next.request.operation, "import");
    assert_eq!(next.request.arguments["paths"], json!(["grok"]));
    assert_eq!(next.request.arguments["wait"], "1s");

    assert_eq!(claim(&d, "ext-1")["job"]["id"], id.as_str());
    std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(300));
            let finished = json!({"kind": "finished", "position": "0:0",
                "counts": {"listed": 3, "captured": 1, "unchanged": 2, "failed": 0}});
            crate::test_answers::ok(
                Call::new(
                    "internal import report",
                    json!({"id": id, "by": "ext-1", "report": finished}),
                )
                .answer(&d.daemon),
            );
        });
        let done = crate::test_answers::ok(import(&d, json!({"paths": ["grok"], "wait": "60s"})));
        assert_eq!(done["job"]["id"], id.as_str());
        assert_eq!(done["job"]["status"], "finished");
        assert_eq!(done["job"]["captured"], 1);
    });
}
