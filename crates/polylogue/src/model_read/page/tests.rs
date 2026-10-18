use super::*;
use std::io::Write;

fn window(text: &str) -> Window {
    let mut w = Window {
        start: 0,
        at: 0,
        text: String::new(),
        ends: Vec::new(),
        limit: 100000,
        full: false,
        present: true,
    };
    for c in text.chars() {
        w.write_all(c.to_string().as_bytes()).unwrap();
    }
    w
}
fn identity(i: usize) -> Value {
    json!({"block":format!("block:neutral:{i}"),"sources":[{"start":i,"end":i+1}],"field_origins":[{"path":[i]}]})
}
fn envelope() -> Value {
    json!({"outcome":"ok","extra_status":{"null":null,"false":false,"extra":"preserved"},"authored_associations_complete":false,"authored_notes":{"next":{"view":"annotations"}}})
}

#[test]
fn consecutive_fragments_keep_envelope_fields_and_note_pages() {
    for mcp in [false, true] {
        let mut args = json!({"reference":"claude-code:neutral","budget_tokens":4000});
        if mcp {
            args["presentation"] = json!("mcp");
        }
        let request = Request::decode(&args).unwrap();
        let mut assembler = Assembler::new(
            envelope(),
            &request,
            [0; 32],
            12,
            Position {
                block: 0,
                scalar: 0,
                budget: 4000,
            },
        );
        for i in 0..12 {
            assert!(assembler.push(identity(i), window("equal 🙂\u{1b}"), i as u64));
        }
        let page = assembler.finish().unwrap();
        assert_eq!(page["shown"].as_array().unwrap().len(), 12);
        assert_eq!(page["extra_status"], envelope()["extra_status"]);
        assert_eq!(page["authored_notes"], envelope()["authored_notes"]);
        assert_eq!(page["authored_associations_complete"], false);
        assert_eq!(page["primary_content_complete"], true);
        for i in 0..12 {
            let shown = &page["shown"][i];
            assert_eq!(shown["block"], identity(i)["block"]);
            assert_eq!(shown["field_origins"], identity(i)["field_origins"]);
            assert_eq!(shown["text"], "equal 🙂\\x1b");
        }
        assert!(request.cost(&page) <= request.budget);
    }
}

#[test]
fn final_null_envelope_may_fit_when_a_nonfinal_prefix_does_not() {
    let args = json!({"reference":"claude-code:neutral","budget_tokens":4000});
    let request = Request::decode(&args).unwrap();
    let mut a = Assembler::new(
        envelope(),
        &request,
        [0; 32],
        2,
        Position {
            block: 0,
            scalar: 0,
            budget: 4000,
        },
    );
    a.push(identity(0), window("x"), 0);
    let nonfinal = request.cost(&a.candidate(1, false));
    a.push(identity(1), window("y"), 1);
    let final_cost = request.cost(&a.candidate(2, false));
    assert!(final_cost < nonfinal);
    let args = json!({"reference":"claude-code:neutral","budget_tokens":final_cost.max(MINIMUM)});
    let request = Request::decode(&args).unwrap();
    let mut a = Assembler::new(
        envelope(),
        &request,
        [0; 32],
        2,
        Position {
            block: 0,
            scalar: 0,
            budget: request.budget,
        },
    );
    assert!(a.push(identity(0), window("x"), 0));
    assert!(a.push(identity(1), window("y"), 1));
    assert_eq!(a.finish().unwrap()["shown"].as_array().unwrap().len(), 2);
}

#[test]
fn partial_window_never_splits_a_replacement_or_control_escape_and_retries_same_position() {
    for budget in [128, 200, 300, 600] {
        let args = json!({"reference":"claude-code:neutral","budget_tokens":budget});
        let request = Request::decode(&args).unwrap();
        let mut w = window("🙂\u{1b}");
        let mut masker = credential_output::Masker::new();
        masker
            .push(&format!("ghp_{}", "A".repeat(36)), &mut w)
            .unwrap();
        masker.finish(&mut w).unwrap();
        for _ in 0..1000 {
            w.write_all(b"x").unwrap();
        }
        w.full = true;
        let ends = w.ends.clone();
        let mut a = Assembler::new(
            envelope(),
            &request,
            [0; 32],
            1,
            Position {
                block: 0,
                scalar: 0,
                budget,
            },
        );
        assert!(!a.push(identity(0), w, 0));
        match a.finish() {
            Ok(page) => {
                assert!(request.cost(&page) <= budget);
                let end = page["shown"][0]["display_end"].as_u64().unwrap();
                assert!(ends.iter().any(|(_, n)| *n == end));
                let token = page["next"]["continuation"].as_str().unwrap();
                let (_, _, p) = super::super::open(token, super::super::key().unwrap()).unwrap();
                assert_eq!((p.block, p.scalar), (0, end));
            }
            Err(error) => assert_eq!(error["error"]["reason"], "insufficient_budget"),
        }
    }
}

#[test]
fn empty_and_absent_fragments_consume_boundaries_without_inventing_text() {
    let args = json!({"reference":"claude-code:neutral","budget_tokens":4000});
    let request = Request::decode(&args).unwrap();
    let mut a = Assembler::new(
        envelope(),
        &request,
        [0; 32],
        2,
        Position {
            block: 0,
            scalar: 0,
            budget: 4000,
        },
    );
    a.push(identity(0), window(""), 0);
    let mut absent = window("");
    absent.present = false;
    a.push(identity(1), absent, 1);
    let p = a.finish().unwrap();
    assert_eq!(p["shown"][0]["text"], "");
    assert!(p["shown"][1].get("text").is_none());
    assert!(p["next"].is_null());
}

#[test]
fn re_masking_earlier_blocks_counts_as_work_even_for_a_tiny_window() {
    use polylogue_engine::owner::{ImportRequest, Observation, Owner};
    use polylogue_store::{InputFormat, Writer};
    use std::{
        sync::{Arc, atomic::AtomicBool},
        time::{Duration, Instant},
    };
    let dir = tempfile::tempdir().unwrap();
    let writer = Writer::open(&dir.path().join("archive")).unwrap();
    let reader = writer.read_handle().unwrap();
    let owner = Owner::start(writer, Arc::new(AtomicBool::new(false))).unwrap();
    let path = dir.path().join("neutral.jsonl");
    let sid = "94fdef03-2101-4000-8000-000000000003";
    let record = json!({"type":"user","uuid":"work","sessionId":sid,"parentUuid":null,"promptSource":"typed","message":{"role":"user","content":[{"type":"text","text":"x".repeat(64)},{"type":"text","text":"y"}]}});
    std::fs::write(&path, format!("{record}\n")).unwrap();
    let input = owner
        .handle()
        .try_import(ImportRequest {
            format: InputFormat::ClaudeCodeJsonl,
            context: vec!["claude-main".into(), sid.into()],
            path: path.clone(),
            observation: Observation {
                id: "neutral-work".into(),
                machine: "synthetic".into(),
                host: "synthetic".into(),
                path: path.to_str().unwrap().into(),
                seen_at: 0,
            },
            metadata_allowance: 65536,
        })
        .unwrap()
        .recv_timeout(Duration::from_secs(30))
        .unwrap()
        .unwrap();
    let until = Instant::now() + Duration::from_secs(30);
    loop {
        let progress = owner
            .handle()
            .try_progress(&input.id)
            .unwrap()
            .recv_timeout(Duration::from_secs(30))
            .unwrap()
            .unwrap();
        assert!(progress.failure.is_none(), "{progress:?}");
        if matches!(
            progress.source_rows.state,
            polylogue_model::source_rows::SourceRowState::Complete
        ) {
            break;
        }
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(5));
    }
    let m = super::super::message(&reader, &format!("claude-code:{sid}:work")).unwrap();
    let b = m.block_at(1).unwrap().unwrap();
    let mut work = super::super::Work { remaining: 63 };
    let error = super::super::window(&m, 1, &b, 0, 1, &mut work, until, &mut || false)
        .err()
        .expect("prefix must consume work before the one-byte selected body");
    assert_eq!(error["error"]["reason"], "transform_work_limit");
    let mut work = super::super::Work { remaining: 65 };
    let w = super::super::window(&m, 1, &b, 0, 1, &mut work, until, &mut || false).unwrap();
    assert_eq!(w.text, "y");
    assert_eq!(work.remaining, 0);
    owner.shutdown().unwrap();
}
