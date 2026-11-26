use super::identical_chains::{capture_log, dump, unnamed};
use super::*;
use polylogue_testkit::{Home, claude_user as user, delete_index};

/// A Claude Code assistant record that is one part of response `response`;
/// the parts of one response make one message.
fn part(uuid: &str, response: &str, text: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({
            "type": "assistant", "uuid": uuid, "sessionId": "same",
            "timestamp": "2026-03-01T10:00:01Z",
            "message": {"id": response, "role": "assistant",
                        "content": [{"type": "text", "text": text}]}
        })
    )
}

/// One input's history, written as six file states: `a`, then `a+b` (an
/// append, so chain 1 grows), `a+c` (chain 3), `a+b` again (chain 4, the
/// bytes of chain 1 stored as one blob), `a+c` again (chain 5, identical to
/// chain 3) and `a+b+d` (chain 6, the latest). Chains 1 and 4 are byte
/// prefixes of chain 6 cut into other blobs, chain 3 is identical to chain
/// 5, and chain 5 diverges from chain 6 after `a`.
struct History {
    a: Vec<u8>,
    b: Vec<u8>,
    c: Vec<u8>,
    d: Vec<u8>,
}

impl History {
    fn states(&self) -> [Vec<u8>; 6] {
        let join = |parts: &[&[u8]]| parts.concat();
        let (a, b, c, d) = (&self.a[..], &self.b[..], &self.c[..], &self.d[..]);
        [
            join(&[a]),
            join(&[a, b]),
            join(&[a, c]),
            join(&[a, b]),
            join(&[a, c]),
            join(&[a, b, d]),
        ]
    }
}

/// A rebuild reads every chain of an input's history whole and gives the
/// same rows whether or not copies of older chains' readings are allowed
/// (docs/archive.md, versioned rows; docs/architecture.md, one path). A
/// later record that changes an earlier row (the second part of a response)
/// does not reach the older chain that never held it.
#[test]
fn a_rebuild_reads_older_transcript_chains_whole_with_or_without_copies() {
    let transcript = History {
        a: [
            user("u1", "same", "first prompt"),
            part("a1", "msg_r", "first part"),
        ]
        .concat()
        .into_bytes(),
        b: [
            part("a2", "msg_r", "second part"),
            user("u2", "same", "next"),
        ]
        .concat()
        .into_bytes(),
        c: [
            unnamed("same", "only in the diverging chain"),
            part("a3", "msg_c", "diverging answer"),
        ]
        .concat()
        .into_bytes(),
        d: [
            user("u3", "same", "last prompt"),
            part("a4", "msg_d", "last answer"),
        ]
        .concat()
        .into_bytes(),
    };
    rebuild_history(&transcript, |w| w.file("same.jsonl"), "claude-code:u1");
}

/// The same for a browser capture log, whose frames each hold a whole
/// conversation.
#[test]
fn a_rebuild_reads_older_capture_log_chains_whole_with_or_without_copies() {
    let frames = |captures: &[&[&str]]| {
        capture_log(captures)[polylogue_model::capture_log::MAGIC.len()..].to_vec()
    };
    let log = History {
        a: capture_log(&[&["a question", "an answer"]]),
        b: frames(&[&["a question", "an answer", "more", "more answer"]]),
        c: frames(&[&["a question", "an edited answer"]]),
        d: frames(&[&["a question", "an answer", "more", "more answer", "last"]]),
    };
    let location = |w: &Home| {
        let log = w
            .config(64 << 20)
            .spool_dir("browser-capture")
            .join("gemini/same.json");
        fs::create_dir_all(log.parent().unwrap()).unwrap();
        log
    };
    rebuild_history(&log, location, "gemini-app:t0");
}

/// Captures `history` at the path `location` gives, then rebuilds the index
/// twice, once with copies refused so every chain is parsed whole, and
/// compares the two. One input per archive, so both rebuilds write in the
/// same order. `message` is a message every chain holds.
fn rebuild_history(
    history: &History,
    location: impl Fn(&Home) -> std::path::PathBuf,
    message: &str,
) {
    fault::clear();
    let w = Home::new();
    let path = location(&w);
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        for bytes in history.states() {
            fs::write(&path, bytes).unwrap();
            assert!(engine.run_pass().unwrap().errors.is_empty());
        }
    }
    // Chains 1, 3, 4, 5 and 6; chain 1 has two revisions.
    assert_eq!(
        w.count("SELECT count(*) AS n FROM (SELECT DISTINCT input, chain FROM a.revision)"),
        5
    );

    let rebuild = |copies: bool| {
        delete_index(&w.root);
        fault::clear();
        if !copies {
            fault::arm("write.copy_reading", "", Action::Fail, 0, u32::MAX);
        }
        let (archive, _lock) = polylogue_store::Archive::open_or_create(&w.root).unwrap();
        let mut config = w.config(64 << 20);
        config.workers = 1;
        let mut engine =
            polylogue_engine::Engine::new(config, archive, polylogue_testkit::stopped_attempt())
                .unwrap();
        let pass = engine.run_pass().unwrap();
        fault::clear();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        drop(engine);
        (pass, dump(&w))
    };
    let (parsed, every_chain_parsed) = rebuild(false);
    assert_eq!((parsed.readings_copied, parsed.bytes_copied), (0, 0));
    let every_chain: u64 = history.states()[1..].iter().map(|s| s.len() as u64).sum();
    assert_eq!(parsed.bytes_read, every_chain);
    // A build reads every chain from its first byte and copies no reading
    // (docs/architecture.md, one path): with copies allowed it reads the
    // same bytes and writes the same rows.
    let (copied, with_copies) = rebuild(true);
    assert_eq!((copied.readings_copied, copied.bytes_copied), (0, 0));
    assert_eq!(copied.bytes_read, every_chain);
    let differing: Vec<String> = every_chain_parsed
        .iter()
        .filter(|(table, rows)| with_copies.get(*table) != Some(rows))
        .map(|(table, rows)| {
            format!(
                "{table}:\n  parsed {rows:?}\n  copied {:?}",
                with_copies.get(table)
            )
        })
        .collect();
    assert!(differing.is_empty(), "{}", differing.join("\n"));
    assert_eq!(
        every_chain_parsed.keys().collect::<Vec<_>>(),
        with_copies.keys().collect::<Vec<_>>()
    );
    // Every chain is read to its end, and every chain a message appeared
    // in stays recorded.
    assert_eq!(
        w.count("SELECT count(*) AS n FROM parse_state WHERE status = 'done'"),
        5
    );
    assert_eq!(
        w.count(&format!(
            "SELECT count(DISTINCT s.chain) AS n FROM message_parse s WHERE s.message_id = '{message}'"
        )),
        5
    );
}
