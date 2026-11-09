//! An export's items commit in batches of about `chunk` bytes.
use super::*;

const CONVERSATIONS: usize = 10;

/// Ten conversations whose array elements all have the same length, in
/// delivery order (oldest first).
fn equal_conversations() -> Vec<u8> {
    let values: Vec<_> = (0..CONVERSATIONS)
        .map(|i| {
            let id = format!("c{i:02}");
            let message = format!("{id}-m");
            let text = format!("body of {id}");
            conversation(&id, 11 + i as u64, &[(message.as_str(), text.as_str())])
        })
        .collect();
    serde_json::to_vec_pretty(&values).unwrap()
}

fn write_counter(w: &ClaudeHome) -> i64 {
    w.count(
        "SELECT coalesce((SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'), 0) AS n",
    )
}

fn all_texts(w: &ClaudeHome) -> Vec<Vec<String>> {
    (0..CONVERSATIONS)
        .map(|i| texts(w, &format!("chatgpt:c{i:02}")))
        .collect()
}

/// Imports `bytes` with `chunk`; returns the home and the index transactions
/// the import committed.
fn import(bytes: &[u8], chunk: u64) -> (ClaudeHome, i64) {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("conversations.json");
    fs::write(&path, bytes).unwrap();
    let (mut e, _lock) = engine(&w, chunk);
    let before = write_counter(&w);
    let report = e.import(std::slice::from_ref(&path)).unwrap();
    assert!(report.refused.is_empty(), "{report:?}");
    assert!(report.pass.errors.is_empty(), "{report:?}");
    let committed = write_counter(&w) - before;
    drop(e);
    (w, committed)
}

/// The length every item of [`equal_conversations`] has in the stored export.
fn item_length(w: &ClaudeHome) -> u64 {
    assert_eq!(
        w.count("SELECT count(DISTINCT source_end - source_start) AS n FROM message_source"),
        1,
        "the items differ in length"
    );
    w.count("SELECT source_end - source_start AS n FROM message_source LIMIT 1") as u64
}

#[test]
fn an_export_commits_one_transaction_per_chunk_of_items() {
    let bytes = equal_conversations();
    let (whole, whole_commits) = import(&bytes, 1 << 40);
    let item = item_length(&whole);
    // Three items reach the chunk and two do not: items 1-3, 4-6 and 7-9 each
    // commit, and the tenth commits with the export's completion.
    let (batched, batched_commits) = import(&bytes, item * 5 / 2);
    assert_eq!(batched_commits - whole_commits, 3);
    assert_eq!(all_texts(&batched), all_texts(&whole));
    assert_eq!(all_texts(&whole)[9], ["body of c09"]);
    let records = "SELECT sum(records) AS n FROM parse_state";
    assert_eq!(batched.count(records), whole.count(records));
    assert_eq!(
        batched.count("SELECT parsed_upto AS n FROM parse_state"),
        bytes.len() as i64
    );
}

/// A parse stopped after its second batch keeps the six items those batches
/// committed; parsed again, the export continues after them and ends with
/// what an uninterrupted parse commits.
#[cfg(feature = "faults")]
#[test]
fn a_stopped_export_resumes_after_its_last_committed_batch() {
    use polylogue_store::fault::{self, Action};
    let bytes = equal_conversations();
    let (whole, _) = import(&bytes, 1 << 40);
    let chunk = item_length(&whole) * 5 / 2;
    let w = ClaudeHome::new();
    let path = w.dir.path().join("conversations.json");
    fs::write(&path, &bytes).unwrap();
    {
        let (mut e, _lock) = engine(&w, chunk);
        fault::clear();
        // Items 1-7 parse; the eighth stops the parse with the seventh
        // parsed but not yet committed.
        fault::arm("parse", &path.to_string_lossy(), Action::Fail, 7, 1);
        let report = e.import(std::slice::from_ref(&path)).unwrap();
        fault::clear();
        assert_eq!(report.pass.errors.len(), 1, "{report:?}");
        assert!(
            report.pass.errors[0].contains("injected fault at parse"),
            "{report:?}"
        );
    }
    let committed: Vec<_> = all_texts(&w)
        .iter()
        .map(|texts| !texts.is_empty())
        .collect();
    assert_eq!(
        committed,
        [
            true, true, true, true, true, true, false, false, false, false
        ]
    );
    assert_eq!(w.count("SELECT parsed_upto AS n FROM parse_state"), 0);
    let (e, _lock) = engine(&w, chunk);
    let config = e.config.clone();
    let archive = e.archive.clone();
    let mut attempt = e.stopped_attempt;
    attempt.executable = polylogue_model::BlobHash::of(b"changed export reader executable");
    drop(e);
    let mut e = Engine::new(config, archive, attempt).unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(all_texts(&w), all_texts(&whole));
    let records = "SELECT sum(records) AS n FROM parse_state";
    assert_eq!(w.count(records), whole.count(records));
    assert_eq!(
        w.count("SELECT parsed_upto AS n FROM parse_state"),
        bytes.len() as i64
    );
}
