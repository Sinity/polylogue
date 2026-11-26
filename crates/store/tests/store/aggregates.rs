//! `sum_<field>` and `pending_<field>` (docs/transforms.md, `agg`): the
//! sum counts every selected session's rows whether its summary is
//! computed or not, and `pending_<field>` counts the sessions whose summary
//! is not computed yet.

use crate::rows;
use polylogue_store::fault::{self, Action};
use polylogue_testkit::{Morrow, Set, morrow_dir};
use serde_json::json;

const SUM: &str = "kind:* | agg sum(messages)";

/// A session is pending between the commit of its rows and the summary
/// worker's pass over it. An import that fails after storing one export's
/// conversation leaves that conversation there (the engine summarizes at
/// the end of a successful import or pass), until the writer's next
/// derivation computes it.
#[test]
fn pending_counts_the_sessions_stored_before_their_summary_and_none_after() {
    let morrow = Morrow::lay_out(Set::Main);
    let (mut engine, _lock) = morrow.engine();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let pool = morrow.reads();
    // Sessions #1-#6 (FACTS.md): 12 + 18 + 6 + 11 + 13 + 10 own messages.
    assert_eq!(
        rows(&pool.reader().unwrap(), SUM, &[]),
        [json!({"sum_messages": 70, "pending_messages": 0})]
    );

    // The ChatGPT export (#7) imports; reading the claude.ai export fails
    // before its first conversation.
    fault::arm("parse", "claude-ai/export", Action::Fail, 0, 1);
    let exports = [
        morrow_dir().join("chatgpt/export/conversations.json"),
        morrow_dir().join("claude-ai/export/conversations.json"),
    ];
    assert!(engine.import(&exports).is_err());
    fault::clear();
    engine.writer().flush_rows().unwrap();
    // #7's 6 messages on the active path are counted while its summary is
    // pending.
    assert_eq!(
        rows(&pool.reader().unwrap(), SUM, &[]),
        [json!({"sum_messages": 76, "pending_messages": 1})]
    );

    engine.writer().derive().unwrap();
    assert_eq!(
        rows(&pool.reader().unwrap(), SUM, &[]),
        [json!({"sum_messages": 76, "pending_messages": 0})]
    );
}

/// The whole main set, built and summarized: all eight sessions' own
/// messages (FACTS.md: 12 + 18 + 6 + 11 + 13 + 10 + 6 + 4), none pending.
#[test]
fn a_summarized_archive_has_nothing_pending() {
    let morrow = Morrow::build(Set::Main);
    let pool = morrow.reads();
    assert_eq!(
        rows(&pool.reader().unwrap(), SUM, &[]),
        [json!({"sum_messages": 80, "pending_messages": 0})]
    );
}
