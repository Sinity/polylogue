//! `sum_<field>`, `unknown_<field>` and `pending_<field>`
//! (docs/transforms.md, `agg`): the sum counts every selected session's
//! rows whether its summary is computed or not, a message count is never
//! unknown, and `pending_<field>` counts the sessions whose summary is not
//! computed yet.

use crate::rows;
use polylogue_testkit::{Morrow, Set};
use serde_json::json;

const SUM: &str = "kind:* | agg sum(messages)";

/// A session is pending from the commit of its rows until the summary
/// worker's pass over it is applied. Importing Morrow's two exports into an
/// archive of its other six sessions commits #7's and #8's rows, then asks
/// for their summaries; the derive thread is held once it has computed
/// them and before the writer applies anything, and the reply then counts
/// both sessions' rows and names them pending.
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
        [json!({"sum_messages": 70, "unknown_messages": 0, "pending_messages": 0})]
    );

    let pause = engine.writer().pause_derive_for_test();
    std::thread::scope(|scope| {
        let import = scope.spawn(|| morrow.import_exports(&mut engine));
        pause.wait().expect("the import's summaries are computed");
        // #8's 4 messages and all 8 of #7's: which of them are on the
        // active path is itself a summary, and a message whose path is not
        // known yet counts (docs/query.md, defaults). Once computed, #7's
        // abandoned branch of 2 leaves the count.
        assert_eq!(
            rows(&pool.reader().unwrap(), SUM, &[]),
            [json!({"sum_messages": 82, "unknown_messages": 0, "pending_messages": 2})]
        );
        pause.release();
        let report = import.join().unwrap().expect("the main set imports");
        assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    });
    assert_eq!(
        rows(&pool.reader().unwrap(), SUM, &[]),
        [json!({"sum_messages": 80, "unknown_messages": 0, "pending_messages": 0})]
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
        [json!({"sum_messages": 80, "unknown_messages": 0, "pending_messages": 0})]
    );
}
