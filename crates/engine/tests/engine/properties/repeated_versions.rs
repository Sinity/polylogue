//! A file rewritten into versions that repeat each other: a rewrite that
//! repeats a prefix of the version before, a version repeated whole after
//! another, a repeat that later diverges, a rewrite flushed half way. The
//! index after incremental updates, one pass after each version, holds the
//! same tables as a rebuild of the same archive (docs/archive.md, readings;
//! docs/architecture.md, one path).

use crate::tables::{differences, dump};
use polylogue_testkit::{
    ClaudeHome, claude_assistant as assistant, claude_user as user, delete_index,
};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;
use std::fs;

/// One version of the file: the first `length` lines of `branch`, the
/// last of them cut in half when `cut`.
#[derive(Debug, Clone)]
struct Version {
    branch: usize,
    length: usize,
    cut: bool,
}

/// How many leading lines each branch shares with the common base.
const SHARED: [usize; 3] = [0, 2, 4];

fn line(branch: usize, i: usize) -> String {
    let (owner, uuid) = if i < SHARED[branch] {
        ("base".to_owned(), format!("base-{i}"))
    } else {
        (format!("b{branch}"), format!("b{branch}-{i}"))
    };
    if i.is_multiple_of(2) {
        user(&uuid, "same", &format!("{owner} says {i}"))
    } else {
        assistant(&uuid, "same", &format!("{owner} answers {i}"))
    }
}

fn bytes(version: &Version) -> String {
    let mut text: String = (0..version.length)
        .map(|i| line(version.branch, i))
        .collect();
    if version.cut {
        let last = line(version.branch, version.length - 1);
        text.truncate(text.len() - last.len() / 2);
    }
    text
}

fn version() -> impl Strategy<Value = Version> {
    (0..SHARED.len(), 1usize..7, prop::bool::weighted(0.15)).prop_map(|(branch, length, cut)| {
        Version {
            branch,
            length,
            cut,
        }
    })
}

/// Writes each version in turn with a pass after each, then rebuilds the
/// index from the archive; returns both indexes' tables.
fn updated_and_rebuilt(versions: &[Version]) -> (ClaudeHome, String) {
    let w = ClaudeHome::new();
    let path = w.file("same.jsonl");
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        for version in versions {
            fs::write(&path, bytes(version)).unwrap();
            let pass = engine.run_pass().unwrap();
            assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        }
    }
    let updated = dump(&w);
    delete_index(&w.root);
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    }
    let rebuilt = dump(&w);
    (w, differences(&updated, &rebuilt))
}

/// The revert of the task: versions `abc`, `abX`, `abcde`. The third
/// repeats the first whole and the second up to `ab`, which the archive
/// records through the agreement of the third with the first.
#[test]
fn a_revert_is_covered_through_recorded_agreement() {
    let versions = [
        Version {
            branch: 2,
            length: 3,
            cut: false,
        },
        Version {
            branch: 1,
            length: 3,
            cut: false,
        },
        Version {
            branch: 2,
            length: 5,
            cut: false,
        },
    ];
    let (w, differ) = updated_and_rebuilt(&versions);
    assert!(differ.is_empty(), "{differ}");
    // The first chain has no parse; the second holds only its third line.
    assert_eq!(w.count("SELECT count(*) AS n FROM parse_state"), 2);
    assert_eq!(
        w.count(
            "SELECT count(*) AS n FROM message_parse
             WHERE chain = (SELECT min(chain) FROM parse_state)"
        ),
        1
    );
}

/// A rewrite that keeps the first lines and then differs: the older chain
/// reads only the records after the agreement.
#[test]
fn a_diverging_older_chain_reads_only_records_after_the_agreement() {
    let versions = [
        Version {
            branch: 2,
            length: 6,
            cut: false,
        },
        Version {
            branch: 1,
            length: 4,
            cut: false,
        },
    ];
    let (w, differ) = updated_and_rebuilt(&versions);
    assert!(differ.is_empty(), "{differ}");
    // Branches 1 and 2 share two base lines; the older chain keeps its
    // four lines after them.
    let ids: Vec<String> = w
        .read(|reader| {
            reader.query(
                "SELECT message_id FROM message_parse
                 WHERE chain = (SELECT min(chain) FROM parse_state) ORDER BY source_start",
                &[],
            )
        })
        .iter()
        .map(|row| row["message_id"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        ids,
        [
            "claude-code:base-2",
            "claude-code:base-3",
            "claude-code:b2-4",
            "claude-code:b2-5"
        ]
    );
}

/// A rewrite flushed half way through a line and never written further:
/// its parser has seen through the bytes it shares with the older version,
/// so the older version's records within them leave, as in a rebuild.
#[test]
fn a_rewrite_flushed_half_way_takes_the_records_it_shares() {
    let versions = [
        Version {
            branch: 2,
            length: 5,
            cut: false,
        },
        Version {
            branch: 2,
            length: 3,
            cut: true,
        },
    ];
    let (w, differ) = updated_and_rebuilt(&versions);
    assert!(differ.is_empty(), "{differ}");
    // The first two lines are the newer chain's; the older keeps the rest.
    assert_eq!(
        w.count(
            "SELECT count(*) AS n FROM message_parse
             WHERE chain = (SELECT min(chain) FROM parse_state)"
        ),
        3
    );
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 24,
        failure_persistence: Some(Box::new(FileFailurePersistence::Off)),
        ..ProptestConfig::default()
    })]

    /// Whatever versions repeat each other, incremental updates and a
    /// rebuild give the same index tables.
    #[test]
    fn chains_stored_in_order_through_upkeep_equal_all_at_once(
        versions in prop::collection::vec(version(), 1..6),
    ) {
        let (_, differ) = updated_and_rebuilt(&versions);
        prop_assert!(differ.is_empty(), "{}", differ);
    }
}
