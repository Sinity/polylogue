//! Diagnostic only: race the actual complete key algorithms, never row samples.
use super::*;
use crate::query::compile::{DerivedRows, SearchPage, Statements};
use crate::read::search_budget_diagnostic::{attempt, measured};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

fn owner_program(text: &str) -> Program {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            crate::query::compile::DIAGNOSTIC_SUPPLIED.with(|flag| flag.set(false));
        }
    }
    crate::query::compile::DIAGNOSTIC_SUPPLIED.with(|flag| assert!(!flag.replace(true)));
    let _restore = Restore;
    crate::query::compile(text, &options()).unwrap()
}

fn store_error(error: crate::selection::SelectionError) -> crate::StoreError {
    match error {
        crate::selection::SelectionError::Store(error) => error,
        error => crate::StoreError::Invalid(format!("diagnostic selection: {error}")),
    }
}

/// The same key builders visit_page uses; no hydration or replacement algorithm.
fn key_page(reader: &Reader<'_>, program: &Program) -> crate::Result<Vec<Value>> {
    let limit = i64::try_from(program.page_size + 1).unwrap();
    let mut params = crate::selection::values(&program.params);
    params.push(crate::SqlValue::Int(limit));
    let page = program.statements.page(
        &program.order,
        program.params.len(),
        false,
        false,
        crate::query::compile::PageColumns::All,
    );
    let Statements::Keyed(rows) = &program.statements else {
        panic!("unit keys")
    };
    let Some(search) = &rows.search else {
        return reader.query(&page, &params);
    };
    match &search.page {
        SearchPage::Ranked => reader
            .ranked_keys(&page, params, search, limit)
            .map_err(store_error),
        SearchPage::Owners { keys } => {
            let page = Statements::Derived(DerivedRows {
                rows: keys.clone(),
                count: None,
                members: None,
            })
            .page(
                &program.order,
                program.params.len(),
                false,
                false,
                crate::query::compile::PageColumns::All,
            );
            reader.query(&page, &params)
        }
        SearchPage::Recent { keys, .. } => {
            let page = Statements::Derived(DerivedRows {
                rows: keys.clone(),
                count: None,
                members: None,
            })
            .page(
                &program.order,
                program.params.len(),
                false,
                false,
                crate::query::compile::PageColumns::All,
            );
            reader
                .recent_keys(program, &page, params, search, None, limit)
                .map_err(store_error)
        }
    }
}

fn fixture_case(outside: usize, common: bool) -> (tempfile::TempDir, Fixture, i64) {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let wanted = fixture
        .session(SessionFacts {
            id: "test:wanted",
            kind: "main",
            repo: Some("/synthetic/selected"),
            last: Some(1000),
            ..Default::default()
        })
        .unwrap();
    fixture
        .message(
            wanted,
            MessageFacts {
                id: "wanted-message",
                role: "user",
                written_by: "human",
                at: Some(1000),
                text: Some(if common { "needle shared" } else { "needle" }),
                ..Default::default()
            },
        )
        .unwrap();
    for i in 0..outside {
        let id = format!("test:outside-{i}");
        let session = fixture
            .session(SessionFacts {
                id: &id,
                kind: "main",
                repo: Some("/synthetic/outside"),
                last: Some(2000 + i as i64),
                ..Default::default()
            })
            .unwrap();
        fixture
            .message(
                session,
                MessageFacts {
                    id: &format!("outside-message-{i}"),
                    role: "user",
                    written_by: "human",
                    at: Some(2000 + i as i64),
                    text: Some(if common { "needle shared" } else { "ordinary" }),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    (dir, fixture, wanted)
}

#[test]
fn search_budget_diagnostic_actual_key_work_races_rare_common_and_recency_cases() {
    let mut growth = std::collections::BTreeMap::<&str, Vec<u64>>::new();
    let mut oracle_differences = Vec::new();
    for (case, common, text) in [
        ("rare-unbounded", false, "sessions where needle"),
        (
            "common-selective-repo",
            true,
            "sessions where repo:\"/synthetic/selected\" needle",
        ),
        (
            "rare-old-recency",
            false,
            "sessions where needle | sort by last_active desc",
        ),
    ] {
        for outside in [128, 2048] {
            let (_dir, fixture, wanted) = fixture_case(outside, common);
            let program = crate::query::compile(text, &options()).unwrap();
            let choices = if case == "rare-old-recency" {
                let mut indexed = program.clone();
                let Statements::Keyed(rows) = &mut indexed.statements else {
                    panic!("unit")
                };
                rows.search = None;
                vec![("recent", program), ("index", indexed)]
            } else {
                let owners = owner_program(text);
                if common {
                    vec![("index", program), ("owners", owners)]
                } else {
                    vec![("owners", owners), ("index", program)]
                }
            };
            let pool = fixture.reads().unwrap();
            let reader = pool.reader().unwrap();
            let mut complete = Vec::new();
            let mut expected = None;
            for (name, program) in &choices {
                let (rows, exact) = measured(&reader, || key_page(&reader, program));
                let rows = rows.unwrap();
                assert_eq!(rows.len(), 1, "{case}/{name}");
                assert_eq!(rows[0]["_pk"], wanted, "{case}/{name}");
                if let Some(expected) = &expected {
                    assert_eq!(&rows, expected, "{case}/{name}");
                } else {
                    expected = Some(rows);
                }
                assert!(exact.steps > 0);
                complete.push((*name, exact.steps));
            }
            let best = complete.iter().map(|(_, work)| *work).min().unwrap();
            let mut budget = 1000u64; // Existing Reader progress-callback granularity.
            let mut total = 0;
            let mut attempts = 0;
            let winner = 'race: loop {
                for (name, program) in &choices {
                    let (rows, exact) = measured(&reader, || {
                        attempt(&reader, budget, None, || key_page(&reader, program))
                    });
                    total += exact.steps;
                    attempts += 1;
                    if let Some(rows) = rows.unwrap() {
                        assert_eq!(
                            Some(rows),
                            expected,
                            "{case}/{name} chose the same key page"
                        );
                        break 'race *name;
                    }
                }
                budget = budget.checked_mul(2).unwrap();
            };
            assert_eq!(
                winner,
                if common { "owners" } else { "index" },
                "{case}: {complete:?}"
            );
            assert!(
                total <= 16 * (best + 1000),
                "geometric work bound: {case} {complete:?}, race={total}"
            );
            growth.entry(case).or_default().push(total);
            eprintln!(
                "KEY_RACE {}",
                serde_json::json!({"case":case,"outside":outside,"complete":complete,"winner":winner,"race_vm_steps":total,"attempts":attempts,"final_budget":budget})
            );
            drop(reader);
            // Calibrate the exact trace against the original per-op oracle
            // on every complete algorithm, including FTS internal statements.
            // This reader is discarded afterwards; it never runs a budget.
            let calibration = pool.reader().unwrap();
            let counter = Arc::new(AtomicU64::new(0));
            let observed = counter.clone();
            calibration
                .conn()
                .unwrap()
                .progress_handler(
                    1,
                    Some(move || {
                        observed.fetch_add(1, Ordering::Relaxed);
                        false
                    }),
                )
                .unwrap();
            for (name, program) in &choices {
                counter.store(0, Ordering::Relaxed);
                let (rows, exact) = measured(&calibration, || key_page(&calibration, program));
                assert_eq!(Some(rows.unwrap()), expected);
                let original = counter.load(Ordering::Relaxed);
                eprintln!(
                    "ORACLE {}",
                    serde_json::json!({"case":case,"outside":outside,"algorithm":name,"profile_vm_steps":exact.steps,"original_progress_steps":original,"statements":exact.statements})
                );
                if exact.steps != original {
                    oracle_differences.push(format!(
                        "{case}/{outside}/{name}: profile={} original={original}",
                        exact.steps
                    ));
                }
            }
        }
    }
    assert!(
        oracle_differences.is_empty(),
        "exact oracle differences: {oracle_differences:?}"
    );
    for (case, work) in growth {
        assert!(
            work[1] <= work[0] + work[0] / 5 + 2000,
            "unrelated history grows while the winning work stays bounded: {case}: {work:?}"
        );
    }
}
