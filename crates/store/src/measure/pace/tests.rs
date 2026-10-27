use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

#[test]
fn threshold_boundary_model_work_idle_negative_and_missing() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "codex:pacing",
            ..Default::default()
        })
        .unwrap();
    for (i, (role, at)) in [
        ("user", Some(0)),
        ("assistant", Some(60_000)),
        ("user", None),
        ("assistant", Some(120_001)),
        ("user", Some(200_000)),
        ("assistant", Some(100_000)),
    ]
    .into_iter()
    .enumerate()
    {
        f.message(
            s,
            MessageFacts {
                id: &format!("m{i}"),
                role,
                written_by: if role == "user" { "human" } else { "model" },
                at,
                ..Default::default()
            },
        )
        .unwrap();
    }
    f.session(SessionFacts {
        id: "codex:sparse",
        ..Default::default()
    })
    .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let scope = Selection {
        sessions: Some(vec![s]),
        ..Default::default()
    };
    let x = pace(&r, &scope, "60").unwrap();
    assert_eq!(x.total_gaps, 3);
    let report = &x.sessions[0];
    assert_eq!(report.active_ms, 60_000);
    assert_eq!(report.model_work_ms, 60_001);
    assert_eq!(report.user_idle_ms, 79_999);
    assert_eq!(report.ignored_negative_gaps, 1);
    assert_eq!(x.gaps[0].kind, "user_idle");
    assert_eq!(x.gaps[0].to_position, 4);
    assert_eq!(report.user_gaps, 0);
    assert_eq!(report.burstiness, None);
    assert_eq!(
        pace(&r, &Selection::default(), "").unwrap().sessions.len(),
        2
    );
    assert!(pace(&r, &scope, "-1").is_err());
}

#[test]
fn burstiness_uses_population_sigma_and_requires_ten_user_gaps() {
    assert_eq!(burstiness(&[10.0; 9]), None);
    assert_eq!(burstiness(&[10.0; 10]), Some(-1.0));
    assert_eq!(burstiness(&[0.0; 10]), None);
    let mut gaps = [0.0; 10];
    gaps[9] = 100.0;
    assert!((burstiness(&gaps).unwrap() - 0.5).abs() < 1e-12);
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "codex:users",
            ..Default::default()
        })
        .unwrap();
    for i in 0..11 {
        f.message(
            s,
            MessageFacts {
                id: &format!("u{i}"),
                role: "user",
                written_by: "human",
                at: Some(i * 10),
                ..Default::default()
            },
        )
        .unwrap();
    }
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let x = pace(&r, &Selection::default(), "").unwrap();
    assert_eq!(x.sessions[0].burstiness, Some(-1.0));
    assert_eq!(x.sessions[0].user_gaps, 10);
}

#[test]
fn harness_tool_result_does_not_count_as_user_idle() {
    let mut f = crate::fixture::writer::WriterArchive::new();
    let mut calls = crate::fixture::writer::bash("long", 1, 0, "synthetic", true, "ok");
    calls[1].at = polylogue_model::Timestamp::from_unix_ms(720_000);
    f.write("long", calls);
    let pool = crate::ReadPool::open(&f.archive).unwrap();
    let r = pool.reader().unwrap();
    let x = pace(&r, &Selection::default(), "").unwrap();
    assert_eq!(x.sessions[0].turns, 1);
    assert_eq!(x.sessions[0].user_idle_ms, 0);
    assert!(x.gaps.is_empty());
}
