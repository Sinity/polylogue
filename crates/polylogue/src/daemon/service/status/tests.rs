use super::*;

#[test]
fn building_status_keeps_progress_and_does_not_invent_empty_vectors() {
    let daemon = crate::test_daemon::start("synthetic source");
    let progress = polylogue_store::archive::index_build::IndexBuild {
        serving_previous: false,
        chains: 3,
        chains_read: 1,
        bytes: 900,
        bytes_read: 100,
    };
    for detail in [false, true] {
        let answer = building(&daemon.daemon, progress.clone(), detail, false);
        assert_eq!(answer.exit_code(), 5);
        let status = answer.value().unwrap();
        assert_eq!(status.index_build, Some(progress.clone()));
        assert!(status.vectors.is_none() && status.verification.is_none());
        assert_eq!(
            status.vectors_error,
            detail.then(|| progress.to_string()),
            "the unread reason is the build's progress"
        );
    }
}

#[test]
fn verification_retains_runtime_pending_and_named_problem_outcomes() {
    let daemon = crate::test_daemon::start("synthetic source");
    let reader = daemon.daemon.reads.reader().unwrap();
    let status = answer(&daemon.daemon, &reader, false, false);
    drop(reader);
    let mut status = status.map(|mut status| {
        status.pending_writes.delivery_outcomes = 2;
        status.pending_writes.last_error = Some("synthetic retry pending".into());
        status
    });
    let value = crate::protocol::into_value(status).unwrap();
    status = command_surface::Answer::degraded(value, vec!["awaiting source confirmation".into()]);
    status.envelope.notes.push("synthetic runtime note".into());
    let checked = crate::daemon::verify::run(&daemon.daemon, None);
    let found = crate::protocol::into_value(checked).unwrap();
    let expected_verification = found.clone();
    let checked =
        command_surface::Answer::degraded(found, vec!["synthetic storage problem".into()]);
    let combined = with_verification(status, checked);
    let command_surface::Outcome::Degraded { value, missing } = combined.outcome else {
        panic!("{combined:?}");
    };
    assert_eq!(
        missing,
        ["awaiting source confirmation", "synthetic storage problem"]
    );
    assert_eq!(combined.envelope.notes, ["synthetic runtime note"]);
    assert_eq!(value.pending_writes.delivery_outcomes, 2);
    assert_eq!(
        value.pending_writes.last_error.as_deref(),
        Some("synthetic retry pending")
    );
    assert_eq!(value.verification, Some(expected_verification));
}

/// `recent_log` carries the daemon's latest records, a logged warning
/// among them; `--log` returns every kept record, more than an ordinary
/// answer's twenty.
#[test]
fn status_carries_the_recent_log_and_log_returns_every_kept_record() {
    let daemon = crate::test_daemon::start("synthetic source");
    for n in 0..30 {
        polylogue_store::log!(info, "synthetic progress {n}");
    }
    polylogue_store::log!(warning, "synthetic degraded reading");
    let status = |arguments| {
        let answer = crate::test_daemon::Call::new("status", arguments).answer(&daemon.daemon);
        let value = answer.value().expect("status answers").clone();
        serde_json::from_value::<crate::protocol::status::Answer>(value)
            .unwrap()
            .recent_log
    };
    let recent = status(serde_json::json!({}));
    assert_eq!(recent.len(), crate::protocol::status::RECENT_LOG);
    let last = recent.last().unwrap();
    assert_eq!(
        (last.priority, last.text.as_str()),
        (
            polylogue_store::log::Priority::Warning,
            "synthetic degraded reading"
        )
    );
    let whole = status(serde_json::json!({"log": true}));
    assert!(whole.len() > crate::protocol::status::RECENT_LOG);
    let progress: Vec<&str> = whole
        .iter()
        .map(|record| record.text.as_str())
        .filter(|text| text.starts_with("synthetic progress"))
        .collect();
    assert_eq!(progress.len(), 30);
    assert_eq!(progress.first(), Some(&"synthetic progress 0"));
}
