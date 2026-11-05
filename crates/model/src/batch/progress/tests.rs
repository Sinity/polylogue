use super::*;

#[test]
fn a_step_that_reaches_the_end_before_the_stored_length_waits_at_that_length() {
    let cases = [
        (
            StepProgress::AtEnd,
            90,
            ParseProgress::Waiting { length: 100 },
        ),
        (StepProgress::AtEnd, 100, ParseProgress::Done),
        (StepProgress::Partway, 90, ParseProgress::Parsing),
        (
            StepProgress::NoParserYet { at_end: true },
            40,
            ParseProgress::Waiting { length: 100 },
        ),
        (
            StepProgress::NoParserYet { at_end: true },
            100,
            ParseProgress::NoParserYet,
        ),
        (
            StepProgress::NoParserYet { at_end: false },
            40,
            ParseProgress::NoParserYet,
        ),
    ];
    for (step, parsed_upto, expected) in cases {
        assert_eq!(
            ParseProgress::after(step.clone(), parsed_upto, 100),
            expected,
            "{step:?} at {parsed_upto}"
        );
    }
}

#[test]
fn every_standing_round_trips_through_its_stored_columns() {
    for progress in [
        ParseProgress::Parsing,
        ParseProgress::Waiting { length: 7 },
        ParseProgress::Done,
        ParseProgress::NoParserYet,
        ParseProgress::Partial,
        ParseProgress::Refused {
            reason: "synthetic refusal".into(),
            length: 9,
        },
        ParseProgress::Failed {
            error: "synthetic error".into(),
            length: 9,
        },
    ] {
        assert_eq!(
            ParseProgress::from_stored(
                progress.status(),
                progress.at_length(),
                progress.reason().map(str::to_owned)
            ),
            Ok(progress)
        );
    }
    assert!(ParseProgress::from_stored("waiting", None, None).is_err());
    assert!(ParseProgress::from_stored("done", Some(7), None).is_err());
    assert!(ParseProgress::from_stored("waiting at", Some(7), None).is_err());
    assert!(ParseProgress::from_stored("failed", Some(9), None).is_err());
    assert!(ParseProgress::from_stored("failed", None, Some("error".into())).is_err());
    assert!(ParseProgress::from_stored("done", None, Some("why".into())).is_err());
}

#[test]
fn reaching_the_end_keeps_whether_the_origin_has_a_parser() {
    assert_eq!(StepProgress::Partway.with_end(true), StepProgress::AtEnd);
    assert_eq!(StepProgress::AtEnd.with_end(false), StepProgress::Partway);
    assert_eq!(
        StepProgress::NoParserYet { at_end: false }.with_end(true),
        StepProgress::NoParserYet { at_end: true }
    );
    assert!(StepProgress::NoParserYet { at_end: true }.reached_end());
    assert!(!StepProgress::Partway.reached_end());
}

#[test]
fn a_refused_or_failed_step_records_its_reason_at_the_stored_length() {
    assert_eq!(
        ParseProgress::after(
            StepProgress::Failed {
                error: "synthetic error".into()
            },
            40,
            100
        ),
        ParseProgress::Failed {
            error: "synthetic error".into(),
            length: 100
        }
    );
    assert_eq!(
        ParseProgress::after(
            StepProgress::Refused {
                reason: "synthetic reason".into()
            },
            0,
            100
        ),
        ParseProgress::Refused {
            reason: "synthetic reason".into(),
            length: 100
        }
    );
    let failed = StepProgress::Failed {
        error: "synthetic error".into(),
    };
    assert!(!failed.reached_end());
    assert_eq!(failed.clone().with_end(true), failed);
}
