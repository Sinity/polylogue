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
            ParseProgress::after(step, parsed_upto, 100),
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
        },
        ParseProgress::Failed {
            error: "synthetic error".into(),
        },
    ] {
        assert_eq!(
            ParseProgress::from_stored(
                progress.status(),
                progress.waiting_at(),
                progress.reason().map(str::to_owned)
            ),
            Ok(progress)
        );
    }
    assert!(ParseProgress::from_stored("waiting", None, None).is_err());
    assert!(ParseProgress::from_stored("done", Some(7), None).is_err());
    assert!(ParseProgress::from_stored("waiting at", Some(7), None).is_err());
    assert!(ParseProgress::from_stored("failed", None, None).is_err());
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
