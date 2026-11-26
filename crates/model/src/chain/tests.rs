use super::*;

fn facts(stored: u64) -> ChainFacts {
    ChainFacts {
        stored,
        repeated_by: None,
        own_from: 0,
        parser: Some("v2".into()),
        finished: false,
    }
}

fn parsed(parsed_to: u64, seen_to: u64, resume: bool) -> ChainState {
    ChainState::Parsed {
        version: "v2".into(),
        parsed_to,
        seen_to,
        resume: resume.then(ParserState::default),
        counts: Outcomes::default(),
    }
}

fn failed(kind: ChainFailureKind, seen_to: u64) -> ChainState {
    ChainState::Failed {
        version: "v2".into(),
        parsed_to: 40,
        seen_to,
        failure: ChainFailure {
            kind,
            detail: "line 3 is not JSON".into(),
        },
        counts: Outcomes::default(),
    }
}

#[test]
fn a_chain_is_due_only_for_a_new_length_or_a_new_parser_version() {
    let cases = [
        ("never parsed", ChainState::Unparsed, facts(100), Due::Whole),
        (
            "empty and never parsed",
            ChainState::Unparsed,
            facts(0),
            Due::No,
        ),
        ("complete", parsed(100, 100, true), facts(100), Due::No),
        (
            "waiting at the stored length",
            parsed(90, 100, true),
            facts(100),
            Due::No,
        ),
        (
            "waiting, then grown",
            parsed(90, 100, true),
            facts(130),
            Due::Extend { from: 90 },
        ),
        (
            "done, then grown",
            parsed(100, 100, true),
            facts(130),
            Due::Extend { from: 100 },
        ),
        (
            "grown without a resume state",
            parsed(100, 100, false),
            facts(130),
            Due::Whole,
        ),
        (
            "stopped partway",
            parsed(50, 50, true),
            facts(100),
            Due::Extend { from: 50 },
        ),
        (
            "failed at the stored length",
            failed(ChainFailureKind::Malformed, 100),
            facts(100),
            Due::No,
        ),
        (
            "failed, then grown",
            failed(ChainFailureKind::Malformed, 100),
            facts(120),
            Due::Whole,
        ),
        (
            "refused at the stored length",
            failed(ChainFailureKind::Refused, 100),
            facts(100),
            Due::No,
        ),
    ];
    for (case, state, facts, expected) in cases {
        assert_eq!(due(&state, &facts), expected, "{case}");
    }
}

#[test]
fn a_new_parser_version_reads_every_state_again_whole() {
    let mut newer = facts(100);
    newer.parser = Some("v3".into());
    for state in [
        parsed(100, 100, true),
        parsed(90, 100, true),
        failed(ChainFailureKind::Malformed, 100),
        failed(ChainFailureKind::Refused, 100),
    ] {
        assert_eq!(due(&state, &newer), Due::Whole, "{state:?}");
    }
}

#[test]
fn a_repeated_chain_has_no_parse_and_no_parser_reads_nothing() {
    let mut repeated = facts(100);
    repeated.repeated_by = Some(7);
    assert_eq!(due(&ChainState::Unparsed, &repeated), Due::No);
    assert_eq!(due(&parsed(100, 100, true), &repeated), Due::Remove);
    assert_eq!(
        status(&parsed(100, 100, true), &repeated),
        ChainStatus::Repeated
    );
    let mut no_parser = facts(100);
    no_parser.parser = None;
    assert_eq!(due(&ChainState::Unparsed, &no_parser), Due::No);
    assert_eq!(
        status(&ChainState::Unparsed, &no_parser),
        ChainStatus::NoParser
    );
}

#[test]
fn status_shows_due_as_parsing_and_otherwise_the_state_at_the_stored_length() {
    let cases = [
        (ChainState::Unparsed, facts(100), ChainStatus::Parsing),
        (parsed(100, 100, true), facts(100), ChainStatus::Done),
        (
            parsed(90, 100, true),
            facts(100),
            ChainStatus::Waiting { length: 100 },
        ),
        (parsed(90, 100, true), facts(130), ChainStatus::Parsing),
        (
            failed(ChainFailureKind::Refused, 100),
            facts(100),
            ChainStatus::Refused {
                reason: "line 3 is not JSON".into(),
            },
        ),
        (
            failed(ChainFailureKind::ParserPanicked, 100),
            facts(100),
            ChainStatus::Failed {
                error: "line 3 is not JSON".into(),
            },
        ),
    ];
    for (state, facts, expected) in cases {
        assert_eq!(status(&state, &facts), expected, "{state:?}");
    }
    let names: Vec<_> = ChainStatusName::ALL
        .iter()
        .map(|name| name.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "parsing",
            "repeated",
            "no_parser",
            "done",
            "waiting",
            "refused",
            "failed"
        ]
    );
}
