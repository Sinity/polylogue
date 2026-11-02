use super::*;
use serde_json::json;

#[test]
fn two_flags_map_onto_the_one_progress_value() {
    assert_eq!(progress(false, false), StepProgress::Partway);
    assert_eq!(progress(true, false), StepProgress::AtEnd);
    assert_eq!(
        progress(false, true),
        StepProgress::NoParserYet { at_end: false }
    );
    assert_eq!(
        progress(true, true),
        StepProgress::NoParserYet { at_end: true }
    );
}

#[test]
fn a_reading_with_flags_is_rewritten_and_one_with_progress_is_kept() {
    let mut earlier = json!({"start":0,"at_end":true,"no_parser_yet":true});
    rewrite(earlier.as_object_mut().unwrap()).unwrap();
    assert_eq!(
        earlier,
        json!({"start":0,"progress":{"no_parser_yet":{"at_end":true}}})
    );
    let mut current = json!({"start":0,"progress":"partway"});
    rewrite(current.as_object_mut().unwrap()).unwrap();
    assert_eq!(current, json!({"start":0,"progress":"partway"}));
    let mut neither = json!({"start":0,"at_end":true});
    assert!(matches!(
        rewrite(neither.as_object_mut().unwrap()),
        Err(Error::Invalid(message))
            if message == "stored reading has no `progress` and no boolean `no_parser_yet`"
    ));
}
