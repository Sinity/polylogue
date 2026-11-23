use super::*;

fn refusal(format: &str) -> (String, String) {
    let step = ReadStep {
        format: Some(format.into()),
        ..ReadStep::default()
    };
    let failure = arguments(&step, "sessions".into())
        .err()
        .expect("format is refused");
    (failure.code().to_owned(), failure.error.message.clone())
}

/// A typo is a bad value naming the formats, not a planned feature.
#[test]
fn an_unknown_format_is_a_bad_value_naming_the_formats() {
    let (code, message) = refusal("markdwon");
    assert_eq!(code, "bad_value");
    assert!(
        message.contains("terminal, markdown, html, json and jsonl"),
        "{message}"
    );
}

/// A documented format whose encoder is not built is reported as such.
#[test]
fn a_documented_format_without_an_encoder_is_not_built() {
    let (code, message) = refusal("citation");
    assert_eq!(code, "not_built");
    assert!(message.contains("--format citation"), "{message}");
}

#[test]
fn a_known_format_is_accepted() {
    let step = ReadStep {
        format: Some("jsonl".into()),
        ..ReadStep::default()
    };
    let read = arguments(&step, "sessions".into()).unwrap();
    assert_eq!(read.format, Some(Format::Jsonl));
}
