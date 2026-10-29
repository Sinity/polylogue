use super::*;

#[test]
fn a_start_names_a_provider_with_a_conversation_list_and_an_optional_date() {
    let job = new_job("chatgpt", Some("2026-01-02")).unwrap();
    assert_eq!(job.provider, "chatgpt");
    assert_eq!(
        job.since,
        Some(
            "2026-01-02T00:00:00Z"
                .parse::<jiff::Timestamp>()
                .unwrap()
                .as_millisecond()
        )
    );
    assert_eq!(
        new_job("claude-ai", Some("2026-01-02T10:00:00Z"))
            .unwrap()
            .since,
        Some(
            "2026-01-02T10:00:00Z"
                .parse::<jiff::Timestamp>()
                .unwrap()
                .as_millisecond()
        )
    );
    assert_eq!(new_job("grok", None).unwrap().since, None);
    assert_eq!(
        new_job("gemini", None).unwrap_err(),
        Refused("unsupported_provider")
    );
    assert_eq!(
        new_job("chatgpt", Some("last week")).unwrap_err(),
        Refused("since_is_a_date_or_time")
    );
}
