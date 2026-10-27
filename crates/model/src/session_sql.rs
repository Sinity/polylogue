//! Session clock expressions shared by profile readers and query fields.
pub fn first(s: &str) -> String {
    format!(
        "coalesce((SELECT first_at FROM session_survival WHERE session_pk={s}.pk),(SELECT min(at) FROM event WHERE session_pk={s}.pk),{s}.started_at,{s}.last_activity_at)"
    )
}
pub fn last(s: &str) -> String {
    format!(
        "coalesce((SELECT last_at FROM session_survival WHERE session_pk={s}.pk),(SELECT max(at) FROM event WHERE session_pk={s}.pk),{s}.last_activity_at,{s}.started_at)"
    )
}
pub fn source(s: &str) -> String {
    format!(
        "CASE WHEN (SELECT first_at FROM session_survival WHERE session_pk={s}.pk) IS NOT NULL THEN 'provider_supplied' WHEN EXISTS(SELECT 1 FROM event WHERE session_pk={s}.pk AND at IS NOT NULL) THEN 'session_events_fallback' WHEN {s}.started_at IS NOT NULL OR {s}.last_activity_at IS NOT NULL THEN 'session_timestamp_fallback' ELSE 'absent' END"
    )
}
pub fn wall_duration(s: &str) -> String {
    format!("max(0,{}-{})", last(s), first(s))
}
