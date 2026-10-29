//! Session populations and current derived fields shared by query readers.

/// Messages shown on the active path, optionally joined to selected sessions.
pub fn own_messages(x: &str, join: Option<&str>) -> String {
    let session = join
        .map(|join| format!(" AND {x}.session_pk {join}"))
        .unwrap_or_default();
    format!("{x}.duplicate_of IS NULL AND COALESCE({x}.on_active_path, 1) = 1{session}")
}

/// A script call counts through its recorded commands, or itself when none exist.
pub fn counts_as_action(t: &str) -> String {
    format!(
        "NOT (coalesce({t}.action,'') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = {t}.session_pk AND cw.run_by_script = 1))"
    )
}

pub fn current(s: &str, name: &str) -> String {
    format!(
        "EXISTS(SELECT 1 FROM derive_state ds WHERE ds.session_pk={s}.pk AND ds.name='{name}' AND ds.from_changed={s}.changed AND pl_derived_current(ds.name,ds.version))"
    )
}

pub fn span(s: &str, column: &str) -> String {
    format!(
        "(SELECT ss.{column} FROM session_span ss WHERE ss.session_pk={s}.pk AND {})",
        current(s, "session_span")
    )
}

pub fn count(s: &str, column: &str) -> String {
    format!(
        "CASE WHEN {} THEN {s}.{column} ELSE {} END",
        current(s, "counts"),
        compute_count(s, column)
    )
}

/// SQL aggregates over the shown message population, in stored column order.
pub const MESSAGE_COUNTS: &[(&str, &str)] = &[
    ("message_count", "count(*)"),
    (
        "human_prompt_count",
        "coalesce(sum(m.written_by='human' AND m.prompt_kind='prose'),0)",
    ),
    (
        "human_command_count",
        "coalesce(sum(m.written_by='human' AND m.prompt_kind='command'),0)",
    ),
    (
        "word_count",
        "coalesce(sum(CASE WHEN m.written_by IN ('human','model') THEN m.words ELSE 0 END),0)",
    ),
    (
        "human_word_count",
        "coalesce(sum(CASE WHEN m.written_by='human' THEN m.words ELSE 0 END),0)",
    ),
    (
        "model_word_count",
        "coalesce(sum(CASE WHEN m.written_by='model' THEN m.words ELSE 0 END),0)",
    ),
    (
        "thinking_count",
        "coalesce(sum(EXISTS(SELECT 1 FROM block b WHERE b.message_pk=m.pk AND b.kind='thinking')),0)",
    ),
];

pub use polylogue_model::derived::COUNT_FIELDS;


/// The same aggregate as the stored column, computed for a behind session.
pub fn compute_count(s: &str, column: &str) -> String {
    if let Some((_, expr)) = MESSAGE_COUNTS.iter().find(|(name, _)| *name == column) {
        return format!(
            "(SELECT {expr} FROM message m WHERE {})",
            own_messages("m", Some(&format!("={s}.pk")))
        );
    }
    if column == "compaction_count" {
        return format!(
            "(SELECT count(*) FROM event e WHERE e.session_pk={s}.pk AND e.kind='compaction')"
        );
    }
    let expr = if column == "error_count" {
        "coalesce(sum(t.outcome='error'),0)"
    } else {
        "count(*)"
    };
    format!(
        "(SELECT {expr} FROM tool_call t JOIN block b ON b.pk=t.call_block_pk JOIN message m ON m.pk=b.message_pk WHERE {} AND {})",
        own_messages("m", Some(&format!("={s}.pk"))),
        counts_as_action("t")
    )
}
