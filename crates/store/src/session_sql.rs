//! Session populations and current derived fields shared by query readers.

/// Exact directory and descendant range, compared as whole path components.
pub fn directory_bounds(directory: &str) -> [String; 3] {
    let base = directory.trim_end_matches('/');
    let exact = if base.is_empty() && !directory.is_empty() {
        "/"
    } else {
        base
    };
    [exact.into(), format!("{base}/"), format!("{base}0")]
}

pub fn path_under(path: &str, directory: &str) -> bool {
    let [exact, first, end] = directory_bounds(directory);
    path == exact || (path >= first.as_str() && path < end.as_str())
}

fn path_range(path: &str, [exact, first, end]: [String; 3]) -> String {
    format!("({path} = {exact} OR ({path} >= {first} AND {path} < {end}))")
}

/// Bound directory values let SQLite scan the path column by range.
pub fn path_under_sql(path: &str, directory: &str, mut bind: impl FnMut(&str) -> String) -> String {
    path_range(path, directory_bounds(directory).map(|value| bind(&value)))
}

/// A directory read from another row uses the same component range.
pub fn path_under_column(path: &str, directory: &str) -> String {
    let base = format!("rtrim({directory}, '/')");
    path_range(
        path,
        [
            format!("CASE WHEN {base}='' AND {directory}<>'' THEN '/' ELSE {base} END"),
            format!("{base}||'/'"),
            format!("{base}||'0'"),
        ],
    )
}

pub fn live_root(s: &str) -> String {
    format!(
        "(json_extract({s}.live_facts, '$.root') IS NOT NULL AND json_extract({s}.live_facts, '$.root_gone_at') IS NULL AND json_extract({s}.live_facts, '$.ended_at') IS NULL)"
    )
}

/// Candidates include live roots regardless of the age of their last event.
pub fn live_candidates(s: &str, now: &str) -> String {
    use polylogue_model::live;
    format!(
        "({} OR {s}.live_last_at >= {now}-{} OR ({s}.live_last_at IS NULL AND {s}.last_activity_at >= {now}-{}))",
        live_root(s),
        live::STALE_FOR_MS + live::LIVE_WINDOW_MS,
        live::LIVE_WINDOW_MS + 1
    )
}

pub fn unanswered_prompt(session_pk: &str) -> String {
    format!(
        "coalesce((SELECT m.written_by='human' AND coalesce(m.prompt_kind,'prose')<>'command' FROM message m WHERE {} ORDER BY m.session_order DESC,m.id DESC LIMIT 1),0)",
        own_messages("m", Some(&format!("={session_pk}")))
    )
}

pub fn unfinished_tool_call(session_pk: &str) -> String {
    format!(
        "EXISTS(SELECT 1 FROM tool_call t JOIN block b ON b.pk=t.call_block_pk JOIN message m ON m.pk=b.message_pk WHERE {} AND t.result_message_id IS NULL)",
        own_messages("m", Some(&format!("={session_pk}")))
    )
}

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

/// The title a session is shown with, everywhere: its current authored
/// `title` annotations (several current families are a conflict, shown as
/// every value joined by ` | `), else the recorded title, else the derived
/// one. Withdrawn titles are not shown.
pub fn display_title(s: &str) -> String {
    format!(
        "COALESCE((SELECT group_concat(tv.value, ' | ') FROM (SELECT DISTINCT json_extract(ta.value, '$') AS value FROM source_annotation_link tl \
         JOIN source_annotation ta ON ta.id = tl.annotation AND ta.kind = 'title' \
         JOIN source_revision_state ts ON ts.id = ta.id AND ts.current = 1 AND ts.withdrawn = 0 \
         WHERE tl.ref = 'session:' || {s}.id AND tl.role = 'about' ORDER BY 1) tv), \
         {s}.title, {s}.derived_title)"
    )
}

pub fn current(s: &str, name: &str) -> String {
    format!(
        "(NOT EXISTS(SELECT 1 FROM summary_due dd WHERE dd.session_pk={s}.pk) AND EXISTS(SELECT 1 FROM summary_state ds WHERE ds.session_pk={s}.pk AND ds.name='{name}' AND pl_summary_current(ds.name,ds.version)))"
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

/// `column` (a reference) names the record `root` (an SQL expression giving
/// its reference without a revision) or one exact revision of it.
pub(crate) fn names_record(column: &str, root: &str) -> String {
    format!("({column}={root} OR ({column}>{root}||'@' AND {column}<{root}||'A'))")
}

/// `column` (a reference) starts with `prefix` (an SQL expression) and `#`:
/// it names a part of what `prefix` names.
fn names_part(column: &str, prefix: &str) -> String {
    format!("({column}>{prefix}||'#' AND {column}<{prefix}||'$')")
}

/// `column` (a reference) names the message whose ID is `id` (an SQL
/// expression) or something inside it: a block, or a span of either.
fn names_message(column: &str, id: &str) -> String {
    format!(
        "({column}='message:'||{id} OR {} OR {} OR {})",
        names_part(column, &format!("'block:'||{id}")),
        names_part(column, &format!("'span:message:'||{id}")),
        names_part(column, &format!("'span:block:'||{id}")),
    )
}

/// Annotation `a` is about session `s` or something in it: one of its
/// messages or tool calls, or a block or span inside a message.
/// The containment is `Reference::container`'s, written over stored IDs.
fn about_session(a: &str, s: &str) -> String {
    format!(
        "EXISTS(SELECT 1 FROM source_annotation_link l WHERE l.annotation={a}.id AND l.role='about' AND (l.ref='session:'||{s}.id OR {} OR EXISTS(SELECT 1 FROM message m WHERE m.session_pk={s}.pk AND {})))",
        names_part("l.ref", &format!("'tool:'||{s}.id")),
        names_message("l.ref", "m.id")
    )
}

/// A current reply names the question's root or one exact revision.
pub fn answered(r: &str) -> String {
    format!(
        "({r}.kind='question' AND EXISTS(SELECT 1 FROM source_annotation_link al JOIN source_annotation aa ON aa.id=al.annotation JOIN source_revision_state ar ON ar.id=aa.id AND ar.current=1 AND ar.withdrawn=0 WHERE al.role='replies_to' AND {}))",
        names_record("al.ref", &format!("'annotation:'||{r}.root"))
    )
}

/// Open work follows current authored answers, decisions, judgments and blockers.
pub fn open_annotation(r: &str) -> String {
    let root = format!("'annotation:'||{r}.root");
    let reply = names_record("rl.ref", &root);
    let blocks = names_record("bl.ref", &root);
    let blocker_answered = names_record("repl.ref", "'annotation:'||ba.root");
    let closed = format!(
        "EXISTS(SELECT 1 FROM source_annotation_link rl JOIN source_annotation ra ON ra.id=rl.annotation JOIN source_revision_state rr ON rr.id=ra.id AND rr.current=1 AND rr.withdrawn=0 WHERE {reply} AND ((rl.role='about' AND ra.kind='decision') OR (rl.role ='replies_to' AND EXISTS(SELECT 1 FROM source_annotation_link jl JOIN source_annotation ja ON ja.id=jl.annotation AND ja.kind='judgment' AND ja.author='operator' AND ja.key='accepted' JOIN source_revision_state jr ON jr.id=ja.id AND jr.current=1 AND jr.withdrawn=0 WHERE jl.role='about' AND (jl.ref='annotation:'||ra.root OR jl.ref='annotation:'||ra.root||'@'||ra.id)))))"
    );
    let blocked = format!(
        "EXISTS(SELECT 1 FROM source_annotation_link bl JOIN source_annotation ba ON ba.id=bl.annotation JOIN source_revision_state br ON br.id=ba.id AND br.current=1 AND br.withdrawn=0 WHERE bl.role='blocks' AND {blocks} AND NOT EXISTS(SELECT 1 FROM source_annotation_link repl JOIN source_revision_state repls ON repls.id=repl.annotation AND repls.current=1 AND repls.withdrawn=0 WHERE repl.role ='replies_to' AND {blocker_answered}))"
    );
    format!(
        "(EXISTS(SELECT 1 FROM source_revision_state os WHERE os.id={r}.id AND os.current=1 AND os.withdrawn=0) AND (({r}.kind IN ('question','goal','problem') AND NOT {} AND NOT {closed}) OR {blocked}))",
        answered(r)
    )
}

pub fn ends_open(s: &str) -> String {
    let last = format!(
        "({} OR {})",
        unanswered_prompt(&format!("{s}.pk")),
        unfinished_tool_call(&format!("{s}.pk"))
    );
    let replied = names_record("reply.ref", "'annotation:'||a.root");
    format!(
        "(coalesce({last},0) OR json_extract({s}.live_facts,'$.todo') IS NOT NULL OR EXISTS(SELECT 1 FROM source_annotation a JOIN source_annotation_link l ON l.annotation=a.id AND l.role='about' JOIN source_revision_state ar ON ar.id=a.id AND ar.current=1 AND ar.withdrawn=0 WHERE l.ref='session:'||{s}.id AND ((a.kind='outcome' AND (json_extract(a.value,'$')='partial' OR json_extract(a.value,'$.text')='partial')) OR (a.kind='handoff' AND NOT EXISTS(SELECT 1 FROM source_annotation_link reply JOIN source_revision_state rr ON rr.id=reply.annotation AND rr.current=1 AND rr.withdrawn=0 WHERE reply.role ='replies_to' AND {replied})))))"
    )
}

pub fn unrecovered_failures(s: &str) -> String {
    format!(
        "(SELECT count(*) FROM (SELECT t.command_norm FROM tool_call t WHERE t.session_pk={s}.pk AND t.command_norm IS NOT NULL AND t.outcome='error' AND NOT EXISTS(SELECT 1 FROM tool_call later WHERE later.session_pk=t.session_pk AND later.command_norm=t.command_norm AND later.outcome='ok' AND (coalesce(later.at,0),later.pk)>(coalesce(t.at,0),t.pk)) GROUP BY t.command_norm))"
    )
}
pub fn failure_streak(s: &str) -> String {
    format!(
        "(SELECT coalesce(max(streak),0) FROM (SELECT count(*) AS streak FROM (SELECT t.command_norm,t.outcome,row_number() OVER(PARTITION BY t.command_norm ORDER BY coalesce(t.at,0),t.pk)-row_number() OVER(PARTITION BY t.command_norm,t.outcome ORDER BY coalesce(t.at,0),t.pk) AS run FROM tool_call t WHERE t.session_pk={s}.pk AND t.command_norm IS NOT NULL) WHERE outcome='error' GROUP BY command_norm,run))"
    )
}
pub fn corrections(s: &str) -> String {
    format!(
        "(SELECT count(DISTINCT a.id) FROM source_annotation a JOIN source_revision_state ar ON ar.id=a.id AND ar.current=1 AND ar.withdrawn=0 WHERE a.kind='correction' AND a.author='operator' AND {})",
        about_session("a", s)
    )
}
pub fn open_items(s: &str) -> String {
    format!(
        "(SELECT count(*) FROM source_annotation a WHERE a.kind IN ('question','problem') AND {} AND {})",
        open_annotation("a"),
        about_session("a", s)
    )
}
fn own_cost(s: &str) -> String {
    crate::measure::cost_of(&format!("{s}.pk"))
}
/// Compare only the same repository's preceding thirty days, never future work.
pub fn outlier(s: &str) -> String {
    let distribution = |value: String| {
        format!(
            "(SELECT avg(value) FROM (SELECT value,row_number() OVER(ORDER BY value) AS n,count(*) OVER() AS total FROM (SELECT {value} AS value FROM session previous WHERE (({s}.repo_root IS NOT NULL AND previous.repo_root={s}.repo_root) OR ({s}.repo_root IS NULL AND {s}.repo IS NOT NULL AND previous.repo={s}.repo)) AND previous.last_activity_at>={s}.last_activity_at-2592000000 AND previous.last_activity_at<{s}.last_activity_at) WHERE value IS NOT NULL) WHERE n IN ((total+1)/2,(total+2)/2))"
        )
    };
    let cost = distribution(own_cost("previous"));
    let wall = distribution(span("previous", "wall_ms"));
    format!(
        "(coalesce(({})>=3*{cost} AND {cost}>0,0) OR coalesce(({})>=3*{wall} AND {wall}>0,0))",
        own_cost(s),
        span(s, "wall_ms")
    )
}
pub fn triage_score(s: &str) -> String {
    format!(
        "CASE WHEN EXISTS(SELECT 1 FROM source_annotation a JOIN source_annotation_link l ON l.annotation=a.id AND l.role='about' JOIN source_revision_state ar ON ar.id=a.id AND ar.current=1 AND ar.withdrawn=0 WHERE a.kind='tag' AND a.key='triaged' AND l.ref='session:'||{s}.id AND a.created_at>{s}.last_activity_at) THEN 0 ELSE {}+({}>=3)+{}+({}>=3)+{} END",
        unrecovered_failures(s),
        failure_streak(s),
        outlier(s),
        corrections(s),
        open_items(s)
    )
}

/// The message population of the reader's recorded parent prefixes, including own rows.
/// Limits compare the position of a branch point in the composed ancestry rather
/// than treating an ancestor's entire history as inherited.
pub fn parent_prefix_messages(s: &str) -> String {
    format!(
        "WITH RECURSIVE ancestors(pk,id,depth,path,child_id,point,copied) AS (SELECT {s}.pk,{s}.id,0,'/'||{s}.pk||'/',NULL,NULL,0 UNION ALL SELECT p.pk,p.id,a.depth+1,a.path||p.pk||'/',a.id,l.branch_point,(EXISTS(SELECT 1 FROM prefix_copy pc WHERE pc.child_id=a.id AND pc.parent_id=p.id AND pc.explicit_members=1) OR l.copies_prefix=1 AND l.branch_point IS NULL) FROM ancestors a JOIN session_prefix l ON l.child_id=a.id JOIN session p ON p.id=l.parent_id AND p.stub=0 WHERE a.copied=0 AND instr(a.path,'/'||p.pk||'/')=0 AND (l.branch_point IS NULL OR (SELECT count(*) FROM session_prefix conflicting WHERE conflicting.child_id=a.id AND conflicting.branch_point IS NOT NULL)=1)), candidates AS (SELECT a.*,m.pk AS message_pk,m.id AS message_id,m.session_order FROM ancestors a JOIN message m ON m.session_pk=a.pk WHERE {}) SELECT DISTINCT c.message_pk FROM candidates c WHERE c.depth=0 OR NOT EXISTS(SELECT 1 FROM ancestors edge WHERE edge.depth>0 AND instr(c.path,edge.path)=1 AND ((edge.copied=1 AND NOT(EXISTS(SELECT 1 FROM prefix_member pm WHERE pm.child_id=edge.child_id AND pm.parent_id=edge.id AND pm.message_id=c.message_id) OR EXISTS(SELECT 1 FROM message_parse ms JOIN prefix_copy pc ON pc.input_pk=ms.input_pk AND pc.chain=ms.chain AND pc.explicit_members=0 WHERE ms.message_id=(SELECT id FROM message WHERE pk=c.message_pk) AND pc.child_id=edge.child_id AND pc.parent_id=edge.id AND NOT EXISTS(SELECT 1 FROM prefix_copy explicit WHERE explicit.child_id=pc.child_id AND explicit.parent_id=pc.parent_id AND explicit.input_pk=pc.input_pk AND explicit.chain=pc.chain AND explicit.explicit_members=1)))) OR (edge.copied=0 AND NOT EXISTS(SELECT 1 FROM candidates point WHERE point.message_id=edge.point AND instr(point.path,edge.path)=1 AND (c.depth>point.depth OR c.depth=point.depth AND (c.session_order,c.message_id)<=(point.session_order,point.message_id))))))",
        own_messages("m", None)
    )
}
pub fn cost_with_parents(s: &str) -> String {
    format!(
        "(SELECT sum(m.cost_nusd) FROM message m WHERE m.pk IN ({}) AND {})",
        parent_prefix_messages(s),
        crate::measure::RESPONSE_ROWS
    )
}

pub fn cost_complete(s: &str) -> String {
    format!(
        "(NOT EXISTS(SELECT 1 FROM message m WHERE m.session_pk={s}.pk AND m.duplicate_of IS NULL AND {}) AND NOT EXISTS(SELECT 1 FROM event e WHERE e.session_pk={s}.pk AND e.kind='response_boundaries_unknown'))",
        response_cost_unknown()
    )
}

pub(crate) fn response_value(column: &str) -> String {
    format!("CASE WHEN {} THEN m.{column} END", own_responses())
}

pub(crate) fn response_cost_unknown() -> String {
    format!(
        "((m.role='assistant' AND m.written_by='model' AND (m.cost_nusd IS NULL OR NOT ({}))) OR EXISTS(SELECT 1 FROM event e WHERE e.session_pk=m.session_pk AND e.kind='response_boundaries_unknown'))",
        own_responses()
    )
}

pub(crate) const RESPONSE_ROWS: &str = "m.role = 'assistant' AND m.written_by = 'model'
 AND coalesce(json_extract(m.attrs_json, '$.production.response_boundary_unknown'), 0) = 0
 AND m.session_pk NOT IN (SELECT e.session_pk FROM event e WHERE e.kind='response_boundaries_unknown' AND e.session_pk IS NOT NULL)";

/// The `own` population (docs/measures.md, populations): each recorded
/// response once, its shown version, on and off the active path. Every
/// cost and token figure sums over it.
pub(crate) fn own_responses() -> String {
    format!("m.duplicate_of IS NULL AND {RESPONSE_ROWS}")
}

pub(crate) fn response_unpriced() -> String {
    format!("({} AND m.cost_nusd IS NULL)", own_responses())
}

/// The sum of `column` over the own responses of the sessions `sessions`.
pub(crate) fn own_responses_sum(sessions: &str, column: &str) -> String {
    format!(
        "(SELECT sum({}) FROM message m WHERE m.session_pk IN ({sessions}))",
        response_value(column)
    )
}

/// Whether an own response of the sessions `sessions` has no price: the
/// session cost is then a lower bound.
pub(crate) fn unpriced_in(sessions: &str) -> String {
    format!(
        "EXISTS (SELECT 1 FROM message m WHERE m.session_pk IN ({sessions}) AND {})",
        response_unpriced()
    )
}

/// The one session cost: the known cost of the own responses of the
/// sessions `sessions` (an SQL set of session keys); none when no response
/// is priced. The field, `agg`, `compare` and `usage` all read it.
pub(crate) fn cost_of(sessions: &str) -> String {
    own_responses_sum(sessions, "cost_nusd")
}

/// A stored message timestamp, for reads that take millisecond bounds.
pub fn message_time_ms(at: jiff::Timestamp) -> i64 {
    at.as_millisecond()
}

#[cfg(test)]
mod tests;
