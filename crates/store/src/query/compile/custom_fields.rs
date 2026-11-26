//! Fields compiled by a rule of their own rather than from the field
//! table's SQL: each needs a subquery, a parameter shape, or a check
//! against other conditions. A special field in [`crate::query::fields::FIELDS`]
//! names its rule here, so a field without one cannot be declared.

use super::*;
use polylogue_model::{CallOutcome, LINK_HOW_NAMES, LINK_KIND_NAMES};
use polylogue_query::filter::{Special, Test, TimeTest, closed_set_message};

/// A special field's compile rule.
pub(crate) type Custom = fn(&mut Compiler<'_>, &Call<'_, '_, '_>) -> Result<String, QueryError>;

/// One condition on a special field, as its rule receives it.
pub(crate) struct Call<'q, 's, 'l> {
    pub unit: Unit,
    pub name: &'static str,
    /// The value as binding gave it: a special value, or for `since:`,
    /// `until:` and `seen_by:` the time it resolved.
    pub test: &'q Test,
    pub scope: &'s RowContext,
    pub span: Span,
    /// Where the condition stands.
    pub cx: Cx<'l>,
}

impl Call<'_, '_, '_> {
    /// The special field's bound value.
    fn special(&self) -> Option<&Special> {
        match self.test {
            Test::Special(special) => Some(special),
            _ => None,
        }
    }

    fn one(&self) -> Option<String> {
        match self.special() {
            Some(Special::One { text, .. }) => Some(text.clone()),
            _ => None,
        }
    }

    /// The time a `since:`, `until:` or `seen_by:` condition names.
    fn time(&self, c: &Compiler<'_>) -> Result<Time, QueryError> {
        match self.test {
            Test::Time(TimeTest::Within(at)) => Ok(c.resolved(at)),
            _ => Err(self.need_value()),
        }
    }

    /// The single value, or the refusal naming the field's form.
    fn value(&self) -> Result<String, QueryError> {
        self.one().ok_or_else(|| self.need_value())
    }

    fn need_value(&self) -> QueryError {
        QueryError::Invalid {
            message: format!("{0} takes {0}:VALUE", self.name),
            position: self.span.start,
        }
    }

    fn values(&self) -> Vec<String> {
        self.special().map(Special::listed).unwrap_or_default()
    }

    fn unknown(&self) -> bool {
        matches!(self.special(), Some(Special::Unknown))
    }

    fn bad(&self, value: String, message: String) -> QueryError {
        QueryError::BadValue {
            field: self.name.into(),
            value,
            position: self.span.start,
            message,
        }
    }

    fn s(&self) -> &str {
        &self.scope.a.s
    }

    fn row(&self) -> &str {
        &self.scope.a.row
    }
}

/// `physical:true` lifts logical-session folding; false is the default form.
pub(crate) fn physical(_: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    match x.test {
        Test::Bool(true) => Ok("1".into()),
        Test::Bool(false) => Err(x.bad(
            "false".into(),
            "physical:false is the default listing; leave the condition out".into(),
        )),
        _ => Err(x.need_value()),
    }
}

/// `(… OR …)` of the parts, refusing an empty list.
fn any_of(x: &Call<'_, '_, '_>, parts: Vec<String>) -> Result<String, QueryError> {
    if parts.is_empty() {
        return Err(x.need_value());
    }
    Ok(format!("({})", parts.join(" OR ")))
}

/// A `similar:` term the selection did not take as its ranking: under OR or
/// NOT, in a stage, a second one, or in a nested selection.
pub(crate) fn near(_: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    Err(similar_misplaced(x.span))
}

pub(super) fn similar_misplaced(span: Span) -> QueryError {
    QueryError::Invalid {
        message:
            "similar: ranks the whole query, so it appears once among the top-level conditions"
                .into(),
        position: span.start,
    }
}

impl Compiler<'_> {
    /// The similarities of the rows `where_sql` selects from `rows` to the
    /// `similar:` text, ranking by `scope`'s unit. A text's similarity is
    /// looked up for the selected rows' texts, never for the archive's
    /// (docs/search.md: Forms and meaning).
    pub(super) fn similarity(
        &mut self,
        near: &Near,
        scope: &RowContext,
        rows: &RowContext,
        where_sql: &[String],
    ) -> ranking::Similarity {
        let a = &scope.a;
        let selected = format!(
            "{} WHERE {}",
            base_from(rows.unit, &rows.a),
            and_all(where_sql)
        );
        // A message's similarity is its best text's.
        let best = |messages: &str| {
            format!(
                "(SELECT max(pl_similarity(xt.hash)) FROM block xb JOIN text xt ON xt.id = xb.text_id \
                 WHERE xb.message_pk {messages})"
            )
        };
        let (expression, join) = match scope.unit {
            Unit::Sessions => {
                // Each selected session's best message across its logical session.
                let sessions = format!("n{}", self.ctes.len() + 1);
                self.ctes.push(format!(
                    "{sessions} AS MATERIALIZED (SELECT xs.pk AS p, {} AS s \
                     FROM (SELECT DISTINCT {}.pk AS pk FROM {selected}) xs)",
                    best(&format!(
                        "IN (SELECT xm.pk FROM message xm WHERE xm.session_pk IN ({}))",
                        fields::descendants("xs")
                    )),
                    a.s
                ));
                (
                    "-near_s.s".to_owned(),
                    Some(format!(
                        "LEFT JOIN {sessions} near_s ON near_s.p = {}.pk",
                        a.s
                    )),
                )
            }
            Unit::Messages => {
                let messages = format!("n{}", self.ctes.len() + 1);
                self.ctes.push(format!(
                    "{messages} AS MATERIALIZED (SELECT xs.pk AS m, {} AS s \
                     FROM (SELECT DISTINCT {}.pk AS pk FROM {selected}) xs)",
                    best("= xs.pk"),
                    a.m
                ));
                (
                    "-near_m.s".to_owned(),
                    Some(format!(
                        "LEFT JOIN {messages} near_m ON near_m.m = {}.pk",
                        a.m
                    )),
                )
            }
            Unit::Blocks => (
                format!(
                    "-(SELECT pl_similarity(xt.hash) FROM text xt WHERE xt.id = {}.text_id)",
                    a.b
                ),
                None,
            ),
            _ => unreachable!("near units are checked when requested"),
        };
        self.similar_text = Some(near.clone());
        ranking::Similarity { expression, join }
    }
}

pub(crate) fn output(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    words_in(c, x, Index::Output)
}

pub(crate) fn thinking(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    words_in(c, x, Index::Thinking)
}

fn words_in(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
    index: Index,
) -> Result<String, QueryError> {
    let text = x.value()?;
    let quoted = matches!(x.special(), Some(Special::One { quoted: true, .. }));
    c.text(&[(text, quoted, x.span)], x.scope, index, x.cx)
}

pub(crate) fn since(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    if matches!(x.special(), Some(Special::SinceSeen)) {
        return since_seen(c, x);
    }
    let t = x.time(c)?;
    if x.cx.is_top() && x.unit == Unit::Sessions {
        // Conditions joined by AND narrow: the latest start wins.
        let start = t.start_ms();
        c.session_window.0 = Some(c.session_window.0.map_or(start, |s| s.max(start)));
    }
    let window = units::spec(x.unit)
        .since
        .ok_or_else(|| c.unknown_field(x.name, x.unit, x.span))?;
    if let Some(stored) = fields::time_window(x.unit, true, &x.scope.a) {
        let test = c.time_test(
            &stored,
            time::TimePredicate::Overlap {
                since: Some(t.start()),
                until: None,
            },
        );
        return Ok(if x.unit == Unit::ToolCalls {
            format!("({test} OR {}.result_message_id IS NULL)", x.scope.a.t)
        } else {
            test
        });
    }
    Ok(x.scope
        .a
        .expand(window)
        .replace("{v}", &t.start_ms().to_string()))
}

pub(crate) fn until(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let t = x.time(c)?;
    if x.cx.is_top() && x.unit == Unit::Sessions {
        let bound = crate::calendar::boundary_ms(t.until_bound());
        c.session_window.1 = Some(c.session_window.1.map_or(bound, |u| u.min(bound)));
    }
    let window = units::spec(x.unit)
        .until
        .ok_or_else(|| c.unknown_field(x.name, x.unit, x.span))?;
    if let Some(stored) = fields::time_window(x.unit, false, &x.scope.a) {
        return Ok(c.time_test(
            &stored,
            time::TimePredicate::Overlap {
                since: None,
                until: Some(t.until_bound()),
            },
        ));
    }
    let bound = crate::calendar::boundary_ms(t.until_bound());
    Ok(x.scope.a.expand(window).replace("{v}", &bound.to_string()))
}

/// `seen_by:T`: what Polylogue had seen by T, read with today's
/// parsers and prices. Rows count when their bytes were first
/// seen by then; notes as each family stood then.
pub(crate) fn seen_by(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let end = x.time(c)?.end_ms()?;
    let a = &x.scope.a;
    let cutoff = match x.unit {
        Unit::Messages | Unit::Blocks | Unit::ToolCalls => {
            format!("{} < {end}", fields::first_seen_of(&a.m))
        }
        Unit::Events => format!("{} < {end}", fields::event_first_seen(a)),
        Unit::Annotations => {
            if !c.annotation_cutoffs.contains(&end) {
                c.annotation_cutoffs.push(end);
            }
            format!(
                "EXISTS(SELECT 1 FROM temp.annotation_history ah WHERE ah.cutoff={end} AND ah.id={r}.id AND ah.root={r}.root AND ah.current=1)",
                r = x.row()
            )
        }
        // A session's counts, cost, times and title as they stood then
        // would each need recomputing from those rows; the rows of that
        // time answer the question instead.
        other => {
            return Err(QueryError::Invalid {
                message: format!(
                    "seen_by asks for rows as Polylogue had them; figures of {} are not recomputed \
                     as they stood then: ask for the session's messages, tool calls or notes \
                     (`messages where session:ID seen_by:…`)",
                    other.name()
                ),
                position: x.span.start,
            });
        }
    };
    let note = "seen_by: the rows Polylogue had by then, read with today's parsers and prices";
    if !c.notes.iter().any(|n| n == note) {
        c.notes.push(note.into());
    }
    Ok(cutoff)
}

pub(crate) fn changes_after(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
) -> Result<String, QueryError> {
    let position = x
        .one()
        .and_then(|text| polylogue_model::ChangePosition::parse(&text));
    let Some(position) = position else {
        return Err(QueryError::Invalid {
            message: "after: takes a feed position `GENERATION:INDEX:ARCHIVE`".into(),
            position: x.span.start,
        });
    };
    c.change_generations.push(position.generation);
    Ok(format!(
        "(({r}.kind = 'annotation' AND {r}.seq > {archive}) OR ({r}.kind = 'removed' AND {r}.seq > {index}) \
         OR ({r}.kind = 'session' AND ({r}.changed > {index} OR {r}.live_changed > {index} OR {r}.derived_changed > {index})))",
        r = x.row(),
        index = position.index,
        archive = position.archive,
    ))
}

pub(crate) fn repo(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let s = x.s();
    if x.unknown() {
        return Ok(format!("({s}.repo IS NULL AND {s}.repo_root IS NULL)"));
    }
    let mut parts = vec![];
    for v in x.values() {
        let v = v.to_lowercase().trim_end_matches('/').to_owned();
        if v == "here" {
            parts.push(here(c, x, false)?);
            continue;
        }
        let p = c.text_param(&v);
        let tail = c.text_param(&format!("%/{}", like_escape(&v)));
        let tail_git = c.text_param(&format!("%/{}.git", like_escape(&v)));
        parts.push(format!(
            "CASE WHEN {s}.repo IS NOT NULL OR {s}.repo_root IS NOT NULL THEN \
             (COALESCE(lower(rtrim({s}.repo, '/')) = {p} OR lower(rtrim({s}.repo, '/')) LIKE {tail} ESCAPE '\\' \
             OR lower({s}.repo) LIKE {tail_git} ESCAPE '\\', 0) OR COALESCE({s}.repo_root = {p}, 0)) END"
        ));
    }
    // Ordinary repository names remain unknown without repository metadata.
    // `here` also has indexed working-directory evidence and stands on its own.
    any_of(x, parts)
}

pub(crate) fn live(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let mut states = vec![];
    for v in x.values() {
        let v = v.to_ascii_lowercase();
        if !fields::LIVE_STATES.contains(&v.as_str()) {
            let message = format!("live is one of {}", fields::LIVE_STATES.join(", "));
            return Err(x.bad(v, message));
        }
        states.push(format!("'{v}'"));
    }
    if states.is_empty() {
        return Err(x.need_value());
    }
    Ok(format!(
        "{} IN ({})",
        fields::live_state(x.s(), c.opts.clock.now.as_millisecond()),
        states.join(", ")
    ))
}

pub(crate) fn waiting(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    live_flag(c, x, "('waiting')")
}

pub(crate) fn running(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    live_flag(c, x, &fields::running_states_sql())
}

/// A true/false field: the live state is one of `states`.
fn live_flag(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
    states: &str,
) -> Result<String, QueryError> {
    let v = x.value()?;
    let want = match v.to_ascii_lowercase().as_str() {
        "true" | "yes" | "1" => true,
        "false" | "no" | "0" => false,
        _ => {
            let message = format!("{} is true or false", x.name);
            return Err(x.bad(v, message));
        }
    };
    let not = if want { "" } else { "NOT " };
    Ok(format!(
        "{} {not}IN {states}",
        fields::live_state(x.s(), c.opts.clock.now.as_millisecond())
    ))
}

pub(crate) fn commit(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let v = x.value()?;
    if v.len() < 7 || !v.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(x.bad(
            v,
            "commit takes a hash prefix of at least 7 hex digits".into(),
        ));
    }
    let p = c.text_param(&format!("{}%", v.to_lowercase()));
    let q = c.text_param(&v.to_lowercase());
    // A hash is stored as the command printed it, often short: either may
    // be a prefix of the other.
    Ok(format!(
        "EXISTS (SELECT 1 FROM commit_link cl WHERE cl.session_pk = {}.pk \
         AND (cl.commit_hash LIKE {p} OR {q} LIKE cl.commit_hash || '%'))",
        x.s()
    ))
}

pub(crate) fn pr(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let v = x.value()?;
    let Some((repo, number)) = parse_pr(&v) else {
        return Err(x.bad(
            v,
            "pr takes a number, owner/name#N, or a pull request URL".into(),
        ));
    };
    let s = x.s();
    let n = c.bind(Param::Int(number));
    Ok(match repo {
        Some(repo) => {
            let p = c.text_param(&repo.to_lowercase());
            format!(
                "EXISTS (SELECT 1 FROM pr_link pl WHERE pl.session_pk = {s}.pk AND pl.number = {n} AND lower(pl.repo) = {p})"
            )
        }
        None => format!(
            "EXISTS (SELECT 1 FROM pr_link pl WHERE pl.session_pk = {s}.pk AND pl.number = {n})"
        ),
    })
}

pub(crate) fn cwd(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let s = x.s();
    let mut parts = vec![];
    for v in x.values() {
        if v.eq_ignore_ascii_case("here") {
            parts.push(here(c, x, true)?);
            continue;
        }
        // A key range over every directory the session recorded, its last
        // one included.
        let range = c.directory_range("session_cwd_member.cwd", &v);
        parts.push(format!(
            "{s}.pk IN (SELECT session_cwd_member.session_pk FROM session_cwd_member WHERE {range})"
        ));
    }
    any_of(x, parts)
}

pub(crate) fn model(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let s = x.s();
    if x.unknown() {
        return Ok(format!(
            "NOT EXISTS (SELECT 1 FROM message x WHERE x.session_pk = {s}.pk AND x.model IS NOT NULL)"
        ));
    }
    let vs: Vec<String> = x
        .values()
        .iter()
        .map(|v| c.text_param(&v.to_lowercase()))
        .collect();
    if vs.is_empty() {
        return Err(x.need_value());
    }
    // Unknown when no message records a model, so neither the condition
    // nor its NOT matches (three-valued).
    Ok(format!(
        "CASE WHEN EXISTS (SELECT 1 FROM message x WHERE x.session_pk = {s}.pk AND x.model IS NOT NULL) \
         THEN EXISTS (SELECT 1 FROM message x WHERE x.session_pk = {s}.pk AND lower(x.model) IN ({})) END",
        vs.join(", ")
    ))
}

/// `outcome:` compares the stored column through `NULLIF(outcome,
/// 'unknown')`, the expression `tool_call_action` indexes, so a stored
/// `unknown` stays unknown (neither the condition nor its NOT matches).
/// Only `pending`, a stored `unknown` call the session's live state names
/// open, is decided beside it.
pub(crate) fn tool_call_outcome(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
) -> Result<String, QueryError> {
    let a = &x.scope.a;
    let t = &a.t;
    let pending = a.expand(fields::OPEN_CALL);
    let unknown = format!("({t}.outcome = 'unknown' AND NOT {pending})");
    let def = fields::FieldId::ToolCallsOutcome.declaration();
    let (values, mut with_unknown) = match x.test {
        Test::Unknown => return Ok(unknown),
        Test::Known => return Ok(format!("NOT {unknown}")),
        Test::OneOf { values, unknown } => (values.clone(), *unknown),
        Test::TextCompare(Op::Eq, value) => (vec![value.clone()], false),
        Test::TextCompare(Op::Ne, value) => {
            let equal = Test::TextCompare(Op::Eq, value.clone());
            let equal = Call { test: &equal, ..*x };
            return Ok(format!("NOT {}", tool_call_outcome(c, &equal)?));
        }
        test => return c.typed(def, def.kind, &fields::call_outcome(a), test),
    };
    let mut listed = vec![];
    let mut with_pending = false;
    for value in &values {
        match CallOutcome::NAMES
            .iter()
            .find(|name| name.eq_ignore_ascii_case(value))
            .copied()
        {
            Some("unknown") => with_unknown = true,
            Some(name) => {
                with_pending |= name == "pending";
                listed.push(format!("'{name}'"));
            }
            None => return Err(x.bad(value.clone(), closed_set_message(def, value))),
        }
    }
    // `pending` is never stored, so listing it leaves a stored `unknown`
    // unknown; the open call itself matches through the live state.
    let stored = (!listed.is_empty()).then(|| {
        let known = format!("NULLIF({t}.outcome, 'unknown') IN ({})", listed.join(", "));
        if with_pending {
            format!("({known} OR {pending})")
        } else {
            format!("({known} AND NOT {pending})")
        }
    });
    match (stored, with_unknown) {
        (Some(stored), false) => Ok(stored),
        (Some(stored), true) => Ok(format!("({stored} OR {unknown})")),
        (None, true) => Ok(unknown),
        (None, false) => Err(x.need_value()),
    }
}

pub(crate) fn tag(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let s = x.s();
    let mut parts = vec![];
    for v in x.values() {
        let p = c.text_param(&v.to_lowercase());
        parts.push(format!(
            "EXISTS (SELECT 1 FROM source_annotation tn JOIN source_annotation_link tt ON tt.record = tn.id AND tt.role = 'about' \
             WHERE tn.kind = 'tag' AND tt.ref = 'session:' || {s}.id AND {} \
             AND ((tn.key IS NULL AND lower(json_extract(tn.value,'$.value')) = {p}) \
               OR lower(tn.key) || ':' || lower(json_extract(tn.value,'$.value')) = {p} \
               OR (json_extract(tn.value,'$.value') IS NULL AND lower(tn.key) = {p})))",
            fields::current_annotation("tn")
        ));
    }
    any_of(x, parts)
}

pub(crate) fn has(_: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let (table, alias) = if x.unit == Unit::Sessions {
        (&fields::SESSION_HAS[..], x.s())
    } else {
        (&fields::MESSAGE_HAS[..], x.scope.a.m.as_str())
    };
    let mut parts = vec![];
    for v in x.values() {
        let Some(has) = table.iter().find(|h| h.name.eq_ignore_ascii_case(&v)) else {
            let names: Vec<&str> = table.iter().map(|h| h.name).collect();
            let message = format!(
                "has on {} is one of {}; not {v:?}",
                x.unit.name(),
                names.join(", ")
            );
            return Err(x.bad(v, message));
        };
        parts.push((has.sql)(alias));
    }
    any_of(x, parts)
}

pub(crate) fn parent(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let s = x.s();
    let v = x.value()?;
    if v.eq_ignore_ascii_case("missing") {
        return Ok(format!(
            "EXISTS (SELECT 1 FROM link l WHERE l.child_id = {s}.id AND NOT EXISTS (SELECT 1 FROM session lp WHERE lp.id = l.parent_id))"
        ));
    }
    let reference = c.session_reference("l.parent_id", "id", &v);
    Ok(format!(
        "EXISTS (SELECT 1 FROM link l WHERE l.child_id = {s}.id AND {reference})"
    ))
}

pub(crate) fn root(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let root = crate::lineage::lineage_column_sql(&format!("{}.pk", x.s()), "root");
    if x.unknown() {
        return Ok(format!("{root} IS NULL"));
    }
    if matches!(x.special(), Some(Special::Known)) {
        return Ok(format!("{root} IS NOT NULL"));
    }
    Ok(c.session_reference(&root, "id", &x.value()?))
}

pub(crate) fn link(_: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let s = x.s();
    let mut parts = vec![];
    for v in x.values() {
        let v = v.to_ascii_lowercase();
        let column = if LINK_KIND_NAMES.contains(&v.as_str()) {
            "kind"
        } else if LINK_HOW_NAMES.contains(&v.as_str()) {
            "how"
        } else {
            let message = format!(
                "link is one of {}, or how it was made: {}; not {v:?}",
                LINK_KIND_NAMES.join(", "),
                LINK_HOW_NAMES.join(", ")
            );
            return Err(x.bad(v, message));
        };
        parts.push(format!(
            "EXISTS (SELECT 1 FROM link l WHERE l.child_id = {s}.id AND l.{column} = '{v}')"
        ));
    }
    any_of(x, parts)
}

pub(crate) fn continues(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let reference = c.session_reference("l.parent_id", "id", &x.value()?);
    Ok(format!(
        "EXISTS (SELECT 1 FROM link l WHERE l.child_id = {}.id AND l.kind IN ('fork','continuation','resume') AND {reference})",
        x.s()
    ))
}

pub(crate) fn continued_by(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
) -> Result<String, QueryError> {
    let reference = c.session_reference("lc.id", "id", &x.value()?);
    Ok(format!(
        "EXISTS (SELECT 1 FROM link l JOIN session lc ON lc.id = l.child_id WHERE l.parent_id = {}.id \
         AND l.kind IN ('fork','continuation','resume') AND {reference})",
        x.s()
    ))
}

pub(crate) fn connected(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let reference = c.session_reference("rs.id", "id", &x.value()?);
    Ok(format!(
        "{}.id IN (WITH RECURSIVE comp(id) AS (SELECT rs.id FROM session rs WHERE {reference} \
         UNION SELECT CASE WHEN l.parent_id = comp.id THEN lc.id ELSE l.parent_id END \
         FROM link l JOIN session lc ON lc.id = l.child_id JOIN comp ON comp.id = l.parent_id OR comp.id = lc.id) \
         SELECT id FROM comp)",
        x.s()
    ))
}

/// `path:`, `tool:` and `action:` on sessions: the session made such a tool
/// call.
pub(crate) fn session_tool_call(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
) -> Result<String, QueryError> {
    let inner = RowContext {
        unit: Unit::ToolCalls,
        a: Aliases {
            t: "tq".into(),
            ..x.scope.a.clone()
        },
    };
    let Some(Special::ToolCalls(filter)) = x.special() else {
        return Err(x.need_value());
    };
    let cond = c.lower(filter, &inner, x.cx.inner())?;
    Ok(format!(
        "EXISTS (SELECT 1 FROM tool_call tq WHERE tq.session_pk = {}.pk AND {cond})",
        x.s()
    ))
}

/// `command:` and `path:` on tool calls: a substring, through the trigram
/// index from three characters.
pub(crate) fn tool_call_text(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
) -> Result<String, QueryError> {
    let t = &x.scope.a.t;
    let name = x.name;
    if x.unknown() {
        return Ok(format!("{t}.{name} IS NULL"));
    }
    let mut parts = vec![];
    for v in x.values() {
        // A full-text query ends at a NUL, so such a value is compared as text.
        if v.chars().count() >= 3 && !v.contains('\0') {
            let phrase = format!("\"{}\"", v.replace('"', "\"\""));
            let p = c.text_param(&phrase);
            // A call without the value is unknown for the condition, as
            // the shorter values' `instr` already makes it.
            parts.push(format!(
                "CASE WHEN {t}.{name} IS NOT NULL THEN {t}.pk IN (SELECT rowid FROM tool_call_trigram WHERE tool_call_trigram MATCH '{name}:' || {p}) END"
            ));
        } else {
            let p = c.text_param(&v.to_lowercase());
            parts.push(format!("instr(lower({t}.{name}), {p}) > 0"));
        }
    }
    any_of(x, parts)
}

pub(crate) fn input(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let p = c.text_param(&x.value()?);
    Ok(format!(
        "instr(pl_key(json_extract({}, '$.input')), pl_key({p})) > 0",
        crate::content::sql_for(crate::content::BODY_SQL, &x.scope.a.cb)
    ))
}

pub(crate) fn mcp_server(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let vs: Vec<String> = x
        .values()
        .iter()
        .map(|v| c.text_param(&v.to_lowercase()))
        .collect();
    if vs.is_empty() {
        return Err(x.need_value());
    }
    Ok(format!(
        "({t}.tool LIKE 'mcp\\_\\_%' ESCAPE '\\' AND instr(substr({t}.tool, 6), '__') > 0 AND lower(substr({t}.tool, 6, instr(substr({t}.tool, 6), '__') - 1)) IN ({}))",
        vs.join(", "),
        t = x.scope.a.t
    ))
}

pub(crate) fn spawned_child(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
) -> Result<String, QueryError> {
    let reference = c.session_reference("lcs.id", "id", &x.value()?);
    let a = &x.scope.a;
    Ok(format!(
        "EXISTS (SELECT 1 FROM link lc JOIN session lcs ON lcs.id = lc.child_id WHERE lc.kind = 'spawn' \
         AND lc.parent_id = {s}.id AND lc.spawning_call = {t}.call_id \
         AND {reference})",
        s = a.s,
        t = a.t,
    ))
}

pub(crate) fn adjacent_call(
    _: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
) -> Result<String, QueryError> {
    Err(QueryError::Invalid {
        message: format!(
            "{0} names a relation; filter on {0}.<field>, such as {0}.action:shell",
            x.name
        ),
        position: x.span.start,
    })
}

/// A field naming a session by ID or unique prefix in a column of the row.
fn session_id_in(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
    column: &str,
) -> Result<String, QueryError> {
    let column = x.scope.a.expand(column);
    Ok(c.session_reference(&column, "id", &x.value()?))
}

pub(crate) fn live_session(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
) -> Result<String, QueryError> {
    session_id_in(c, x, "{s}.id")
}

pub(crate) fn link_parent(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
) -> Result<String, QueryError> {
    session_id_in(c, x, "{r}.parent_id")
}

pub(crate) fn link_child(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    session_id_in(c, x, "{r}c.id")
}

pub(crate) fn recipient(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    session_id_in(c, x, "{r}.recipient_id")
}

/// A note about the reference; a bare session ID is a `session:` reference.
pub(crate) fn annotation_about(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
) -> Result<String, QueryError> {
    let input = x.value()?;
    let reference =
        polylogue_model::Reference::parse(&input).map_err(|error| QueryError::Invalid {
            message: error.to_string(),
            position: x.span.start,
        })?;
    let p = c.text_param(&reference.to_string());
    Ok(format!(
        "EXISTS(SELECT 1 FROM source_annotation_link nt WHERE nt.record={r}.id AND nt.role='about' AND nt.ref={p})",
        r = x.row()
    ))
}

// json_each exposes reference strings as text; they are not JSON documents.
pub(crate) fn refs(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    delivery_member(c, x, "refs_json", "dj.value")
}

pub(crate) fn left_out(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    delivery_member(c, x, "left_out_json", "json_extract(dj.value, '$.what')")
}

fn delivery_member(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
    column: &str,
    member: &str,
) -> Result<String, QueryError> {
    let p = c.text_param(&x.value()?);
    Ok(format!(
        "EXISTS (SELECT 1 FROM json_each({r}.{column}) dj WHERE instr({member}, {p}) > 0)",
        r = x.row()
    ))
}

/// Stored selections are output data, not scalar query operands.
pub(crate) fn selection(_: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    Err(QueryError::Invalid {
        message: format!(
            "{} is a selection array; predicates on it are not supported",
            x.name
        ),
        position: x.span.start,
    })
}

pub(crate) fn event_text(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let parts = x.values().into_iter().map(|value| {
        let parameter = c.text_param(&value);
        format!("EXISTS(SELECT 1 FROM json_tree({}) event_text_value WHERE event_text_value.type='text' AND instr(pl_key(event_text_value.atom),pl_key({parameter}))>0)",crate::content::sql_for(crate::content::DATA_SQL, x.row()))
    }).collect();
    any_of(x, parts)
}

/// All references made by this annotation, excluding what it is about.
pub(crate) fn links_to(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let mut clauses = Vec::new();
    for value in x.values() {
        let reference = polylogue_model::Reference::parse(&value)
            .map_err(|error| x.bad(value.clone(), error.to_string()))?;
        let p = c.text_param(&reference.to_string());
        clauses.push(format!(
            "EXISTS(SELECT 1 FROM source_annotation_link l WHERE l.record={}.id AND l.role<>'about' AND {})",
            x.row(),
            crate::session_sql::names_record("l.ref", &p)
        ));
    }
    any_of(x, clauses)
}
pub(crate) fn request(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let values = match x.test {
        Test::Prefix(values) => values.clone(),
        _ => return Err(x.need_value()),
    };
    let mut clauses = Vec::new();
    for value in values {
        let p = c.text_param(&value);
        clauses.push(format!(
            "({r}.id={p} OR ({r}.id>{p}||'.' AND {r}.id<{p}||'/'))",
            r = x.row()
        ));
    }
    any_of(x, clauses)
}
/// Repository scope is captured from the caller when execution starts.
pub(crate) fn here(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
    cwd_only: bool,
) -> Result<String, QueryError> {
    let scope = c.opts.here.clone().ok_or_else(|| QueryError::Invalid {
        message: "here names the caller's repository, and this caller has no working directory"
            .into(),
        position: x.span.start,
    })?;
    let mut parts = Vec::new();
    for path in scope.paths {
        let range = c.directory_range("session_cwd_member.cwd", &path);
        parts.push(format!(
            "{}.pk IN (SELECT session_pk FROM session_cwd_member WHERE {range})",
            x.s()
        ));
    }
    if !cwd_only {
        for root in scope.repo_roots {
            let p = c.text_param(&root);
            parts.push(format!("coalesce({}.repo_root={p},0)", x.s()));
        }
    }
    any_of(x, parts)
}
fn since_seen(c: &mut Compiler<'_>, x: &Call<'_, '_, '_>) -> Result<String, QueryError> {
    let author = c.opts.seen_marks_of.clone().ok_or_else(|| {
        x.bad(
            "seen".into(),
            "since:seen needs the computed request author".into(),
        )
    })?;
    let author = c.text_param(&author);
    let target = match x.unit {
        Unit::Sessions => format!("'session:'||{}.id", x.s()),
        Unit::Annotations => format!("'annotation:'||{}.root", x.row()),
        _ => {
            return Err(x.bad(
                "seen".into(),
                "since:seen applies to sessions and annotations".into(),
            ));
        }
    };
    let at = "json_extract(mark.value,'$.position.at')";
    let index = format!(
        "CAST(substr({at},instr({at},':')+1,instr(substr({at},instr({at},':')+1),':')-1) AS INTEGER)"
    );
    let archive = format!(
        "CAST(substr(substr({at},instr({at},':')+1),instr(substr({at},instr({at},':')+1),':')+1) AS INTEGER)"
    );
    let unchanged = match x.unit {
        Unit::Sessions => format!(
            "substr({at},1,instr({at},':')-1)=(SELECT value FROM main.meta WHERE key='generation') AND {s}.changed<={index} AND COALESCE({s}.live_changed,0)<={index} AND COALESCE({s}.derived_changed,0)<={index}",
            s = x.s()
        ),
        Unit::Annotations => format!("{}.commit_seq<={archive}", x.row()),
        _ => unreachable!(),
    };
    Ok(format!(
        "NOT EXISTS(SELECT 1 FROM source_annotation mark JOIN source_revision_state ms ON ms.id=mark.id AND ms.current=1 AND ms.withdrawn=0 WHERE mark.kind='seen' AND mark.author={author} AND mark.key={target} AND {unchanged})"
    ))
}

pub(crate) fn alternate_lang(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
) -> Result<String, QueryError> {
    let mut clauses = Vec::new();
    for lang in x.values() {
        let p = c.text_param(&lang);
        clauses.push(format!("EXISTS(SELECT 1 FROM source_alternate_part vp JOIN source_alternate v ON v.id=vp.alternate WHERE v.to_lang={p} AND {} AND {})",crate::alternates::current_sql("v"),fields::message_alternate_match("v","vp",&x.scope.a.m)));
    }
    any_of(x, clauses)
}

pub(crate) fn run_outcome(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_, '_>,
) -> Result<String, QueryError> {
    const STATES: [&str; 2] = ["ok", "error"];
    let mut parts = Vec::new();
    for value in x.values() {
        let value = value.to_ascii_lowercase();
        if !STATES.contains(&value.as_str()) {
            let message = format!("outcome is one of {}", STATES.join(", "));
            return Err(x.bad(value, message));
        }
        let p = c.text_param(&value);
        parts.push(format!("json_extract({}.outcome,'$.state')={p}", x.row()));
    }
    any_of(x, parts)
}
