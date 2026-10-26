//! Fields compiled by a rule of their own rather than from the field
//! table's SQL: each needs a subquery, a parameter shape, or a check
//! against other conditions. A special field in [`crate::fields::FIELDS`]
//! names its rule here, so a field without one cannot be declared.

use super::*;
use polylogue_model::{LINK_HOW_NAMES, LINK_KIND_NAMES};

/// A special field's compile rule.
pub(crate) type Custom = fn(&mut Compiler<'_>, &Call<'_, '_>) -> Result<String, QueryError>;

/// One condition on a special field, as its rule receives it.
pub(crate) struct Call<'q, 's> {
    pub unit: Unit,
    pub name: &'static str,
    pub op: FieldOp<'q>,
    pub scope: &'s RowContext,
    pub span: Span,
    /// The condition is at the top level of the query.
    pub top: bool,
}

impl Call<'_, '_> {
    fn one(&self) -> Option<String> {
        match self.op {
            FieldOp::Match(Value::One { text, .. }) => Some(text.clone()),
            _ => None,
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
        match self.op {
            FieldOp::Match(v) => alternatives(v),
            _ => vec![],
        }
    }

    /// The bare, unquoted value `unknown`.
    fn unknown(&self) -> bool {
        matches!(self.op, FieldOp::Match(Value::One { text, quoted: false })
            if text.eq_ignore_ascii_case("unknown"))
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

/// `(… OR …)` of the parts, refusing an empty list.
fn any_of(x: &Call<'_, '_>, parts: Vec<String>) -> Result<String, QueryError> {
    if parts.is_empty() {
        return Err(x.need_value());
    }
    Ok(format!("({})", parts.join(" OR ")))
}

pub(crate) fn near(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    if !x.top {
        return Err(QueryError::Invalid {
            message:
                "near: ranks the whole query, so it appears once among the top-level conditions"
                    .into(),
            position: x.span.start,
        });
    }
    if !c.opts.embeddings {
        c.incomplete
            .push("semantic search is not enabled; rows are ordered by time and unranked".into());
        return Ok("1".into());
    }
    let text = x.value()?;
    // `pl_similarity` answers from the similarities the caller computed for
    // this text. Each message's best text is found once, in one pass over
    // the texts; a session ranks by the best message in it and its
    // subagents; a row with no vector has no rank. The similarities join
    // the unit's rows, so ranking and fusion read them as columns.
    let a = &x.scope.a;
    let messages = format!("n{}", c.ctes.len() + 1);
    c.ctes.push(format!(
        "{messages} AS MATERIALIZED (SELECT xb.message_pk AS m, max(xs.s) AS s \
         FROM (SELECT xt.id AS t, pl_similarity(xt.hash) AS s FROM text xt) xs \
         JOIN block xb ON xb.text_id = xs.t WHERE xs.s IS NOT NULL GROUP BY xb.message_pk)"
    ));
    let rank = match x.unit {
        Unit::Sessions => {
            // Each session's best message, carried up to the sessions that
            // spawned it, once per query.
            let tree = format!("n{}", c.ctes.len() + 1);
            c.ctes.push(format!(
                "{tree}(p, s) AS (SELECT xm.session_pk, max(nm.s) FROM {messages} nm \
                 JOIN message xm ON xm.pk = nm.m GROUP BY xm.session_pk \
                 UNION SELECT lp.pk, nt.s FROM {tree} nt JOIN link l ON l.child_pk = nt.p \
                 AND l.kind = 'subagent' JOIN session lp ON lp.id = l.parent_id)"
            ));
            let sessions = format!("n{}", c.ctes.len() + 1);
            c.ctes.push(format!(
                "{sessions} AS MATERIALIZED (SELECT p, max(s) AS s FROM {tree} GROUP BY p)"
            ));
            c.near_join = Some(format!(
                "LEFT JOIN {sessions} near_s ON near_s.p = {}.pk",
                a.s
            ));
            "-near_s.s".to_owned()
        }
        Unit::Messages => {
            c.near_join = Some(format!(
                "LEFT JOIN {messages} near_m ON near_m.m = {}.pk",
                a.m
            ));
            "-near_m.s".to_owned()
        }
        Unit::Blocks => format!(
            "-(SELECT pl_similarity(xt.hash) FROM text xt WHERE xt.id = {}.text_id)",
            a.b
        ),
        other => {
            c.ctes.pop();
            return Err(QueryError::Invalid {
                message: format!(
                    "near: ranks sessions, messages and blocks, not {}",
                    other.name()
                ),
                position: x.span.start,
            });
        }
    };
    c.near_rank = Some(rank);
    c.near = Some(Near { text });
    Ok("1".into())
}

pub(crate) fn output(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    words_in(c, x, Index::Output)
}

pub(crate) fn thinking(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    words_in(c, x, Index::Thinking)
}

fn words_in(c: &mut Compiler<'_>, x: &Call<'_, '_>, index: Index) -> Result<String, QueryError> {
    let text = x.value()?;
    let quoted = matches!(x.op, FieldOp::Match(Value::One { quoted: true, .. }));
    c.text(&[(text, quoted, x.span)], x.scope, index, x.top)
}

pub(crate) fn since(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let t = c.time(x.name, &x.value()?, x.span)?;
    let window = units::spec(x.unit)
        .since
        .ok_or_else(|| c.unknown_field(x.name, x.unit, x.span))?;
    Ok(x.scope
        .a
        .expand(window)
        .replace("{v}", &t.start().to_string()))
}

pub(crate) fn until(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let t = c.time(x.name, &x.value()?, x.span)?;
    let window = units::spec(x.unit)
        .until
        .ok_or_else(|| c.unknown_field(x.name, x.unit, x.span))?;
    Ok(x.scope
        .a
        .expand(window)
        .replace("{v}", &t.end().to_string()))
}

/// `evidence_until:T`: the evidence Polylogue had by T, read with today's
/// parsers, prices and excisions. Rows count when their bytes were first
/// seen by then; notes as each family stood then.
pub(crate) fn evidence_until(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let end = c.time(x.name, &x.value()?, x.span)?.end();
    let a = &x.scope.a;
    let cutoff = match x.unit {
        Unit::Messages | Unit::Blocks | Unit::ToolCalls => {
            format!("{} < {end}", fields::first_seen_of(&a.m))
        }
        Unit::Events => format!("{} < {end}", fields::event_first_seen(a)),
        Unit::Notes => format!(
            "({r}.created_at < {end} \
             AND NOT EXISTS (SELECT 1 FROM a.note later WHERE later.first_note = {r}.first_note \
               AND later.commit_seq > {r}.commit_seq AND later.created_at < {end}) \
             AND NOT EXISTS (SELECT 1 FROM a.note_relation nw JOIN a.note wn ON wn.id = nw.note_id \
               WHERE nw.kind = 'withdraws' AND nw.other = 'note:' || {r}.first_note AND wn.created_at < {end}))",
            r = x.row()
        ),
        // A session's counts, cost, times and title as they stood then
        // would each need recomputing from that evidence; the rows of that
        // time answer the question instead.
        other => {
            return Err(QueryError::Invalid {
                message: format!(
                    "evidence_until asks for rows as Polylogue had them; {} figures are not recomputed \
                     as they stood then: ask for the session's messages, tool calls or notes \
                     (`messages where session:ID evidence_until:…`)",
                    other.singular()
                ),
                position: x.span.start,
            });
        }
    };
    let note = "evidence_until: the rows Polylogue had by then, read with today's parsers, prices and excisions";
    if !c.notes.iter().any(|n| n == note) {
        c.notes.push(note.into());
    }
    Ok(cutoff)
}

pub(crate) fn changes_after(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let cursor = x
        .one()
        .and_then(|text| polylogue_model::ChangePosition::parse(&text));
    let Some(cursor) = cursor else {
        return Err(QueryError::Invalid {
            message: "after: takes a cursor `GEN:COUNTER:ARCHIVE`".into(),
            position: x.span.start,
        });
    };
    c.change_generations.push(cursor.generation);
    Ok(format!(
        "(({r}.kind IN ('note','excision') AND {r}.seq > {archive}) OR ({r}.kind = 'removed' AND {r}.seq > {index}) \
         OR ({r}.kind = 'session' AND ({r}.changed > {index} OR {r}.live_changed > {index})))",
        r = x.row(),
        index = cursor.index,
        archive = cursor.archive,
    ))
}

pub(crate) fn repo(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let s = x.s();
    if x.unknown() {
        return Ok(format!("({s}.repo IS NULL AND {s}.repo_root IS NULL)"));
    }
    let mut parts = vec![];
    for v in x.values() {
        let v = v.to_lowercase().trim_end_matches('/').to_owned();
        let p = c.text_param(&v);
        let tail = c.text_param(&format!("%/{}", like_escape(&v)));
        let tail_git = c.text_param(&format!("%/{}.git", like_escape(&v)));
        parts.push(format!(
            "(lower(rtrim({s}.repo, '/')) = {p} OR lower(rtrim({s}.repo, '/')) LIKE {tail} ESCAPE '\\' \
             OR lower({s}.repo) LIKE {tail_git} ESCAPE '\\' OR {s}.repo_root = {p})"
        ));
    }
    any_of(x, parts)
}

pub(crate) fn live(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
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
        fields::live_state(x.s(), c.opts.clock.now),
        states.join(", ")
    ))
}

pub(crate) fn waiting(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    live_flag(c, x, "('waiting')")
}

pub(crate) fn running(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    live_flag(c, x, "('working', 'waiting', 'idle', 'unknown')")
}

/// A true/false field: the live state is one of `states`.
fn live_flag(c: &mut Compiler<'_>, x: &Call<'_, '_>, states: &str) -> Result<String, QueryError> {
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
        fields::live_state(x.s(), c.opts.clock.now)
    ))
}

pub(crate) fn commit(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
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

pub(crate) fn pr(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
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

pub(crate) fn cwd(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let s = x.s();
    let mut parts = vec![];
    for v in x.values() {
        let a1 = c.subtree(&format!("{s}.cwd"), &v);
        let a2 = c.subtree("cj.value", &v);
        parts.push(format!(
            "({a1} OR EXISTS (SELECT 1 FROM json_each({s}.cwd_json) cj WHERE {a2}))"
        ));
    }
    any_of(x, parts)
}

pub(crate) fn model(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let s = x.s();
    if x.unknown() {
        return Ok(format!(
            "NOT EXISTS (SELECT 1 FROM message x WHERE x.session_pk = {s}.pk AND x.model IS NOT NULL)"
        ));
    }
    if matches!(x.op, FieldOp::Match(Value::Any)) {
        return Ok("1".into());
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

pub(crate) fn tag(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let s = x.s();
    let mut parts = vec![];
    for v in x.values() {
        let p = c.text_param(&v.to_lowercase());
        parts.push(format!(
            "EXISTS (SELECT 1 FROM a.note tn JOIN a.note_target tt ON tt.note_id = tn.id \
             WHERE tn.kind = 'tag' AND tt.ref = 'session:' || {s}.id AND {} \
             AND ((tn.key IS NULL AND lower(tn.value) = {p}) \
               OR lower(tn.key) || ':' || lower(tn.value) = {p} \
               OR (tn.value IS NULL AND lower(tn.key) = {p})))",
            fields::live_note("tn")
        ));
    }
    any_of(x, parts)
}

pub(crate) fn has(_: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
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

pub(crate) fn parent(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let s = x.s();
    let v = x.value()?;
    if v.eq_ignore_ascii_case("missing") {
        return Ok(format!(
            "EXISTS (SELECT 1 FROM link l WHERE l.child_pk = {s}.pk AND NOT EXISTS (SELECT 1 FROM session lp WHERE lp.id = l.parent_id))"
        ));
    }
    let (e, p) = c.id_params(&v);
    Ok(format!(
        "EXISTS (SELECT 1 FROM link l WHERE l.child_pk = {s}.pk AND (l.parent_id = {e} OR l.parent_id LIKE {p} ESCAPE '\\'))"
    ))
}

pub(crate) fn root(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let (e, p) = c.id_params(&x.value()?);
    Ok(format!(
        "{}.pk IN (WITH RECURSIVE tr(pk) AS (SELECT pk FROM session WHERE id = {e} OR id LIKE {p} ESCAPE '\\' \
         UNION SELECT l.child_pk FROM link l JOIN session lp ON lp.id = l.parent_id JOIN tr ON tr.pk = lp.pk \
         WHERE l.kind = 'subagent' AND l.child_pk IS NOT NULL) SELECT pk FROM tr)",
        x.s()
    ))
}

pub(crate) fn depth(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let expr = format!(
        "(WITH RECURSIVE up(pk, d) AS (SELECT {}.pk, 0 UNION SELECT lp.pk, up.d + 1 FROM up \
         JOIN link l ON l.child_pk = up.pk AND l.kind = 'subagent' JOIN session lp ON lp.id = l.parent_id \
         WHERE up.d < 1000) SELECT max(d) FROM up)",
        x.s()
    );
    let def = fields::Field {
        unit: Unit::Sessions,
        name: "depth",
        kind: Kind::Number,
        values: None,
        help: "",
        sql: None,
        value_kind: None,
        compile: None,
    };
    if x.top {
        c.show("depth", &expr, ColKind::Number, None);
    }
    c.typed(&def, &expr, x.op, x.span)
}

pub(crate) fn link(_: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
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
            "EXISTS (SELECT 1 FROM link l WHERE l.child_pk = {s}.pk AND l.{column} = '{v}')"
        ));
    }
    any_of(x, parts)
}

pub(crate) fn continues(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let (e, p) = c.id_params(&x.value()?);
    Ok(format!(
        "EXISTS (SELECT 1 FROM link l WHERE l.child_pk = {}.pk AND l.kind = 'continuation' AND (l.parent_id = {e} OR l.parent_id LIKE {p} ESCAPE '\\'))",
        x.s()
    ))
}

pub(crate) fn continued_by(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let (e, p) = c.id_params(&x.value()?);
    Ok(format!(
        "EXISTS (SELECT 1 FROM link l JOIN session lc ON lc.pk = l.child_pk WHERE l.parent_id = {}.id \
         AND l.kind = 'continuation' AND (lc.id = {e} OR lc.id LIKE {p} ESCAPE '\\'))",
        x.s()
    ))
}

pub(crate) fn connected(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let (e, p) = c.id_params(&x.value()?);
    Ok(format!(
        "{}.id IN (WITH RECURSIVE comp(id) AS (SELECT id FROM session WHERE id = {e} OR id LIKE {p} ESCAPE '\\' \
         UNION SELECT CASE WHEN l.parent_id = comp.id THEN lc.id ELSE l.parent_id END \
         FROM link l JOIN session lc ON lc.pk = l.child_pk JOIN comp ON comp.id = l.parent_id OR comp.id = lc.id) \
         SELECT id FROM comp)",
        x.s()
    ))
}

/// `path:`, `tool:` and `action:` on sessions: the session made such a tool
/// call.
pub(crate) fn session_tool_call(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_>,
) -> Result<String, QueryError> {
    let inner = RowContext {
        unit: Unit::ToolCalls,
        a: Aliases {
            t: "tq".into(),
            ..x.scope.a.clone()
        },
    };
    let cond = c.field_cond(x.name, x.op, &inner, x.span, false)?;
    Ok(format!(
        "EXISTS (SELECT 1 FROM tool_call tq WHERE tq.session_pk = {}.pk AND {cond})",
        x.s()
    ))
}

/// `command:` and `path:` on tool calls: a substring, through the trigram
/// index from three characters.
pub(crate) fn tool_call_text(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let t = &x.scope.a.t;
    let name = x.name;
    let mut parts = vec![];
    for v in x.values() {
        if v.chars().count() >= 3 {
            let phrase = format!("\"{}\"", v.replace('"', "\"\""));
            let p = c.text_param(&phrase);
            parts.push(format!(
                "{t}.pk IN (SELECT rowid FROM tool_call_trigram WHERE tool_call_trigram MATCH '{name}:' || {p})"
            ));
        } else {
            let p = c.text_param(&v.to_lowercase());
            parts.push(format!("instr(lower({t}.{name}), {p}) > 0"));
        }
    }
    any_of(x, parts)
}

pub(crate) fn input(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let p = c.text_param(&x.value()?);
    Ok(format!(
        "instr(pl_key(json_extract(pl_block({}.body), '$.input')), pl_key({p})) > 0",
        x.scope.a.cb
    ))
}

pub(crate) fn mcp_server(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let vs: Vec<String> = x
        .values()
        .iter()
        .map(|v| c.text_param(&v.to_lowercase()))
        .collect();
    if vs.is_empty() {
        return Err(x.need_value());
    }
    Ok(format!(
        "({t}.tool LIKE 'mcp\\_\\_%' ESCAPE '\\' AND lower(substr({t}.tool, 6, instr(substr({t}.tool, 6), '__') - 1)) IN ({}))",
        vs.join(", "),
        t = x.scope.a.t
    ))
}

pub(crate) fn spawned_child(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let (e, p) = c.id_params(&x.value()?);
    let a = &x.scope.a;
    Ok(format!(
        "EXISTS (SELECT 1 FROM link lc JOIN session lcs ON lcs.pk = lc.child_pk WHERE lc.kind = 'subagent' \
         AND lc.parent_id = {s}.id AND lc.spawning_call = {t}.call_id \
         AND (lcs.id = {e} OR lcs.id LIKE {p} ESCAPE '\\'))",
        s = a.s,
        t = a.t,
    ))
}

pub(crate) fn next_call(_: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    Err(QueryError::Invalid {
        message: "next names a relation; filter on next.<field>, such as next.action:shell".into(),
        position: x.span.start,
    })
}

/// A field naming a session by ID or unique prefix in a column of the row.
fn session_id_in(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_>,
    column: &str,
) -> Result<String, QueryError> {
    let (e, p) = c.id_params(&x.value()?);
    let column = x.scope.a.expand(column);
    Ok(format!("({column} = {e} OR {column} LIKE {p} ESCAPE '\\')"))
}

pub(crate) fn link_parent(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    session_id_in(c, x, "{r}.parent_id")
}

pub(crate) fn link_child(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    session_id_in(c, x, "{r}c.id")
}

pub(crate) fn recipient(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    session_id_in(c, x, "{r}.recipient_id")
}

/// A note about the reference; a bare session ID is a `session:` reference.
pub(crate) fn note_target(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let p = c.text_param(&x.value()?);
    Ok(format!(
        "EXISTS (SELECT 1 FROM a.note_target nt WHERE nt.note_id = {r}.id AND (nt.ref = {p} OR nt.ref = 'session:' || {p}))",
        r = x.row()
    ))
}

pub(crate) fn refs(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    delivery_ref(c, x, "refs_json")
}

pub(crate) fn left_out(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    delivery_ref(c, x, "left_out_json")
}

fn delivery_ref(
    c: &mut Compiler<'_>,
    x: &Call<'_, '_>,
    column: &str,
) -> Result<String, QueryError> {
    let p = c.text_param(&x.value()?);
    Ok(format!(
        "EXISTS (SELECT 1 FROM json_each({r}.{column}) dj WHERE instr(COALESCE(json_extract(dj.value, '$.ref'), dj.value), {p}) > 0)",
        r = x.row()
    ))
}

impl Compiler<'_> {
    pub(super) fn id_params(&mut self, v: &str) -> (String, String) {
        let e = self.text_param(v);
        let p = self.text_param(&like_prefix(v));
        (e, p)
    }
}

/// Stored selections are output data, not scalar query operands.
pub(crate) fn selection(_: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    Err(QueryError::Invalid {
        message: format!(
            "{} is a selection array; predicates on it are not supported",
            x.name
        ),
        position: x.span.start,
    })
}

pub(crate) fn event_text(c: &mut Compiler<'_>, x: &Call<'_, '_>) -> Result<String, QueryError> {
    let parts = x.values().into_iter().map(|value| {
        let parameter = c.text_param(&value);
        format!("EXISTS(SELECT 1 FROM json_tree(pl_block({}.data)) event_text_value WHERE event_text_value.type='text' AND instr(pl_key(event_text_value.atom),pl_key({parameter}))>0)",x.row())
    }).collect();
    any_of(x, parts)
}
