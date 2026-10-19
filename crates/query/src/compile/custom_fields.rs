//! Fields whose SQL is written by hand here rather than read from the
//! field table: each needs a rule of its own (a subquery, a parameter
//! shape, a check against other conditions).

use super::*;

impl<'a> Compiler<'a> {
    /// A field whose SQL is written here, by its own rule.
    pub(super) fn custom_field(
        &mut self,
        unit: Unit,
        name: &str,
        op: FieldOp<'_>,
        scope: &Scope,
        span: Span,
        top: bool,
    ) -> Result<String, QueryError> {
        let a = &scope.a;
        let s = &a.s;
        let one = |op: &FieldOp<'_>| -> Option<String> {
            match op {
                FieldOp::Match(Value::One { text, .. }) => Some(text.clone()),
                _ => None,
            }
        };
        let need_value = |name: &str| QueryError::Invalid {
            message: format!("{name} takes {name}:VALUE"),
            position: span.start,
        };
        let values = |op: &FieldOp<'_>| -> Vec<String> {
            match op {
                FieldOp::Match(v) => alternatives(v),
                _ => vec![],
            }
        };
        Ok(match (unit, name) {
            (_, "near") => {
                if !top {
                    return Err(QueryError::Invalid {
                        message: "near: ranks the whole query, so it appears once among the top-level conditions".into(),
                        position: span.start,
                    });
                }
                if !self.opts.embeddings {
                    self.degraded.push(
                        "semantic search is not enabled; rows are ordered by time and unranked"
                            .into(),
                    );
                    return Ok("1".into());
                }
                let text = one(&op).ok_or_else(|| need_value(name))?;
                // The caller binds the nearest texts as `[[text id, similarity], ...]`.
                let param = self.bind(Param::Text("[]".into()));
                let index = self.params.len();
                let cte = format!("n{}", self.ctes.len() + 1);
                self.ctes.push(format!(
                    "{cte} AS MATERIALIZED (SELECT CAST(json_extract(j.value, '$[0]') AS INTEGER) AS text_id, \
                     -json_extract(j.value, '$[1]') AS rank FROM json_each({param}) j)"
                ));
                let a = &scope.a;
                let rank = match unit {
                    Unit::Sessions => format!(
                        "(SELECT min(h.rank) FROM {cte} h JOIN block xb ON xb.text_id = h.text_id \
                         WHERE xb.session_pk IN ({}))",
                        fields::subtree(&a.s)
                    ),
                    Unit::Messages => format!(
                        "(SELECT min(h.rank) FROM {cte} h JOIN block xb ON xb.text_id = h.text_id \
                         WHERE xb.message_pk = {}.pk)",
                        a.m
                    ),
                    Unit::Blocks => {
                        format!(
                            "(SELECT h.rank FROM {cte} h WHERE h.text_id = {}.text_id)",
                            a.b
                        )
                    }
                    other => {
                        return Err(QueryError::Invalid {
                            message: format!(
                                "near: ranks sessions, messages and blocks, not {}",
                                other.name()
                            ),
                            position: span.start,
                        });
                    }
                };
                // Semantic similarity ranks the whole query, ahead of any
                // full-text rank.
                self.rank = Some(rank);
                self.near = Some(Near { text, param: index });
                "1".into()
            }
            (_, "output" | "thinking") => {
                let text = one(&op).ok_or_else(|| need_value(name))?;
                let quoted = matches!(op, FieldOp::Match(Value::One { quoted: true, .. }));
                let index = if name == "output" {
                    Index::Output
                } else {
                    Index::Thinking
                };
                self.text(&[(text, quoted, span)], scope, index, false)?
            }
            (_, "since" | "until") => {
                let t = self.time(name, &one(&op).ok_or_else(|| need_value(name))?, span)?;
                let spec = units::spec(unit);
                let (window, at) = if name == "since" {
                    (spec.since, t.start())
                } else {
                    (spec.until, t.end())
                };
                let window = window.ok_or_else(|| self.unknown_field(name, unit, span))?;
                a.expand(window).replace("{v}", &at.to_string())
            }
            (Unit::Changes, "after") => {
                let text = match &op {
                    FieldOp::Match(Value::One { text, .. }) => text.as_str(),
                    _ => "",
                };
                let cursor = text
                    .split_once('.')
                    .and_then(|(i, a)| Some((i.parse::<i64>().ok()?, a.parse::<i64>().ok()?)));
                let Some((index, archive)) = cursor else {
                    return Err(QueryError::Invalid {
                        message: "after: takes a cursor `INDEX.ARCHIVE`, two numbers".into(),
                        position: span.start,
                    });
                };
                format!(
                    "(({r}.kind = 'note' AND {r}.seq > {archive}) OR ({r}.kind = 'removed' AND {r}.seq > {index}) \
                     OR ({r}.kind = 'session' AND ({r}.changed > {index} OR {r}.live_changed > {index})))",
                    r = a.row
                )
            }
            (Unit::Sessions, "repo")
                if matches!(&op, FieldOp::Match(Value::One { text, quoted: false })
                    if text.eq_ignore_ascii_case("unknown")) =>
            {
                format!("({s}.repo IS NULL AND {s}.repo_root IS NULL)")
            }
            (Unit::Sessions, "repo") => {
                let mut parts = vec![];
                for v in values(&op) {
                    let v = v.to_lowercase().trim_end_matches('/').to_owned();
                    let p = self.text_param(&v);
                    let tail = self.text_param(&format!("%/{}", like_escape(&v)));
                    let tail_git = self.text_param(&format!("%/{}.git", like_escape(&v)));
                    parts.push(format!(
                        "(lower(rtrim({s}.repo, '/')) = {p} OR lower(rtrim({s}.repo, '/')) LIKE {tail} ESCAPE '\\' \
                         OR lower({s}.repo) LIKE {tail_git} ESCAPE '\\' OR {s}.repo_root = {p})"
                    ));
                }
                if parts.is_empty() {
                    return Err(need_value(name));
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::Sessions, "collection") => {
                let mut parts = vec![];
                for v in values(&op) {
                    let p = self.text_param(&v.to_lowercase());
                    parts.push(format!(
                        "(lower({s}.collection_name) = {p} OR lower({s}.collection_id) = {p})"
                    ));
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::Sessions, "live") => {
                let now = self.opts.clock.now;
                let mut states = vec![];
                for v in values(&op) {
                    let v = v.to_ascii_lowercase();
                    if !fields::LIVE_STATES.contains(&v.as_str()) {
                        return Err(QueryError::BadValue {
                            field: name.into(),
                            value: v,
                            position: span.start,
                            message: format!("live is one of {}", fields::LIVE_STATES.join(", ")),
                        });
                    }
                    states.push(format!("'{v}'"));
                }
                if states.is_empty() {
                    return Err(need_value(name));
                }
                format!("{} IN ({})", fields::live_state(s, now), states.join(", "))
            }
            (Unit::Sessions, "waiting" | "running") => {
                let now = self.opts.clock.now;
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let want = match v.to_ascii_lowercase().as_str() {
                    "true" | "yes" | "1" => true,
                    "false" | "no" | "0" => false,
                    _ => {
                        return Err(QueryError::BadValue {
                            field: name.into(),
                            value: v,
                            position: span.start,
                            message: format!("{name} is true or false"),
                        });
                    }
                };
                let states = if name == "waiting" {
                    "('waiting')"
                } else {
                    "('working', 'waiting', 'idle', 'unknown')"
                };
                let not = if want { "" } else { "NOT " };
                format!("{} {not}IN {states}", fields::live_state(s, now))
            }
            (Unit::Sessions, "commit") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                if v.len() < 7 || !v.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Err(QueryError::BadValue {
                        field: name.into(),
                        value: v,
                        position: span.start,
                        message: "commit takes a hash prefix of at least 7 hex digits".into(),
                    });
                }
                let p = self.text_param(&format!("{}%", v.to_lowercase()));
                let q = self.text_param(&v.to_lowercase());
                // A hash is stored as the command printed it, often short:
                // either may be a prefix of the other.
                format!(
                    "EXISTS (SELECT 1 FROM commit_link cl WHERE cl.session_pk = {s}.pk \
                     AND (cl.commit_hash LIKE {p} OR {q} LIKE cl.commit_hash || '%'))"
                )
            }
            (Unit::Sessions, "pr") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let (repo, number) = parse_pr(&v).ok_or_else(|| QueryError::BadValue {
                    field: name.into(),
                    value: v.clone(),
                    position: span.start,
                    message: "pr takes a number, owner/name#N, or a pull request URL".into(),
                })?;
                let n = self.bind(Param::Int(number));
                match repo {
                    Some(repo) => {
                        let p = self.text_param(&repo.to_lowercase());
                        format!(
                            "EXISTS (SELECT 1 FROM pr_link pl WHERE pl.session_pk = {s}.pk AND pl.number = {n} AND lower(pl.repo) = {p})"
                        )
                    }
                    None => format!(
                        "EXISTS (SELECT 1 FROM pr_link pl WHERE pl.session_pk = {s}.pk AND pl.number = {n})"
                    ),
                }
            }
            (Unit::Sessions, "cwd") => {
                let mut parts = vec![];
                for v in values(&op) {
                    let a1 = self.subtree(&format!("{s}.cwd"), &v);
                    let a2 = self.subtree("cj.value", &v);
                    parts.push(format!(
                        "({a1} OR EXISTS (SELECT 1 FROM json_each({s}.cwd_json) cj WHERE {a2}))"
                    ));
                }
                if parts.is_empty() {
                    return Err(need_value(name));
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::Sessions, "model") => match op {
                FieldOp::Match(Value::One {
                    text,
                    quoted: false,
                }) if text.eq_ignore_ascii_case("unknown") => {
                    format!(
                        "NOT EXISTS (SELECT 1 FROM message x WHERE x.session_pk = {s}.pk AND x.model IS NOT NULL)"
                    )
                }
                FieldOp::Match(Value::Any) => "1".into(),
                _ => {
                    let vs: Vec<String> = values(&op)
                        .iter()
                        .map(|v| self.text_param(&v.to_lowercase()))
                        .collect();
                    if vs.is_empty() {
                        return Err(need_value(name));
                    }
                    // Unknown when no message records a model, so neither
                    // the condition nor its NOT matches (three-valued, law 132).
                    format!(
                        "CASE WHEN EXISTS (SELECT 1 FROM message x WHERE x.session_pk = {s}.pk AND x.model IS NOT NULL) \
                         THEN EXISTS (SELECT 1 FROM message x WHERE x.session_pk = {s}.pk AND lower(x.model) IN ({})) END",
                        vs.join(", ")
                    )
                }
            },
            (Unit::Sessions, "tag") => {
                let mut parts = vec![];
                for v in values(&op) {
                    let v = v.to_lowercase();
                    let p = self.text_param(&v);
                    parts.push(format!(
                        "EXISTS (SELECT 1 FROM a.note tn JOIN a.note_target tt ON tt.note_id = tn.id \
                         WHERE tn.kind = 'tag' AND tt.ref = 'session:' || {s}.id AND {} \
                         AND ((tn.key IS NULL AND lower(tn.value) = {p}) \
                           OR lower(tn.key) || ':' || lower(tn.value) = {p} \
                           OR (tn.value IS NULL AND lower(tn.key) = {p})))",
                        fields::live_note("tn")
                    ));
                }
                if parts.is_empty() {
                    return Err(need_value(name));
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::Sessions, "has") => {
                let mut parts = vec![];
                for v in values(&op) {
                    parts.push(self.session_has(s, &v, span)?);
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::Messages, "has") => {
                let mut parts = vec![];
                for v in values(&op) {
                    let m = &a.m;
                    parts.push(match v.to_ascii_lowercase().as_str() {
                        "thinking" => format!("EXISTS (SELECT 1 FROM block xb WHERE xb.message_pk = {m}.pk AND xb.kind = 'thinking')"),
                        "tool_calls" => format!("EXISTS (SELECT 1 FROM block xb WHERE xb.message_pk = {m}.pk AND xb.kind = 'tool_call')"),
                        "attachments" => format!("EXISTS (SELECT 1 FROM block xb WHERE xb.message_pk = {m}.pk AND xb.kind IN ('image', 'file'))"),
                        "code" => format!(
                            "EXISTS (SELECT 1 FROM block xb JOIN text tx ON tx.id = xb.text_id WHERE xb.message_pk = {m}.pk AND xb.kind = 'text' AND instr(pl_text(tx.body), char(96, 96, 96)) > 0)"
                        ),
                        other => {
                            return Err(QueryError::BadValue {
                                field: "has".into(),
                                value: other.into(),
                                position: span.start,
                                message: format!("has on messages is one of thinking, tool_calls, attachments, code; not {other:?}"),
                            });
                        }
                    });
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::Sessions, "parent") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                if v.eq_ignore_ascii_case("missing") {
                    format!(
                        "EXISTS (SELECT 1 FROM link l WHERE l.child_pk = {s}.pk AND NOT EXISTS (SELECT 1 FROM session lp WHERE lp.id = l.parent_id))"
                    )
                } else {
                    let (e, p) = self.id_params(&v);
                    format!(
                        "EXISTS (SELECT 1 FROM link l WHERE l.child_pk = {s}.pk AND (l.parent_id = {e} OR l.parent_id LIKE {p} ESCAPE '\\'))"
                    )
                }
            }
            (Unit::Sessions, "root") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let (e, p) = self.id_params(&v);
                format!(
                    "{s}.pk IN (WITH RECURSIVE tr(pk) AS (SELECT pk FROM session WHERE id = {e} OR id LIKE {p} ESCAPE '\\' \
                     UNION SELECT l.child_pk FROM link l JOIN session lp ON lp.id = l.parent_id JOIN tr ON tr.pk = lp.pk \
                     WHERE l.kind = 'subagent' AND l.child_pk IS NOT NULL) SELECT pk FROM tr)"
                )
            }
            (Unit::Sessions, "depth") => {
                let expr = format!(
                    "(WITH RECURSIVE up(pk, d) AS (SELECT {s}.pk, 0 UNION SELECT lp.pk, up.d + 1 FROM up \
                     JOIN link l ON l.child_pk = up.pk AND l.kind = 'subagent' JOIN session lp ON lp.id = l.parent_id \
                     WHERE up.d < 1000) SELECT max(d) FROM up)"
                );
                let def = fields::Field {
                    unit: Unit::Sessions,
                    name: "depth",
                    kind: Kind::Number,
                    values: None,
                    help: "",
                    sql: None,
                    value_kind: None,
                };
                if top {
                    self.show("depth", &expr, ColKind::Number);
                }
                self.typed(&def, &expr, op, span)?
            }
            (Unit::Sessions, "link") => {
                let mut parts = vec![];
                for v in values(&op) {
                    let cond = match v.to_ascii_lowercase().as_str() {
                        "fork" => "((l.kind = 'continuation' AND l.how = 'fork') OR (l.kind = 'subagent' AND l.how = 'copied'))".to_owned(),
                        "resume" => "(l.kind = 'continuation' AND l.how = 'resume')".to_owned(),
                        k @ ("subagent" | "continuation" | "reference") => format!("l.kind = '{k}'"),
                        other => {
                            return Err(QueryError::BadValue {
                                field: "link".into(),
                                value: other.into(),
                                position: span.start,
                                message: format!("link is one of subagent, continuation, reference, fork, resume; not {other:?}"),
                            });
                        }
                    };
                    parts.push(format!(
                        "EXISTS (SELECT 1 FROM link l WHERE l.child_pk = {s}.pk AND {cond})"
                    ));
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::Sessions, "continues") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let (e, p) = self.id_params(&v);
                format!(
                    "EXISTS (SELECT 1 FROM link l WHERE l.child_pk = {s}.pk AND l.kind = 'continuation' AND (l.parent_id = {e} OR l.parent_id LIKE {p} ESCAPE '\\'))"
                )
            }
            (Unit::Sessions, "continued_by") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let (e, p) = self.id_params(&v);
                format!(
                    "EXISTS (SELECT 1 FROM link l JOIN session lc ON lc.pk = l.child_pk WHERE l.parent_id = {s}.id \
                     AND l.kind = 'continuation' AND (lc.id = {e} OR lc.id LIKE {p} ESCAPE '\\'))"
                )
            }
            (Unit::Sessions, "lineage") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let (e, p) = self.id_params(&v);
                format!(
                    "{s}.id IN (WITH RECURSIVE comp(id) AS (SELECT id FROM session WHERE id = {e} OR id LIKE {p} ESCAPE '\\' \
                     UNION SELECT CASE WHEN l.parent_id = comp.id THEN lc.id ELSE l.parent_id END \
                     FROM link l JOIN session lc ON lc.pk = l.child_pk JOIN comp ON comp.id = l.parent_id OR comp.id = lc.id) \
                     SELECT id FROM comp)"
                )
            }
            (Unit::Sessions, "path" | "tool" | "action") => {
                let inner = Scope {
                    unit: Unit::ToolCalls,
                    a: Aliases {
                        t: "tq".into(),
                        ..a.clone()
                    },
                };
                let cond = self.field_cond(name, op, &inner, span, false)?;
                format!(
                    "EXISTS (SELECT 1 FROM tool_call tq WHERE tq.session_pk = {s}.pk AND {cond})"
                )
            }
            (Unit::ToolCalls, "command" | "path") => {
                let t = &a.t;
                let mut parts = vec![];
                for v in values(&op) {
                    let col = format!("{t}.{name}");
                    if v.chars().count() >= 3 {
                        let phrase = format!("\"{}\"", v.replace('"', "\"\""));
                        let p = self.text_param(&phrase);
                        parts.push(format!(
                            "{t}.pk IN (SELECT rowid FROM tool_call_trigram WHERE tool_call_trigram MATCH '{name}:' || {p})"
                        ));
                    } else {
                        let p = self.text_param(&v.to_lowercase());
                        parts.push(format!("instr(lower({col}), {p}) > 0"));
                    }
                }
                if parts.is_empty() {
                    return Err(need_value(name));
                }
                format!("({})", parts.join(" OR "))
            }
            (Unit::ToolCalls, "input") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let p = self.text_param(&v);
                format!(
                    "instr(pl_key(json_extract(pl_block({}.body), '$.input')), pl_key({p})) > 0",
                    a.cb
                )
            }
            (Unit::ToolCalls, "mcp_server") => {
                let vs: Vec<String> = values(&op)
                    .iter()
                    .map(|v| self.text_param(&v.to_lowercase()))
                    .collect();
                if vs.is_empty() {
                    return Err(need_value(name));
                }
                format!(
                    "({t}.tool LIKE 'mcp\\_\\_%' ESCAPE '\\' AND lower(substr({t}.tool, 6, instr(substr({t}.tool, 6), '__') - 1)) IN ({}))",
                    vs.join(", "),
                    t = a.t
                )
            }
            (Unit::ToolCalls, "child" | "next")
            | (Unit::Links, "parent" | "child")
            | (Unit::Deliveries, "recipient") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let (e, p) = self.id_params(&v);
                match (unit, name) {
                    (Unit::ToolCalls, "child") => format!(
                        "EXISTS (SELECT 1 FROM link lc JOIN session lcs ON lcs.pk = lc.child_pk WHERE lc.kind = 'subagent' \
                         AND lc.parent_id = {s}.id AND (lc.spawning_call = {t}.call_id OR lc.spawned_by = {cb}.pk) \
                         AND (lcs.id = {e} OR lcs.id LIKE {p} ESCAPE '\\'))",
                        t = a.t,
                        cb = a.cb
                    ),
                    (Unit::Links, "parent") => format!(
                        "({r}.parent_id = {e} OR {r}.parent_id LIKE {p} ESCAPE '\\')",
                        r = a.row
                    ),
                    (Unit::Links, "child") => {
                        format!("({r}c.id = {e} OR {r}c.id LIKE {p} ESCAPE '\\')", r = a.row)
                    }
                    (Unit::Deliveries, "recipient") => format!(
                        "({r}.recipient_id = {e} OR {r}.recipient_id LIKE {p} ESCAPE '\\')",
                        r = a.row
                    ),
                    _ => {
                        return Err(QueryError::Invalid {
                            message: "next names a relation; filter on next.<field>, such as next.action:shell".into(),
                            position: span.start,
                        });
                    }
                }
            }
            (Unit::Notes, "target") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let p = self.text_param(&v);
                let pp = self.text_param(&like_prefix(&v));
                format!(
                    "EXISTS (SELECT 1 FROM a.note_target nt WHERE nt.note_id = {r}.id AND (nt.ref = {p} OR nt.ref LIKE {pp} ESCAPE '\\' OR nt.ref = 'session:' || {p}))",
                    r = a.row
                )
            }
            (Unit::Deliveries, "refs" | "left_out") => {
                let v = one(&op).ok_or_else(|| need_value(name))?;
                let col = if name == "refs" {
                    "refs_json"
                } else {
                    "left_out_json"
                };
                let p = self.text_param(&v);
                format!(
                    "EXISTS (SELECT 1 FROM json_each({r}.{col}) dj WHERE instr(COALESCE(json_extract(dj.value, '$.ref'), dj.value), {p}) > 0)",
                    r = a.row
                )
            }
            _ => return Err(self.unknown_field(name, unit, span)),
        })
    }

    pub(super) fn id_params(&mut self, v: &str) -> (String, String) {
        let e = self.text_param(v);
        let p = self.text_param(&like_prefix(v));
        (e, p)
    }

    pub(super) fn session_has(
        &mut self,
        s: &str,
        v: &str,
        span: Span,
    ) -> Result<String, QueryError> {
        let own = fields::own_messages(s, "xm");
        Ok(match v.to_ascii_lowercase().as_str() {
            "thinking" => format!(
                "EXISTS (SELECT 1 FROM block xb JOIN message xm ON xm.pk = xb.message_pk WHERE {own} AND xb.kind = 'thinking')"
            ),
            "tool_calls" => {
                format!("EXISTS (SELECT 1 FROM tool_call xt WHERE xt.session_pk = {s}.pk)")
            }
            "errors" => format!(
                "EXISTS (SELECT 1 FROM tool_call xt WHERE xt.session_pk = {s}.pk AND xt.outcome = 'error')"
            ),
            "attachments" => format!(
                "EXISTS (SELECT 1 FROM block xb WHERE xb.session_pk = {s}.pk AND xb.kind IN ('image', 'file'))"
            ),
            "compaction" => format!(
                "EXISTS (SELECT 1 FROM event xe WHERE xe.session_pk = {s}.pk AND xe.kind = 'compaction')"
            ),
            "branches" => format!(
                "EXISTS (SELECT 1 FROM message xm WHERE xm.session_pk = {s}.pk AND xm.on_active_path = 0)"
            ),
            "secret" => format!(
                "EXISTS (SELECT 1 FROM block xb WHERE xb.session_pk = {s}.pk AND xb.looks_like_secret = 1)"
            ),
            "note" => format!(
                "EXISTS (SELECT 1 FROM a.note hn JOIN a.note_target ht ON ht.note_id = hn.id \
                 WHERE (ht.ref = 'session:' || {s}.id OR ht.ref IN (SELECT 'message:' || xm.id FROM message xm WHERE xm.session_pk = {s}.pk)) \
                 AND {})",
                fields::live_note("hn")
            ),
            "conflict" => format!(
                "EXISTS (SELECT 1 FROM a.note hn JOIN a.note_target ht ON ht.note_id = hn.id \
                 WHERE ht.ref = 'session:' || {s}.id AND {} \
                 GROUP BY hn.root HAVING count(*) > 1)",
                fields::current_note("hn")
            ),
            other => {
                return Err(QueryError::BadValue {
                    field: "has".into(),
                    value: other.into(),
                    position: span.start,
                    message: format!(
                        "has on sessions is one of {}; not {other:?}",
                        fields::SESSION_HAS.join(", ")
                    ),
                });
            }
        })
    }
}
