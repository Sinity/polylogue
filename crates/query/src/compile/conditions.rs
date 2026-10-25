//! Conditions: text terms, references, and fields compared by the type the
//! field table gives them.

use super::*;

impl<'a> Compiler<'a> {
    pub(super) fn is_reference(&self, c: &Cond) -> bool {
        match &c.kind {
            CondKind::Url { .. } | CondKind::Position { .. } => true,
            CondKind::Match { field, .. } => {
                let f = field.to_ascii_lowercase();
                self.opts.origins.contains(&f) || matches!(f.as_str(), "session" | "message")
            }
            CondKind::Text {
                text,
                phrase: false,
            } => matches!(text.as_str(), "last" | "latest" | "self") || id_shaped(text),
            _ => false,
        }
    }

    /// A condition. `top` is true for the query's own `where`, where a
    /// top-level text term ranks the rows.
    pub(super) fn cond(
        &mut self,
        c: &Cond,
        scope: &RowContext,
        top: bool,
    ) -> Result<String, QueryError> {
        Ok(match &c.kind {
            CondKind::And { all } => {
                // Several words in one AND must occur in the same message.
                let mut parts = vec![];
                let mut words = vec![];
                for x in all {
                    match &x.kind {
                        CondKind::Text { text, phrase }
                            if !self.is_reference(x) && text_unit(scope.unit) =>
                        {
                            words.push((text.clone(), *phrase, x.span));
                        }
                        _ => parts.push(self.cond(x, scope, top)?),
                    }
                }
                if !words.is_empty() {
                    parts.push(self.text(&words, scope, Index::Prose, top)?);
                }
                format!("({})", parts.join(" AND "))
            }
            CondKind::Or { any } => {
                let parts = any
                    .iter()
                    .map(|x| self.cond(x, scope, false))
                    .collect::<Result<Vec<_>, _>>()?;
                format!("({})", parts.join(" OR "))
            }
            CondKind::Not { cond } => format!("NOT ({})", self.cond(cond, scope, false)?),
            CondKind::Text { text, phrase } => {
                if self.is_reference(c) {
                    self.reference_word(text, scope, c.span)?
                } else {
                    self.text(&[(text.clone(), *phrase, c.span)], scope, Index::Prose, top)?
                }
            }
            CondKind::Url { url } => self.url(url, scope, c.span)?,
            CondKind::Position { .. } => {
                return Err(QueryError::Invalid {
                    message: "@N refers to a position in the last listing, which only an interactive surface keeps".into(),
                    position: c.span.start,
                });
            }
            CondKind::Exists { unit, cond } => self.exists(*unit, cond, scope, c.span)?,
            CondKind::Seq { steps, gaps } => self.seq(steps, gaps, scope, c.span)?,
            CondKind::Match { field, value } => {
                self.field_cond(field, FieldOp::Match(value), scope, c.span, top)?
            }
            CondKind::Compare { field, op, value } => {
                self.field_cond(field, FieldOp::Compare(*op, value), scope, c.span, top)?
            }
            CondKind::Between { field, from, to } => {
                self.field_cond(field, FieldOp::Between(from, to), scope, c.span, top)?
            }
        })
    }

    /// A full-text condition over one index. Terms ANDed together must all
    /// occur in the same message (or block).
    pub(super) fn text(
        &mut self,
        words: &[(String, bool, Span)],
        scope: &RowContext,
        index: Index,
        top: bool,
    ) -> Result<String, QueryError> {
        let mut phrases = vec![];
        // A quoted phrase and a bare word with punctuation both search
        // their tokens in order.
        for (word, _, span) in words {
            let p = polylogue_model::text::fts_phrase(word);
            let Some(p) = p else {
                return Err(QueryError::Invalid {
                    message: format!(
                        "{word:?} has no letter or digit to search for; use command:\"…\" or input:\"…\" for punctuation"
                    ),
                    position: span.start,
                });
            };
            self.text_terms.push(word.trim_end_matches('*').to_owned());
            phrases.push(p);
        }
        let refuse = |unit: Unit| QueryError::Invalid {
            message: format!(
                "a search word applies to sessions, messages, blocks and notes, not {}; for tool calls use output:\"…\", command:\"…\" or input:\"…\"",
                unit.name()
            ),
            position: words[0].2.start,
        };
        let a = &scope.a;
        let rows = match units::spec(scope.unit).search {
            Search::Hits { rows, .. } => rows,
            Search::Substring(column) => {
                let mut parts = vec![];
                for (w, _, _) in words {
                    let p = self.text_param(w);
                    parts.push(format!(
                        "instr(pl_key({}), pl_key({p})) > 0",
                        a.expand(column)
                    ));
                }
                return Ok(format!("({})", parts.join(" AND ")));
            }
            Search::None => return Err(refuse(scope.unit)),
        };
        let match_text = phrases.join(" AND ");
        let table = match index {
            Index::Output => "output_fts",
            _ => "text_fts",
        };
        let param = self.text_param(&match_text);
        let cte = format!("h{}", self.ctes.len() + 1);
        let Some((cond, rank)) = rows(a, &units::Hits { cte: &cte, index }) else {
            self.params.pop();
            return Err(refuse(scope.unit));
        };
        self.ctes.push(format!(
            "{cte} AS MATERIALIZED (SELECT rowid AS text_id, bm25({table}) AS rank FROM {table} WHERE {table} MATCH {param})"
        ));
        if top && index != Index::Thinking && self.rank.is_none() {
            self.rank = Some(rank);
        }
        Ok(cond)
    }

    /// A word that names something: `last`, `latest`, `self`, or a pasted
    /// ID (8+ hex digits and dashes).
    pub(super) fn reference_word(
        &mut self,
        word: &str,
        scope: &RowContext,
        span: Span,
    ) -> Result<String, QueryError> {
        let s = self.session_alias(scope, span)?;
        Ok(match word {
            "last" | "latest" => format!(
                "{s}.pk = (SELECT pk FROM session WHERE kind = '{main}' AND stub = 0 \
                 ORDER BY last_activity_at IS NULL, last_activity_at DESC, pk DESC LIMIT 1)",
                main = polylogue_model::SessionKind::Main.as_str()
            ),
            "self" => {
                let Some(me) = self.opts.self_session.clone() else {
                    return Err(QueryError::Invalid {
                        message: "`self` names the calling session, and this caller is not one"
                            .into(),
                        position: span.start,
                    });
                };
                let p = self.text_param(&me);
                format!("{s}.id = {p}")
            }
            id => {
                let p = self.text_param(&like_prefix(id));
                self.id_checks.push(IdCheck {
                    word: id.to_owned(),
                    sql: "SELECT id FROM session WHERE native_id LIKE ?1 ESCAPE '\\' LIMIT 5"
                        .into(),
                    params: vec![Param::Text(like_prefix(id))],
                });
                format!("{s}.native_id LIKE {p} ESCAPE '\\'")
            }
        })
    }

    /// A provider URL pasted as a reference.
    pub(super) fn url(
        &mut self,
        url: &str,
        scope: &RowContext,
        span: Span,
    ) -> Result<String, QueryError> {
        let id = provider_url(&self.opts.chat_pages, url).ok_or_else(|| QueryError::Invalid {
            message: format!("{url} is not a conversation URL Polylogue recognizes"),
            position: span.start,
        })?;
        self.session_ref(&id, scope, span)
    }

    /// `origin:native` (or a unique prefix of it) as a session condition.
    pub(super) fn session_ref(
        &mut self,
        id: &str,
        scope: &RowContext,
        span: Span,
    ) -> Result<String, QueryError> {
        let s = self.session_alias(scope, span)?;
        let exact = self.text_param(id);
        let prefix = self.text_param(&like_prefix(id));
        Ok(format!(
            "({s}.id = {exact} OR {s}.id LIKE {prefix} ESCAPE '\\')"
        ))
    }

    /// The alias of the session a row of this unit belongs to.
    pub(super) fn session_alias(&self, scope: &RowContext, span: Span) -> Result<String, QueryError> {
        if scope.unit == Unit::Sessions || fields::has_session(scope.unit) {
            Ok(scope.a.s.clone())
        } else {
            Err(QueryError::Invalid {
                message: format!("{} rows do not belong to one session", scope.unit.name()),
                position: span.start,
            })
        }
    }

    /// The condition tying the rows of `child` to one row of `parent`:
    /// `key` is that row's key and `session` its session's key. A child
    /// declares the column naming its parent (`UnitSpec::parents`);
    /// otherwise the two relate through the session.
    pub(super) fn under_parent(
        &mut self,
        parent: Unit,
        child: &RowContext,
        key: &str,
        session: Result<String, QueryError>,
    ) -> Result<String, QueryError> {
        let spec = units::spec(child.unit);
        if let Some((_, column)) = spec.parents.iter().find(|(p, _)| *p == parent) {
            return Ok(format!("{} = {key}", child.a.expand(column)));
        }
        let s = session?;
        if spec.in_session {
            Ok(format!(
                "{} = {s}",
                child_session_column(child.unit, &child.a)
            ))
        } else {
            self.session_link(child, &format!("SELECT {s}"))
        }
    }

    /// The rows of a unit without a session alias that relate to the
    /// sessions `sessions` selects.
    pub(super) fn session_link(
        &mut self,
        scope: &RowContext,
        sessions: &str,
    ) -> Result<String, QueryError> {
        let Some(t) = units::spec(scope.unit).session_scope else {
            if scope.unit == Unit::Documents {
                return Err(QueryError::Invalid {
                    message: "documents have no implicit session association".into(),
                    position: 0,
                });
            }
            return Err(not_built(&format!(
                "scoping {} to sessions",
                scope.unit.name()
            )));
        };
        Ok(scope.a.expand(t).replace("{sessions}", sessions))
    }

    /// A condition on a field: plain, related (`session.repo`), or special.
    pub(super) fn field_cond(
        &mut self,
        field: &str,
        op: FieldOp<'_>,
        scope: &RowContext,
        span: Span,
        top: bool,
    ) -> Result<String, QueryError> {
        let lower = field.to_ascii_lowercase();
        // A known origin or a reference prefix before `:` is a reference.
        if let FieldOp::Match(Value::One { text, .. }) = &op {
            if self.opts.origins.contains(&lower) {
                return self.session_ref(&format!("{lower}:{text}"), scope, span);
            }
            match lower.as_str() {
                "session" => return self.session_ref(text, scope, span),
                "message" => {
                    let s = self.session_alias(scope, span)?;
                    let p = self.text_param(text);
                    return Ok(match scope.unit {
                        Unit::Messages => format!("{}.id = {p}", scope.a.m),
                        _ => format!("{s}.pk IN (SELECT session_pk FROM message WHERE id = {p})"),
                    });
                }
                "query" => return self.saved_cond(text, scope, span),
                "result" => {
                    return Err(QueryError::Invalid {
                        message: "result:<id> is a source: start the query with it \
                                  (`result:<id> | count`); it cannot be a condition"
                            .into(),
                        position: span.start,
                    });
                }
                "page" => {
                    return Err(not_built("page: references (pages are not built yet)"));
                }
                "note" | "block" | "file" => {
                    return Err(not_built(&format!("{lower}: references")));
                }
                _ => {}
            }
        }
        if let Some((prefix, rest)) = lower.split_once('.')
            && let Some(target) = fields::relation(scope.unit, prefix)
        {
            return self.related(prefix, rest, target, op, scope, span);
        }
        let (unit, a) = if fields::lookup(scope.unit, &lower).is_some() {
            (scope.unit, scope.a.clone())
        } else if (fields::has_session(scope.unit))
            && fields::lookup(Unit::Sessions, &lower).is_some()
        {
            (Unit::Sessions, scope.a.clone())
        } else {
            return Err(self.unknown_field(field, scope.unit, span));
        };
        let def = fields::lookup(unit, &lower).expect("checked");
        if let Some(compile) = def.compile {
            let call = custom_fields::Call {
                unit,
                name: def.name,
                op,
                scope: &RowContext { unit, a },
                span,
                top,
            };
            return compile(self, &call);
        }
        let expr = def
            .expr(&a, self.opts.clock.now)
            .expect("non-special fields have SQL");
        if top && matches!(op, FieldOp::Compare(..) | FieldOp::Between(..)) {
            self.show(
                def.name,
                &expr,
                def.column_kind(),
                def.sql.and_then(|e| e.reference()),
            );
        }
        self.typed(def, &expr, op, span)
    }

    pub(super) fn show(
        &mut self,
        name: &str,
        expr: &str,
        kind: ColKind,
        reference: Option<ColumnReference>,
    ) {
        let n = flat(name);
        if !self.shown.iter().any(|(x, _, _, _)| *x == n) {
            let e = match kind {
                ColKind::Money => format!("({expr}) / 1e9"),
                _ => expr.to_owned(),
            };
            self.shown.push((n, e, kind, reference));
        }
    }

    pub(super) fn unknown_field(&self, field: &str, unit: Unit, span: Span) -> QueryError {
        let lower = field.to_ascii_lowercase();
        let mut pool: Vec<&str> = fields::names(unit).collect();
        if fields::has_session(unit) {
            pool.extend(fields::names(Unit::Sessions));
        }
        let mut scored: Vec<(usize, &str)> = pool
            .into_iter()
            .map(|n| (strsim::damerau_levenshtein(&lower, n), n))
            .filter(|(d, n)| *d <= 2.max(n.len() / 4))
            .collect();
        scored.sort();
        scored.dedup_by(|a, b| a.1 == b.1);
        // Only the nearest names: one typing mistake beats two.
        let best = scored.first().map_or(0, |(d, _)| *d);
        let similar: Vec<String> = scored
            .iter()
            .filter(|(d, _)| *d == best)
            .take(3)
            .map(|(_, n)| (*n).to_owned())
            .collect();
        let suggestion = match similar.as_slice() {
            [one] => Some(format!(
                "{}{}{}",
                &self.text[..span.start],
                one,
                &self.text[span.start + field.len()..]
            )),
            _ => None,
        };
        QueryError::UnknownField {
            field: field.to_owned(),
            unit: unit.name().to_owned(),
            position: span.start,
            similar,
            suggestion,
        }
    }

    /// A condition on a field of a related row.
    pub(super) fn related(
        &mut self,
        prefix: &str,
        rest: &str,
        target: Unit,
        op: FieldOp<'_>,
        scope: &RowContext,
        span: Span,
    ) -> Result<String, QueryError> {
        let sub = Span {
            start: span.start + prefix.len() + 1,
            end: span.end,
        };
        let a = &scope.a;
        let rel = units::spec(scope.unit)
            .relation(prefix)
            .ok_or_else(|| QueryError::Invalid {
                message: format!("{prefix}.{rest} does not apply to {}", scope.unit.name()),
                position: span.start,
            })?;
        match rel.via {
            Via::Same => {
                let sc = RowContext {
                    unit: target,
                    a: a.clone(),
                };
                self.field_cond(rest, op, &sc, sub, false)
            }
            Via::Session(alias) => {
                let sc = RowContext {
                    unit: Unit::Sessions,
                    a: Aliases {
                        s: a.expand(alias),
                        ..a.clone()
                    },
                };
                self.field_cond(rest, op, &sc, sub, false)
            }
            Via::Exists { cond: template, .. } => {
                let n = self.fresh();
                let inner = RowContext {
                    unit: target,
                    a: n.clone(),
                };
                let cond = self.field_cond(rest, op, &inner, sub, false)?;
                Ok(relation_sql(template, a, &n, target, &cond))
            }
        }
    }

    /// A condition on a plain field, by its type.
    pub(super) fn typed(
        &mut self,
        def: &fields::Field,
        expr: &str,
        op: FieldOp<'_>,
        span: Span,
    ) -> Result<String, QueryError> {
        let name = def.name;
        let bad = |value: &str, message: String| QueryError::BadValue {
            field: name.to_owned(),
            value: value.to_owned(),
            position: span.start,
            message,
        };
        match (def.kind, op) {
            (_, FieldOp::Match(Value::Any)) => Ok("1".into()),
            (
                _,
                FieldOp::Match(Value::One {
                    text,
                    quoted: false,
                }),
            ) if text.eq_ignore_ascii_case("unknown") => Ok(format!("({expr}) IS NULL")),
            (Kind::Match, FieldOp::Match(v)) => {
                let values = match v {
                    Value::One { text, .. } => vec![text.clone()],
                    Value::Alternatives { any } => any.clone(),
                    Value::Any => unreachable!("handled above"),
                };
                if let Some(set) = def.values {
                    for x in &values {
                        if !set.iter().any(|s| s.eq_ignore_ascii_case(x)) {
                            return Err(bad(x, closed_set_message(def, x)));
                        }
                    }
                }
                let ps: Vec<String> = values
                    .iter()
                    .map(|x| self.text_param(&x.to_lowercase()))
                    .collect();
                Ok(format!("lower({expr}) IN ({})", ps.join(", ")))
            }
            (Kind::Match, FieldOp::Compare(op, value)) => {
                let p = self.text_param(&value.to_lowercase());
                Ok(format!("lower({expr}) {} {p}", op.sql()))
            }
            (Kind::Prefix, FieldOp::Match(v)) => {
                let values = alternatives(v);
                let mut parts = vec![];
                for x in values {
                    let exact = self.text_param(&x);
                    let prefix = self.text_param(&like_prefix(&x));
                    parts.push(format!(
                        "{expr} = {exact} OR {expr} LIKE {prefix} ESCAPE '\\'"
                    ));
                }
                Ok(format!("({})", parts.join(" OR ")))
            }
            (Kind::Substring, FieldOp::Match(v)) => {
                let mut parts = vec![];
                for x in alternatives(v) {
                    let p = self.text_param(&x);
                    parts.push(format!("instr(pl_key({expr}), pl_key({p})) > 0"));
                }
                Ok(format!("({})", parts.join(" OR ")))
            }
            (Kind::Subtree, FieldOp::Match(v)) => {
                let mut parts = vec![];
                for x in alternatives(v) {
                    parts.push(self.subtree(expr, &x));
                }
                Ok(format!("({})", parts.join(" OR ")))
            }
            (Kind::Bool, FieldOp::Match(Value::One { text, .. })) => {
                let b = match text.to_ascii_lowercase().as_str() {
                    "true" | "yes" | "1" => 1,
                    "false" | "no" | "0" => 0,
                    _ => return Err(bad(text, format!("{name} is true or false"))),
                };
                Ok(format!("({expr}) = {b}"))
            }
            (
                Kind::Number | Kind::Money | Kind::Duration,
                FieldOp::Match(Value::One { text, .. }),
            ) => {
                let v = self.scalar(def, text, span)?;
                Ok(format!("({expr}) = {v}"))
            }
            (Kind::Number | Kind::Money | Kind::Duration, FieldOp::Compare(op, value)) => {
                let v = self.scalar(def, value, span)?;
                Ok(format!("({expr}) {} {v}", op.sql()))
            }
            (Kind::Number | Kind::Money | Kind::Duration, FieldOp::Between(from, to)) => {
                let a = self.scalar(def, from, span)?;
                let b = self.scalar(def, to, span)?;
                Ok(format!("(({expr}) >= {a} AND ({expr}) <= {b})"))
            }
            (Kind::Time, FieldOp::Match(Value::One { text, .. })) => {
                let t = self.time(name, text, span)?;
                Ok(format!(
                    "(({expr}) >= {} AND ({expr}) < {})",
                    t.start(),
                    t.end()
                ))
            }
            (Kind::Time, FieldOp::Compare(op, value)) => {
                let t = self.time(name, value, span)?;
                Ok(match op {
                    Op::Gt => format!("({expr}) >= {}", t.end()),
                    Op::Ge => format!("({expr}) >= {}", t.start()),
                    Op::Lt => format!("({expr}) < {}", t.start()),
                    Op::Le => format!("({expr}) < {}", t.end()),
                    Op::Eq => format!("(({expr}) >= {} AND ({expr}) < {})", t.start(), t.end()),
                    Op::Ne => format!("(({expr}) < {} OR ({expr}) >= {})", t.start(), t.end()),
                })
            }
            (Kind::Time, FieldOp::Between(from, to)) => {
                let a = self.time(name, from, span)?;
                let b = self.time(name, to, span)?;
                Ok(format!(
                    "(({expr}) >= {} AND ({expr}) < {})",
                    a.start(),
                    b.end()
                ))
            }
            (kind, op) => Err(QueryError::Invalid {
                message: format!(
                    "{name} is a {} field; {}",
                    kind_name(kind),
                    match op {
                        FieldOp::Match(_) => "it takes a comparison such as `>`".to_owned(),
                        _ => format!("write {name}:VALUE"),
                    }
                ),
                position: span.start,
            }),
        }
    }

    pub(super) fn scalar(
        &mut self,
        def: &fields::Field,
        text: &str,
        span: Span,
    ) -> Result<String, QueryError> {
        let bad = |message: String| QueryError::BadValue {
            field: def.name.to_owned(),
            value: text.to_owned(),
            position: span.start,
            message,
        };
        match def.kind {
            Kind::Money => time::money_nusd(text)
                .map(|n| n.to_string())
                .ok_or_else(|| bad(format!("{} is an amount in dollars such as 2.90", def.name))),
            Kind::Duration => time::duration_ms(text)
                .map(|n| n.to_string())
                .ok_or_else(|| bad(format!("{} is a duration such as 90s, 5m or 2h", def.name))),
            _ => {
                if let Ok(n) = text.parse::<i64>() {
                    Ok(n.to_string())
                } else if let Ok(f) = text.parse::<f64>()
                    && f.is_finite()
                {
                    Ok(format!("{f}"))
                } else {
                    Err(bad(format!("{} is a number", def.name)))
                }
            }
        }
    }

    pub(super) fn time(&self, field: &str, text: &str, span: Span) -> Result<Time, QueryError> {
        time::resolve(text, &self.opts.clock).ok_or_else(|| QueryError::BadValue {
            field: field.to_owned(),
            value: text.to_owned(),
            position: span.start,
            message: format!(
                "{text:?} is not a time: use a date (2026-10-05), today, yesterday, this week, last month, 7d, 3h or 09:00"
            ),
        })
    }

    /// `path` and the directories below it, by whole components, exactly.
    pub(super) fn subtree(&mut self, expr: &str, path: &str) -> String {
        let path = path.trim_end_matches('/');
        let path = if path.is_empty() { "/" } else { path };
        let exact = self.text_param(path);
        let below = self.text_param(&format!("{}/", path.trim_end_matches('/')));
        format!("({expr} = {exact} OR substr({expr}, 1, length({below})) = {below})")
    }
}
