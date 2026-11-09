//! Conditions: text terms, references, and fields compared by the type the
//! field table gives them.

use super::*;

impl<'a> Compiler<'a> {
    /// A reference that names a session rather than this unit's own rows:
    /// `session:X`, `origin:native`, `last`, `latest`, `self`. Over another
    /// unit it lifts only the session's visibility default; the unit's own
    /// row defaults (duplicate copies, the active path) still apply, the
    /// same as reading `X | unit`.
    pub(super) fn names_a_session(&self, c: &Condition, unit: Unit) -> bool {
        if unit == Unit::Sessions {
            return false;
        }
        match &c.kind {
            ConditionKind::Match { field, .. } => {
                let f = field.to_ascii_lowercase();
                f == "session" || self.opts.origins.contains(&f)
            }
            ConditionKind::Text {
                text,
                mode: TextMode::Word,
            } => matches!(text.as_str(), "last" | "latest" | "self"),
            _ => false,
        }
    }

    pub(super) fn is_reference(&self, c: &Condition, unit: Unit) -> bool {
        match &c.kind {
            ConditionKind::Reference { .. } => true,
            ConditionKind::Match { field, .. } => {
                let f = field.to_ascii_lowercase();
                self.opts.origins.contains(&f)
                    || (f == "event" && unit != Unit::Deliveries)
                    || matches!(
                        f.as_str(),
                        "session"
                            | "message"
                            | "block"
                            | "run"
                            | "request"
                            | "document"
                            | "span"
                            | "assertion"
                            | "delivery"
                    )
            }
            ConditionKind::Text {
                text,
                mode: TextMode::Word,
            } => matches!(text.as_str(), "last" | "latest" | "self") || id_shaped(text),
            _ => false,
        }
    }

    /// A condition. `top` is true for the query's own `where`, where a
    /// top-level text term ranks the rows.
    pub(super) fn cond(
        &mut self,
        c: &Condition,
        scope: &RowContext,
        top: bool,
    ) -> Result<String, QueryError> {
        Ok(match &c.kind {
            ConditionKind::And { conditions: all } => {
                // Several words in one AND must occur in the same message.
                let mut parts = vec![];
                let mut words = vec![];
                let native_bounds = if scope.unit != Unit::ToolCalls {
                    fields::native_window(scope.unit, true, &scope.a).filter(|lower| {
                        fields::native_window(scope.unit, false, &scope.a).is_some_and(|upper| {
                            lower.bytes == upper.bytes && lower.storage == upper.storage
                        })
                    })
                } else {
                    None
                };
                let mut conjuncts = vec![];
                if native_bounds.is_some() {
                    fn flatten<'b>(conditions: &'b [Condition], output: &mut Vec<&'b Condition>) {
                        for condition in conditions {
                            if let ConditionKind::And { conditions } = &condition.kind {
                                flatten(conditions, output);
                            } else {
                                output.push(condition);
                            }
                        }
                    }
                    flatten(all, &mut conjuncts);
                } else {
                    conjuncts.extend(all);
                }
                let all = conjuncts;
                // `last` beside other session conditions is the newest session
                // that also meets them.
                if scope.unit == Unit::Sessions
                    && all.len() > 1
                    && let Some(at) = all.iter().position(|x| newest_word(x))
                {
                    let rest = Condition {
                        span: c.span,
                        kind: ConditionKind::And {
                            conditions: all
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| *i != at)
                                .map(|(_, x)| (*x).clone())
                                .collect(),
                        },
                    };
                    let inner = RowContext {
                        unit: Unit::Sessions,
                        a: self.fresh(),
                    };
                    let within = self.cond(&rest, &inner, false)?;
                    return Ok(self.newest_session(&scope.a.s, &inner.a, Some(within)));
                }
                let combine_bounds = native_bounds.is_some()
                    && all
                        .iter()
                        .any(|x| matches!(x.kind, ConditionKind::Since { .. }))
                    && all
                        .iter()
                        .any(|x| matches!(x.kind, ConditionKind::Until { .. }));
                let mut since = None;
                let mut until = None;
                for x in all {
                    if combine_bounds {
                        match &x.kind {
                            ConditionKind::Since { value, .. } => {
                                let at = self.time("since", value_text(value)?, x.span)?.start();
                                since = Some(since.map_or(at, |old: jiff::Timestamp| old.max(at)));
                                continue;
                            }
                            ConditionKind::Until { value, .. } => {
                                let at = self
                                    .time("until", value_text(value)?, x.span)?
                                    .until_bound();
                                until = Some(until.map_or(at, |old: jiff::Timestamp| old.min(at)));
                                continue;
                            }
                            _ => {}
                        }
                    }
                    match &x.kind {
                        ConditionKind::Text { text, mode }
                            if !self.is_reference(x, scope.unit) && text_unit(scope.unit) =>
                        {
                            words.push((
                                search_word(text, *mode)?,
                                *mode == TextMode::Phrase,
                                x.span,
                            ));
                        }
                        _ => parts.push(self.cond(x, scope, top)?),
                    }
                }
                if combine_bounds {
                    parts.push(self.native_time_test(
                        &native_bounds.expect("combined direct native bounds"),
                        time::NativePredicate::Overlap { since, until },
                    ));
                }
                if !words.is_empty() {
                    parts.push(self.text(&words, scope, Index::Prose, top)?);
                }
                format!("({})", parts.join(" AND "))
            }
            ConditionKind::Or { conditions: any } => {
                let parts = any
                    .iter()
                    .map(|x| self.cond(x, scope, false))
                    .collect::<Result<Vec<_>, _>>()?;
                format!("({})", parts.join(" OR "))
            }
            ConditionKind::Not { condition: cond } => {
                format!("NOT ({})", self.cond(cond, scope, false)?)
            }
            ConditionKind::Text { text, mode } => {
                if self.is_reference(c, scope.unit) {
                    self.reference_word(text, scope, c.span)?
                } else {
                    self.text(
                        &[(search_word(text, *mode)?, *mode == TextMode::Phrase, c.span)],
                        scope,
                        Index::Prose,
                        top,
                    )?
                }
            }
            ConditionKind::Reference { reference } => {
                let native = matches!(
                    reference.split_once(':').map(|(kind, _)| kind),
                    Some("block" | "document" | "event" | "span" | "assertion" | "delivery")
                );
                if native || (reference.starts_with("message:") && reference.contains("..")) {
                    return self.native_reference(reference, scope, c.span);
                }
                // Pages and commits are designed reference forms with no producer yet.
                if let Some(kind @ ("page" | "file")) =
                    reference.split_once(':').map(|(kind, _)| kind)
                {
                    return Err(QueryError::NotBuilt {
                        what: format!("{kind} references"),
                    });
                }
                if reference.starts_with("http") {
                    self.url(reference, scope, c.span)?
                } else if reference.starts_with('@') {
                    return Err(QueryError::Invalid {
                        message: "@N requires an interactive listing context".into(),
                        position: c.span.start,
                    });
                } else if let Some((field, text)) = reference.split_once(':') {
                    let value = Value::scalar(text);
                    self.field_cond(field, FieldOp::Match(&value), scope, c.span, top)?
                } else {
                    self.reference_word(reference, scope, c.span)?
                }
            }
            ConditionKind::Exists { unit, condition } => {
                self.exists(bind_unit(unit)?, condition.as_deref(), scope, c.span)?
            }
            ConditionKind::Count {
                unit,
                condition,
                op,
                count,
            } => self.counted(
                bind_unit(unit)?,
                condition.as_deref(),
                *op,
                *count,
                scope,
                c.span,
            )?,
            ConditionKind::In {
                left: operand,
                query,
            } => self.member_of(field_operand(operand)?, query, scope, c.span)?,
            ConditionKind::Saved { name, arguments } => {
                self.saved_cond(name, arguments, scope, c.span)?
            }
            ConditionKind::Sequence {
                steps,
                gaps,
                all: false,
                unit,
            } if bind_unit(unit)? == Unit::ToolCalls => self.seq(steps, gaps, scope, c.span)?,
            ConditionKind::Match { field, value } => {
                self.field_cond(field, FieldOp::Match(value), scope, c.span, top)?
            }
            ConditionKind::Compare { left, op, right } => {
                if let (
                    Expression::Field { name },
                    Expression::Value {
                        value: Value::Scalar { text, .. },
                    },
                ) = (left, right)
                    && !matches!(op, Op::Contains | Op::Prefix)
                    && (fields::lookup(scope.unit, name)
                        .or_else(|| {
                            fields::has_session(scope.unit)
                                .then(|| fields::lookup(Unit::Sessions, name))
                                .flatten()
                        })
                        .is_some_and(|field| {
                            matches!(
                                field.kind,
                                Kind::Number | Kind::Money | Kind::Duration | Kind::Time
                            )
                        })
                        || self
                            .expr(name, scope, c.span.start)
                            .is_ok_and(|(_, kind)| kind == ColKind::Time))
                {
                    self.field_cond(name, FieldOp::Compare(*op, text), scope, c.span, top)?
                } else {
                    self.expression_comparison(
                        left,
                        *op,
                        right,
                        c.span.start,
                        &mut |compiler, name, at| compiler.expr(name, scope, at),
                    )?
                }
            }
            ConditionKind::Since { value, .. } | ConditionKind::Until { value, .. } => {
                let field = if matches!(&c.kind, ConditionKind::Since { .. }) {
                    "since"
                } else {
                    "until"
                };
                self.field_cond(field, FieldOp::Match(value), scope, c.span, top)?
            }
            ConditionKind::Between { left, from, to } => {
                if let Expression::Field { name } = left {
                    self.field_cond(
                        name,
                        FieldOp::Between(value_text(from)?, value_text(to)?),
                        scope,
                        c.span,
                        top,
                    )?
                } else {
                    let lower = self.expression_comparison(
                        left,
                        Op::Ge,
                        &Expression::value(from.clone()),
                        c.span.start,
                        &mut |compiler, name, at| compiler.expr(name, scope, at),
                    )?;
                    let upper = self.expression_comparison(
                        left,
                        Op::Le,
                        &Expression::value(to.clone()),
                        c.span.start,
                        &mut |compiler, name, at| compiler.expr(name, scope, at),
                    )?;
                    format!("({lower}) AND ({upper})")
                }
            }
            ConditionKind::Rank {
                list: RankList::Similar,
                value,
            } => self.field_cond("similar", FieldOp::Match(value), scope, c.span, top)?,
            _ => return Err(not_built("this shared condition")),
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
            "last" | "latest" => {
                let inner = self.fresh();
                self.newest_session(&s, &inner, None)
            }
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
            id => self.session_reference(&format!("{s}.native_id"), "native_id", id),
        })
    }

    /// `{outer}.pk` is the most recently active main session read through
    /// `inner` that meets `within`, never the calling session: an agent
    /// asking for the last session means the one before its own.
    fn newest_session(&mut self, outer: &str, inner: &Aliases, within: Option<String>) -> String {
        let n = &inner.s;
        let mut conditions = vec![
            format!(
                "{n}.kind = '{}'",
                polylogue_model::SessionKind::Main.as_str()
            ),
            format!("{n}.stub = 0"),
        ];
        if let Some(me) = self.opts.self_session.clone() {
            conditions.push(format!("{n}.id <> {}", self.text_param(&me)));
        }
        conditions.extend(within);
        format!(
            "{outer}.pk = (SELECT {n}.pk FROM {} WHERE {} \
             ORDER BY {n}.last_activity_at IS NULL, {n}.last_activity_at DESC, {n}.pk DESC LIMIT 1)",
            base_from(Unit::Sessions, inner),
            conditions.join(" AND ")
        )
    }

    /// The one session whose `column` is `id`, or else starts with it: the
    /// caller checks that exactly one does before the query runs, so a
    /// prefix two sessions share is refused rather than answered with both.
    pub(super) fn session_reference(&mut self, expr: &str, column: &str, id: &str) -> String {
        let exact = self.text_param(id);
        let prefix = self.text_param(&like_prefix(id));
        self.id_checks.push(IdCheck {
            word: id.to_owned(),
            sql: format!(
                "SELECT id FROM session WHERE {column} = ?1 UNION \
                 SELECT id FROM session WHERE {column} LIKE ?2 ESCAPE '\\' \
                 AND NOT EXISTS (SELECT 1 FROM session WHERE {column} = ?1) LIMIT 5"
            ),
            params: vec![Param::Text(id.to_owned()), Param::Text(like_prefix(id))],
        });
        format!(
            "({expr} = {exact} OR (NOT EXISTS (SELECT 1 FROM session WHERE {column} = {exact}) \
             AND {expr} LIKE {prefix} ESCAPE '\\'))"
        )
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
        let key = if scope.unit == Unit::Sessions {
            format!("{s}.pk")
        } else {
            match units::spec(scope.unit).session_pk {
                "" | "NULL" => return Ok(self.session_reference(&format!("{s}.id"), "id", id)),
                template => scope.a.expand(template),
            }
        };
        // Resolve the reference to session keys once, so the row's own
        // session key is matched through its index.
        let resolved = self.session_reference("rs.id", "id", id);
        Ok(format!(
            "{key} IN (SELECT rs.pk FROM session rs WHERE {resolved})"
        ))
    }

    /// The alias of the session a row of this unit belongs to.
    pub(super) fn session_alias(
        &self,
        scope: &RowContext,
        span: Span,
    ) -> Result<String, QueryError> {
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

    fn native_reference(
        &mut self,
        text: &str,
        scope: &RowContext,
        span: Span,
    ) -> Result<String, QueryError> {
        let reference =
            polylogue_model::Reference::parse(text).map_err(|error| QueryError::Invalid {
                message: error.to_string(),
                position: span.start,
            })?;
        let key = scope.a.expand(units::spec(scope.unit).key);
        let members = crate::query::references::membership(scope.unit, &reference, &mut |value| {
            self.text_param(&value)
        })
        .ok_or_else(|| not_built("this reference at the selected row grain"))?;
        Ok(format!("{key} IN ({members})"))
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
        if let FieldOp::Match(Value::Scalar { text, .. }) = &op {
            if (lower == "message" && text.contains(".."))
                || (lower == "event" && scope.unit != Unit::Deliveries)
                || matches!(
                    lower.as_str(),
                    "block" | "document" | "span" | "assertion" | "delivery"
                )
            {
                return self.native_reference(&format!("{lower}:{text}"), scope, span);
            }
            if lower == "request" {
                polylogue_model::Reference::parse(&format!("request:{text}")).map_err(|error| {
                    QueryError::Invalid {
                        message: error.to_string(),
                        position: span.start,
                    }
                })?;
                return Err(not_built(
                    "accepted request objects have no query row-domain producer",
                ));
            }
            if lower == "run" {
                polylogue_model::RunId::parse(text).map_err(|error| QueryError::Invalid {
                    message: error.to_string(),
                    position: span.start,
                })?;
                return Err(not_built(
                    "run references require their acknowledged execution binding",
                ));
            }
            if self.opts.origins.contains(&lower) {
                return self.session_ref(&format!("{lower}:{text}"), scope, span);
            }
            match lower.as_str() {
                "session" => return self.session_ref(text, scope, span),
                "message" => {
                    return self.native_reference(&format!("message:{text}"), scope, span);
                }
                "result" => {
                    return Err(QueryError::Invalid {
                        message: "result:<id> is a source: start the query with it \
                                  (`result:<id> | count`); it cannot be a condition"
                            .into(),
                        position: span.start,
                    });
                }
                "page" => {
                    return Err(not_built("page: references (pages do not exist yet)"));
                }
                "file" => {
                    return Err(not_built(
                        "file:<path>@<commit> references (commits are not a unit yet)",
                    ));
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
        // `F:*` names a field without restricting it, whatever its rule.
        if matches!(op, FieldOp::Match(Value::Any)) {
            return Ok("1".into());
        }
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
            .expr(&a, self.opts.clock.now.as_millisecond())
            .expect("non-special fields have SQL");
        if top && matches!(op, FieldOp::Compare(..) | FieldOp::Between(..)) {
            self.show(def.name, &expr, def.column_kind(), def.reference());
        }
        if def.kind == Kind::Time
            && let Some(native) = fields::native_time(def.unit, def.name, &a)
        {
            return self.native_time_condition(def.name, &native, op, span);
        }
        // A structured value matches by its JSON scalar, so a string value
        // matches its text rather than its quoted JSON spelling.
        if def.column_kind() == ColKind::Json {
            return self.typed(def, &format!("json_extract({expr}, '$')"), op, span);
        }
        self.typed(def, &expr, op, span)
    }

    pub(super) fn native_time_test(
        &mut self,
        native: &fields::NativeTime,
        predicate: time::NativePredicate,
    ) -> String {
        let broad = predicate.candidates(&native.cache);
        let encoded = serde_json::to_string(&predicate).expect("typed time predicates serialize");
        let parameter = self.text_param(&encoded);
        let exact = format!(
            "pl_time_test({}, '{}', {parameter})",
            native.bytes,
            native.storage.code()
        );
        if broad == "1" {
            exact
        } else {
            format!(
                "(({broad} OR NOT pl_time_is_point({}, '{}')) AND {exact})",
                native.bytes,
                native.storage.code()
            )
        }
    }

    fn native_time_condition(
        &mut self,
        field: &str,
        native: &fields::NativeTime,
        op: FieldOp<'_>,
        span: Span,
    ) -> Result<String, QueryError> {
        use query_language::time::{TimeCondition as C, TimeRange};
        use time::NativePredicate as P;
        let compare = |value: Time, op: Op| -> Result<P, QueryError> {
            let (condition, invert) = match (op, value) {
                (Op::Eq | Op::Ne, Time::Instant { at }) => (C::Equal { at }, op == Op::Ne),
                (Op::Eq | Op::Ne, Time::Span { start, end }) => (
                    C::Within {
                        range: TimeRange { start, end },
                    },
                    op == Op::Ne,
                ),
                (Op::Gt, Time::Instant { at }) => (C::After { at }, false),
                (Op::Gt, Time::Span { end, .. }) => (C::AtOrAfter { at: end }, false),
                (Op::Ge, value) => (C::AtOrAfter { at: value.start() }, false),
                (Op::Lt, value) => (C::Before { at: value.start() }, false),
                (Op::Le, Time::Instant { at }) => (C::AtOrBefore { at }, false),
                (Op::Le, Time::Span { end, .. }) => (C::Before { at: end }, false),
                _ => return Err(not_built("this time comparison operator")),
            };
            Ok(P::Compare { condition, invert })
        };
        let predicate = match op {
            FieldOp::Match(Value::Any) => return Ok("1".into()),
            FieldOp::Match(Value::Unknown) => {
                return Ok(format!("NOT ({})", self.native_time_test(native, P::Known)));
            }
            FieldOp::Match(Value::Known) => return Ok(self.native_time_test(native, P::Known)),
            FieldOp::Match(Value::Scalar {
                text,
                quoted: false,
            }) if text.eq_ignore_ascii_case("unknown") => {
                return Ok(format!("NOT ({})", self.native_time_test(native, P::Known)));
            }
            FieldOp::Match(Value::Scalar { text, .. }) => {
                compare(self.time(field, text, span)?, Op::Eq)?
            }
            FieldOp::Compare(op, text) => compare(self.time(field, text, span)?, op)?,
            FieldOp::Between(from, to) => {
                let a = self.time(field, from, span)?;
                let b = self.time(field, to, span)?;
                if matches!(b, Time::Instant { at } if at == jiff::Timestamp::MAX) {
                    return Ok(self.native_time_test(
                        native,
                        P::Compare {
                            condition: C::AtOrAfter { at: a.start() },
                            invert: false,
                        },
                    ));
                }
                let end = match b {
                    Time::Instant { at } => at
                        .checked_add(jiff::SignedDuration::from_nanos(1))
                        .map_err(|error| QueryError::Invalid {
                            message: error.to_string(),
                            position: span.start,
                        })?,
                    Time::Span { end, .. } => end,
                };
                let range = TimeRange::new(a.start(), end)
                    .map_err(|error| QueryError::language(error, self.text))?;
                P::Compare {
                    condition: C::Within { range },
                    invert: false,
                }
            }
            _ => return Err(not_built("this time field condition")),
        };
        Ok(self.native_time_test(native, predicate))
    }

    pub(super) fn display_value(expr: &str, kind: ColKind) -> String {
        match kind {
            ColKind::Money => format!("({expr}) / 1e9"),
            ColKind::DurationSeconds => format!("({expr}) / 1000.0"),
            _ => expr.to_owned(),
        }
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
            let e = Self::display_value(expr, kind);
            self.shown.push((n, e, kind, reference));
        }
    }

    pub(super) fn unknown_field(&self, field: &str, unit: Unit, span: Span) -> QueryError {
        unknown_field(self.text, field, unit, span)
    }

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
        self.typed_as(def, expr, op, span, def.kind)
    }
    pub(super) fn typed_as(
        &mut self,
        def: &fields::Field,
        expr: &str,
        op: FieldOp<'_>,
        span: Span,
        kind: Kind,
    ) -> Result<String, QueryError> {
        let name = def.name;
        let bad = |value: &str, message: String| QueryError::BadValue {
            field: name.to_owned(),
            value: value.to_owned(),
            position: span.start,
            message,
        };
        let stores_unknown = def.stored_unknown.is_some();
        let known;
        let expr = if let Some(word) = def.stored_unknown {
            known = format!("NULLIF(lower({expr}), '{word}')");
            known.as_str()
        } else {
            expr
        };
        match (kind, op) {
            (_, FieldOp::Match(Value::Any)) => Ok("1".into()),
            (_, FieldOp::Match(Value::Unknown)) => Ok(format!("({expr}) IS NULL")),
            (_, FieldOp::Match(Value::Known)) => Ok(format!("({expr}) IS NOT NULL")),
            (
                _,
                FieldOp::Match(Value::Scalar {
                    text,
                    quoted: false,
                }),
            ) if text.eq_ignore_ascii_case("unknown") => Ok(format!("({expr}) IS NULL")),
            (Kind::Match, FieldOp::Match(v)) => {
                let values = alternatives(v);
                if let Some(set) = def.values {
                    for x in &values {
                        if !set.iter().any(|s| s.eq_ignore_ascii_case(x)) {
                            return Err(bad(x, closed_set_message(def, x)));
                        }
                    }
                }
                let (unknown, values): (Vec<&String>, Vec<&String>) = values
                    .iter()
                    .partition(|x| stores_unknown && x.eq_ignore_ascii_case("unknown"));
                let ps: Vec<String> = values
                    .iter()
                    .map(|x| self.text_param(&x.to_lowercase()))
                    .collect();
                let listed = format!("lower({expr}) IN ({})", ps.join(", "));
                Ok(match (unknown.is_empty(), ps.is_empty()) {
                    (true, _) => listed,
                    (false, true) => format!("({expr}) IS NULL"),
                    (false, false) => format!("({listed} OR ({expr}) IS NULL)"),
                })
            }
            (Kind::Match, FieldOp::Compare(op, value)) => {
                let p = self.text_param(&value.to_lowercase());
                Ok(format!("lower({expr}) {} {p}", op.sql()))
            }
            (Kind::Prefix, FieldOp::Compare(op @ (Op::Eq | Op::Ne), value)) => {
                let p = self.text_param(value);
                Ok(format!("{expr} {} {p}", op.sql()))
            }
            (Kind::Prefix, FieldOp::Match(v)) => {
                let values = alternatives(v);
                let mut parts = vec![];
                for x in values {
                    if def.reference() == Some(ColumnReference::Session) {
                        parts.push(self.session_reference(expr, "id", &x));
                        continue;
                    }
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
            (Kind::Bool, FieldOp::Match(Value::Scalar { text, .. })) => {
                let b = match text.to_ascii_lowercase().as_str() {
                    "true" | "yes" | "1" => 1,
                    "false" | "no" | "0" => 0,
                    _ => return Err(bad(text, format!("{name} is true or false"))),
                };
                Ok(format!("({expr}) = {b}"))
            }
            (
                Kind::Number | Kind::Money | Kind::Duration,
                FieldOp::Match(Value::Scalar { text, .. }),
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
            (Kind::Time, FieldOp::Match(Value::Scalar { text, .. })) => {
                let t = self.time(name, text, span)?;
                Ok(format!(
                    "(({expr}) >= {} AND ({expr}) < {})",
                    t.start_ms(),
                    t.end_ms()?
                ))
            }
            (Kind::Time, FieldOp::Compare(op, value)) => {
                let t = self.time(name, value, span)?;
                Ok(match op {
                    Op::Gt => format!("({expr}) >= {}", t.end_ms()?),
                    Op::Ge => format!("({expr}) >= {}", t.start_ms()),
                    Op::Lt => format!("({expr}) < {}", t.start_ms()),
                    Op::Le => format!("({expr}) < {}", t.end_ms()?),
                    Op::Eq => format!(
                        "(({expr}) >= {} AND ({expr}) < {})",
                        t.start_ms(),
                        t.end_ms()?
                    ),
                    Op::Ne => format!(
                        "(({expr}) < {} OR ({expr}) >= {})",
                        t.start_ms(),
                        t.end_ms()?
                    ),
                    Op::Contains | Op::Prefix => {
                        return Err(not_built("this time comparison operator"));
                    }
                })
            }
            (Kind::Time, FieldOp::Between(from, to)) => {
                let a = self.time(name, from, span)?;
                let b = self.time(name, to, span)?;
                Ok(format!(
                    "(({expr}) >= {} AND ({expr}) < {})",
                    a.start_ms(),
                    b.end_ms()?
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
        def.kind
            .scalar(def.name, text)
            .map_err(|message| QueryError::BadValue {
                field: def.name.to_owned(),
                value: text.to_owned(),
                position: span.start,
                message,
            })
    }

    pub(super) fn time(&self, field: &str, text: &str, span: Span) -> Result<Time, QueryError> {
        let _ = field;
        time::resolve(text, &self.opts.clock)
            .map(|resolved| {
                let value = resolved.value;
                let mut resolutions = self.resolved_times.borrow_mut();
                if !resolutions.contains(&resolved) {
                    resolutions.push(resolved);
                }
                value
            })
            .map_err(|mut error| {
                error.position = span.start;
                QueryError::language(error, self.text)
            })
    }

    /// `path` and the directories below it, by whole components, as a key
    /// range over `expr` that an index on it serves: `path` itself, then
    /// every value that starts with `path/` (`0` is the character after `/`).
    pub(super) fn directory_range(&mut self, expr: &str, path: &str) -> String {
        let base = path.trim_end_matches('/');
        let exact = self.text_param(if base.is_empty() { "/" } else { base });
        let first = self.text_param(base);
        let below = self.text_param(&format!("{base}/"));
        let end = self.text_param(&format!("{base}0"));
        format!(
            "({expr} >= {first} AND {expr} < {end} AND ({expr} = {exact} OR {expr} >= {below}))"
        )
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

fn search_word(text: &str, mode: TextMode) -> Result<String, QueryError> {
    match mode {
        TextMode::Prefix => Ok(format!("{text}*")),
        TextMode::Word | TextMode::Phrase => Ok(text.to_owned()),
        TextMode::Approximate => Err(not_built("approximate text search")),
    }
}

pub(super) fn unknown_field(text: &str, field: &str, unit: Unit, span: Span) -> QueryError {
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
            &text[..span.start],
            one,
            &text[span.start + field.len()..]
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

/// `last` or `latest`: the newest session, resolved inside the query.
fn newest_word(condition: &Condition) -> bool {
    match &condition.kind {
        ConditionKind::Reference { reference } => matches!(reference.as_str(), "last" | "latest"),
        ConditionKind::Text {
            text,
            mode: TextMode::Word,
        } => matches!(text.as_str(), "last" | "latest"),
        _ => false,
    }
}
