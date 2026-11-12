//! Conditions lowered to SQL from their bound form ([`Filter`]): search
//! words, references, and each declared field compared by the stored form
//! binding gave its value.

use super::*;
use polylogue_query::filter::{Binder, Comparison, FieldTest, Filter, Test, TimeTest};

impl<'a> Compiler<'a> {
    /// What binding reads from the query.
    pub(super) fn binder(&self) -> Binder<'a> {
        Binder {
            text: self.text,
            origins: &self.opts.origins,
            clock: &self.opts.clock,
        }
    }

    /// A condition as written, over `scope`'s rows. `top` is true for the
    /// query's own `where`, where a top-level text term of the search
    /// leg's index ranks the rows.
    pub(super) fn cond(
        &mut self,
        c: &Condition,
        scope: &RowContext,
        top: bool,
    ) -> Result<String, QueryError> {
        let filter = self.binder().bind(c, scope.unit)?;
        self.lower(&filter, scope, top)
    }

    /// A bound condition over `scope`'s rows.
    pub(super) fn lower(
        &mut self,
        filter: &Filter,
        scope: &RowContext,
        top: bool,
    ) -> Result<String, QueryError> {
        query_language::depth::deeper(|| {
            Ok(match filter {
                Filter::All(all) => self.lower_all(all, scope, top)?,
                Filter::Any(any) => {
                    let parts = any
                        .iter()
                        .map(|x| self.lower(x, scope, false))
                        .collect::<Result<Vec<_>, _>>()?;
                    format!("({})", parts.join(" OR "))
                }
                Filter::Not(inner) => format!("NOT ({})", self.lower(inner, scope, false)?),
                Filter::Word(word) => self.text(
                    &[(word.text.clone(), word.phrase, word.span)],
                    scope,
                    Index::Prose,
                    top,
                )?,
                Filter::SessionWord { word, span } => self.reference_word(word, scope, *span)?,
                Filter::Session { id, span } => self.session_ref(id, scope, *span)?,
                Filter::Url { url, span } => self.url(url, scope, *span)?,
                Filter::Rows { reference, span } => {
                    self.native_reference(reference, scope, *span)?
                }
                Filter::Related {
                    prefix,
                    target,
                    filter,
                    span,
                } => self.related(prefix, *target, filter, scope, *span)?,
                Filter::Field(test) => self.field(test, scope, top)?,
                Filter::Written(c) => self.written(c, scope)?,
            })
        })
    }

    /// Conditions joined by AND. Search words among them must occur in the
    /// same message; `since:` and `until:` on one native time read it once.
    fn lower_all(
        &mut self,
        all: &[Filter],
        scope: &RowContext,
        top: bool,
    ) -> Result<String, QueryError> {
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
            fn flatten<'b>(all: &'b [Filter], output: &mut Vec<&'b Filter>) {
                query_language::depth::deeper(|| {
                    for filter in all {
                        if let Filter::All(inner) = filter {
                            flatten(inner, output);
                        } else {
                            output.push(filter);
                        }
                    }
                })
            }
            flatten(all, &mut conjuncts);
        } else {
            conjuncts.extend(all);
        }
        // `last` beside other session conditions is the newest session that
        // also meets them.
        if scope.unit == Unit::Sessions
            && conjuncts.len() > 1
            && let Some(at) = conjuncts.iter().position(|x| {
                matches!(x, Filter::SessionWord { word, .. } if matches!(word.as_str(), "last" | "latest"))
            })
        {
            let rest = Filter::All(
                conjuncts
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != at)
                    .map(|(_, x)| (*x).clone())
                    .collect(),
            );
            let inner = RowContext {
                unit: Unit::Sessions,
                a: self.fresh(),
            };
            let within = self.lower(&rest, &inner, false)?;
            return Ok(self.newest_session(&scope.a.s, &inner.a, Some(within)));
        }
        let bound = |x: &Filter, name: &str| match x {
            Filter::Field(FieldTest {
                field,
                test: Test::Time(TimeTest::Within(at)),
                ..
            }) if field.declaration().name == name => Some(at.clone()),
            _ => None,
        };
        let combine_bounds = native_bounds.is_some()
            && conjuncts.iter().any(|x| bound(x, "since").is_some())
            && conjuncts.iter().any(|x| bound(x, "until").is_some());
        let mut since = None;
        let mut until = None;
        let mut parts = vec![];
        let mut words = vec![];
        for x in conjuncts {
            if combine_bounds {
                if let Some(at) = bound(x, "since") {
                    let at = self.resolved(&at).start();
                    since = Some(since.map_or(at, |old: jiff::Timestamp| old.max(at)));
                    continue;
                }
                if let Some(at) = bound(x, "until") {
                    let at = self.resolved(&at).until_bound();
                    until = Some(until.map_or(at, |old: jiff::Timestamp| old.min(at)));
                    continue;
                }
            }
            match x {
                Filter::Word(word) if text_unit(scope.unit) => {
                    words.push((word.text.clone(), word.phrase, word.span));
                }
                _ => parts.push(self.lower(x, scope, top)?),
            }
        }
        if let Some(native) = native_bounds.filter(|_| combine_bounds) {
            parts.push(
                self.native_time_test(&native, time::NativePredicate::Overlap { since, until }),
            );
        }
        if !words.is_empty() {
            parts.push(self.text(&words, scope, Index::Prose, top)?);
        }
        Ok(format!("({})", parts.join(" AND ")))
    }

    /// A condition binding left as written: over other rows, another
    /// query, or computed values.
    fn written(&mut self, c: &Condition, scope: &RowContext) -> Result<String, QueryError> {
        match &c.kind {
            ConditionKind::Exists { unit, condition } => {
                self.exists(bind_unit(unit)?, condition.as_deref(), scope, c.span)
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
            ),
            ConditionKind::In {
                left: operand,
                query,
            } => self.member_of(field_operand(operand)?, query, scope, c.span),
            ConditionKind::Saved { name, arguments } => {
                self.saved_cond(name, arguments, scope, c.span)
            }
            ConditionKind::Sequence { steps, gaps, .. } => self.seq(steps, gaps, scope, c.span),
            ConditionKind::Compare { left, op, right } => self.expression_comparison(
                left,
                *op,
                right,
                c.span.start,
                &mut |compiler, name, at| compiler.expr(name, scope, at),
            ),
            ConditionKind::Between { left, from, to } => {
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
                Ok(format!("({lower}) AND ({upper})"))
            }
            _ => Err(not_built("this shared condition")),
        }
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
        let Some(hits) = rows(a, &units::Hits { cte: &cte, index }) else {
            self.params.pop();
            return Err(refuse(scope.unit));
        };
        let ranks = match index {
            Index::Prose => Some(polylogue_query::search::SnippetIndex::Prose),
            Index::Output => Some(polylogue_query::search::SnippetIndex::Output),
            Index::Thinking => None,
        };
        let mut prefix = String::new();
        if let Some(rank_index) = ranks
            && top
            && Some(rank_index) == self.rank_index
            && self.rank.is_none()
        {
            self.rank = Some(hits.rank);
            if self.answer
                && !self.measuring
                && self.near_request.is_none()
                && self.ranked_search.is_none()
            {
                let limit = self.params.len() + 1;
                let bound = self.bind(Param::Int(-1));
                self.ranked_search = Some(RankedSearch {
                    index: rank_index,
                    expression: match_text,
                    limit,
                });
                prefix = format!(" ORDER BY {table}.rank LIMIT {bound}");
            }
        }
        self.ctes.push(format!(
            "{cte} AS MATERIALIZED (SELECT rowid AS text_id, bm25({table}) AS rank FROM {table} WHERE {table} MATCH {param}{prefix})"
        ));
        self.ctes.extend(hits.ctes);
        Ok(hits.condition)
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
        if fields::has_session(child.unit) {
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

    /// A condition on one declared field.
    fn field(
        &mut self,
        t: &FieldTest,
        scope: &RowContext,
        top: bool,
    ) -> Result<String, QueryError> {
        let def = t.field.declaration();
        // `F:*` names a field without restricting it, whatever its rule.
        if matches!(t.test, Test::Any) {
            return Ok("1".into());
        }
        let rule = fields::rule(t.field);
        let a = scope.a.clone();
        if let Some(compile) = rule.compile {
            let call = custom_fields::Call {
                unit: def.unit,
                name: def.name,
                test: &t.test,
                scope: &RowContext { unit: def.unit, a },
                span: t.span,
                top,
            };
            return compile(self, &call);
        }
        let expr = rule
            .expr(&a, self.opts.clock.now.as_millisecond())
            .expect("non-special fields have SQL");
        if top && t.test.is_comparison() {
            self.show(def.name, &expr, def.column_kind(), def.column_reference());
        }
        if def.kind == Kind::Time
            && let Some(native) = fields::native_time(def.unit, def.name, &a)
        {
            return self.native_time_condition(&native, &t.test, t.span);
        }
        // A structured value matches by its JSON scalar, so a string value
        // matches its text rather than its quoted JSON spelling.
        if def.column_kind() == ColKind::Json {
            return self.typed(
                def,
                def.kind,
                &format!("json_extract({expr}, '$')"),
                &t.test,
            );
        }
        self.typed(def, def.kind, &expr, &t.test)
    }

    /// `comparison`, as written, on `expr`, a value of `def` compared as
    /// `kind`: for a field whose own rule compares a value it computes.
    pub(super) fn compare_as(
        &mut self,
        def: &fields::Field,
        kind: Kind,
        expr: &str,
        comparison: &Comparison,
        span: Span,
    ) -> Result<String, QueryError> {
        let test = self.binder().test(def, kind, comparison, span)?;
        self.typed(def, kind, expr, &test)
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
        native: &fields::NativeTime,
        test: &Test,
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
        let predicate = match test {
            Test::Any => return Ok("1".into()),
            Test::Unknown => {
                return Ok(format!("NOT ({})", self.native_time_test(native, P::Known)));
            }
            Test::Known => return Ok(self.native_time_test(native, P::Known)),
            Test::Time(TimeTest::Within(at)) => compare(self.resolved(at), Op::Eq)?,
            Test::Time(TimeTest::Compare(op, at)) => compare(self.resolved(at), *op)?,
            Test::Time(TimeTest::Between(from, to)) => {
                let a = self.resolved(from);
                let b = self.resolved(to);
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

    /// A condition on a related unit's row (`session.repo:x`), read through
    /// the relation the unit declares under `prefix`.
    fn related(
        &mut self,
        prefix: &str,
        target: Unit,
        filter: &Filter,
        scope: &RowContext,
        span: Span,
    ) -> Result<String, QueryError> {
        let a = &scope.a;
        let rel = units::spec(scope.unit)
            .relation(prefix)
            .ok_or_else(|| QueryError::Invalid {
                message: format!("{prefix}. does not apply to {}", scope.unit.name()),
                position: span.start,
            })?;
        match rel.via {
            Via::Same => {
                let inner = RowContext {
                    unit: target,
                    a: a.clone(),
                };
                self.lower(filter, &inner, false)
            }
            Via::Session(alias) => {
                let inner = RowContext {
                    unit: Unit::Sessions,
                    a: Aliases {
                        s: a.expand(alias),
                        ..a.clone()
                    },
                };
                self.lower(filter, &inner, false)
            }
            Via::Exists { cond: template, .. } => {
                let n = self.fresh();
                let inner = RowContext {
                    unit: target,
                    a: n.clone(),
                };
                let cond = self.lower(filter, &inner, false)?;
                Ok(relation_sql(template, a, &n, target, &cond))
            }
        }
    }

    /// A bound test on `expr`, a value of `def` compared as `kind`.
    pub(super) fn typed(
        &mut self,
        def: &fields::Field,
        kind: Kind,
        expr: &str,
        test: &Test,
    ) -> Result<String, QueryError> {
        // A closed set's values are stored in their declared spelling, so
        // the column is compared as stored and its indexes can serve it.
        let closed = kind == Kind::Match && def.values.is_some();
        let known;
        let expr = match def.stored_unknown {
            Some(word) if closed => {
                known = format!("NULLIF({expr}, '{word}')");
                known.as_str()
            }
            Some(word) => {
                known = format!("NULLIF(lower({expr}), '{word}')");
                known.as_str()
            }
            None => expr,
        };
        Ok(match test {
            Test::Any => "1".into(),
            Test::Unknown => format!("({expr}) IS NULL"),
            Test::Known => format!("({expr}) IS NOT NULL"),
            Test::OneOf { values, unknown } => {
                let ps: Vec<String> = values.iter().map(|x| self.text_param(x)).collect();
                let listed = if closed {
                    format!("{expr} IN ({})", ps.join(", "))
                } else {
                    format!("lower({expr}) IN ({})", ps.join(", "))
                };
                match (*unknown, values.is_empty()) {
                    (false, _) => listed,
                    (true, true) => format!("({expr}) IS NULL"),
                    (true, false) => format!("({listed} OR ({expr}) IS NULL)"),
                }
            }
            Test::TextCompare(op, value) => {
                let p = self.text_param(value);
                format!("lower({expr}) {} {p}", op.sql())
            }
            Test::PrefixCompare(op, value) => {
                let p = self.text_param(value);
                format!("{expr} {} {p}", op.sql())
            }
            Test::Prefix(values) => {
                let mut parts = vec![];
                for x in values {
                    if def.column_reference() == Some(ColumnReference::Session) {
                        parts.push(self.session_reference(expr, "id", x));
                        continue;
                    }
                    let exact = self.text_param(x);
                    let prefix = self.text_param(&like_prefix(x));
                    parts.push(format!(
                        "{expr} = {exact} OR {expr} LIKE {prefix} ESCAPE '\\'"
                    ));
                }
                format!("({})", parts.join(" OR "))
            }
            Test::Substring(values) => {
                let mut parts = vec![];
                for x in values {
                    let p = self.text_param(x);
                    parts.push(format!("instr(pl_key({expr}), pl_key({p})) > 0"));
                }
                format!("({})", parts.join(" OR "))
            }
            Test::Subtree(values) => {
                let mut parts = vec![];
                for x in values {
                    parts.push(self.subtree(expr, x));
                }
                format!("({})", parts.join(" OR "))
            }
            Test::Bool(b) => format!("({expr}) = {}", u8::from(*b)),
            Test::NumberEquals(v) => format!("({expr}) = {v}"),
            Test::NumberCompare(op, v) => format!("({expr}) {} {v}", op.sql()),
            Test::NumberBetween(a, b) => format!("(({expr}) >= {a} AND ({expr}) <= {b})"),
            Test::Time(TimeTest::Within(at)) => {
                let t = self.resolved(at);
                format!(
                    "(({expr}) >= {} AND ({expr}) < {})",
                    t.start_ms(),
                    t.end_ms()?
                )
            }
            Test::Time(TimeTest::Compare(op, at)) => {
                let t = self.resolved(at);
                match op {
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
                }
            }
            Test::Time(TimeTest::Between(from, to)) => {
                let a = self.resolved(from);
                let b = self.resolved(to);
                format!(
                    "(({expr}) >= {} AND ({expr}) < {})",
                    a.start_ms(),
                    b.end_ms()?
                )
            }
            // Binding gives a written value only to a field with its own
            // rule, which reads it there.
            Test::Written(_) => unreachable!("{} has its own rule", def.name),
        })
    }

    /// A time binding resolved, recorded for the answer's explanation.
    pub(super) fn resolved(&self, resolved: &query_language::time::ResolvedTime) -> Time {
        let mut resolutions = self.resolved_times.borrow_mut();
        if !resolutions.contains(resolved) {
            resolutions.push(resolved.clone());
        }
        resolved.value
    }

    /// A time expression in a computed value, resolved and recorded.
    pub(super) fn time(&self, text: &str, span: Span) -> Result<Time, QueryError> {
        Ok(self.resolved(&self.binder().time(text, span)?))
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

pub(super) use polylogue_query::filter::unknown_field;
