//! Conditions lowered to SQL from their bound form ([`Filter`]): search
//! words, references, and each declared field compared by the stored form
//! binding gave its value.

use super::*;
use polylogue_query::filter::{FieldTest, Filter, Special, Test, TimeTest};

impl<'a> Compiler<'a> {
    /// A bound condition over `scope`'s rows. At the selection's top level
    /// (`cx.top`), the first text term over the search leg's index puts
    /// its ranking there.
    pub(super) fn lower(
        &mut self,
        filter: &Filter,
        scope: &RowContext,
        cx: Cx<'_>,
    ) -> Result<String, QueryError> {
        query_language::depth::deeper(|| {
            Ok(match filter {
                Filter::All(all) => self.lower_all(all, scope, cx)?,
                Filter::Any(any) => {
                    let parts = any
                        .iter()
                        .map(|x| self.lower(x, scope, cx.inner()))
                        .collect::<Result<Vec<_>, _>>()?;
                    format!("({})", parts.join(" OR "))
                }
                Filter::Not(inner) => format!("NOT ({})", self.lower(inner, scope, cx.inner())?),
                Filter::Word(word) => self.text(
                    &[(word.text.clone(), word.phrase, word.span)],
                    scope,
                    Index::Prose,
                    cx,
                )?,
                Filter::SessionWord { word, span } => self.reference_word(word, scope, *span)?,
                Filter::Session { id, span } => self.session_ref(id, scope, *span)?,
                Filter::Rows { reference, span } => {
                    self.reference_membership(reference, scope, *span)?
                }
                Filter::Related {
                    prefix,
                    target,
                    filter,
                    span,
                } => self.related(prefix, *target, filter, scope, *span, cx)?,
                Filter::Field(test) => self.field(test, scope, cx)?,
                Filter::Saved(relation) => self.saved_relation(relation, scope, cx)?,
                Filter::Exists { rows, span } => self.exists(rows, scope, *span, cx.inner())?,
                Filter::Count {
                    rows,
                    op,
                    count,
                    span,
                } => self.counted(rows, (*op, *count), scope, *span, cx.inner())?,
                Filter::In { field, query, span } => {
                    self.member_of(field, query, scope, *span, cx.inner())?
                }
                Filter::Sequence { steps, gaps, span } => {
                    self.seq(steps, gaps, scope, *span, cx.inner())?
                }
                Filter::Compare {
                    left,
                    op,
                    right,
                    span,
                } => self.expression_comparison(
                    left,
                    *op,
                    right,
                    span.start,
                    &mut |compiler, name, at| compiler.expr(name, scope, at),
                )?,
                Filter::Between {
                    left,
                    from,
                    to,
                    span,
                } => {
                    let lower = self.expression_comparison(
                        left,
                        Op::Ge,
                        &Expression::value(from.clone()),
                        span.start,
                        &mut |compiler, name, at| compiler.expr(name, scope, at),
                    )?;
                    let upper = self.expression_comparison(
                        left,
                        Op::Le,
                        &Expression::value(to.clone()),
                        span.start,
                        &mut |compiler, name, at| compiler.expr(name, scope, at),
                    )?;
                    format!("({lower}) AND ({upper})")
                }
            })
        })
    }

    /// Conditions joined by AND. Search words among them must occur in the
    /// same message; `since:` and `until:` on one stored time read it once.
    fn lower_all(
        &mut self,
        all: &[Filter],
        scope: &RowContext,
        cx: Cx<'_>,
    ) -> Result<String, QueryError> {
        let time_bounds = if scope.unit != Unit::ToolCalls {
            fields::time_window(scope.unit, true, &scope.a).filter(|lower| {
                fields::time_window(scope.unit, false, &scope.a).is_some_and(|upper| {
                    lower.bytes == upper.bytes && lower.storage == upper.storage
                })
            })
        } else {
            None
        };
        let mut conjuncts = vec![];
        if time_bounds.is_some() {
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
        // `latest` beside other session conditions is the newest session that
        // also meets them.
        if scope.unit == Unit::Sessions
            && conjuncts.len() > 1
            && let Some(at) = conjuncts.iter().position(|x| {
                matches!(x, Filter::SessionWord { word, .. } if matches!(word.as_str(), "latest"))
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
            let within = self.lower(&rest, &inner, cx.inner())?;
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
        let combine_bounds = time_bounds.is_some()
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
                Filter::Word(word) if polylogue_query::units::spec(scope.unit).words.plain() => {
                    words.push((word.text.clone(), word.phrase, word.span));
                }
                _ => parts.push(self.lower(x, scope, cx)?),
            }
        }
        if let Some(stored) = time_bounds.filter(|_| combine_bounds) {
            parts.push(self.time_test(&stored, time::TimePredicate::Overlap { since, until }));
        }
        if !words.is_empty() {
            parts.push(self.text(&words, scope, Index::Prose, cx)?);
        }
        Ok(format!("({})", parts.join(" AND ")))
    }

    /// A full-text condition over one index. Terms ANDed together must all
    /// occur in the same message (or block). At the top level, the first
    /// term over the search leg's index ranks the rows.
    pub(super) fn text(
        &mut self,
        words: &[(String, bool, Span)],
        scope: &RowContext,
        index: Index,
        cx: Cx<'_>,
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
        let match_text = if phrases.len() > 1 {
            format!("({})", phrases.join(" AND "))
        } else {
            phrases.join(" AND ")
        };
        let snippet_index = match index {
            Index::Prose => Some(polylogue_query::search::SnippetIndex::Prose),
            Index::Output => Some(polylogue_query::search::SnippetIndex::Output),
            Index::Thinking => None,
        };
        // The first top-level term over the search leg's index ranks.
        let top = cx.top.filter(|top| {
            snippet_index.is_some() && snippet_index == top.index && top.ranking.borrow().is_none()
        });
        let alternate_languages = (scope.unit == Unit::Messages && index == Index::Prose)
            .then_some(cx.alternates)
            .flatten();
        let source = alternate_languages.is_none() && top.is_some();
        let plans = source && top.is_some_and(|top| top.plans);
        let scoped = plans && matches!(scope.unit, Unit::Sessions | Unit::Messages);
        let parameter_start = self.params.len();
        let param = self.text_param(&match_text);
        let cte = format!("h{}", self.ctes.len() + 1);
        let candidates = format!("{cte}_scope_candidates");
        let hits = |scope| {
            rows(
                a,
                &units::Hits {
                    cte: &cte,
                    index,
                    source,
                    scope,
                },
            )
        };
        let Some(hits_relation) = hits(None) else {
            self.params.truncate(parameter_start);
            return Err(refuse(scope.unit));
        };
        let scoped_ctes = scoped
            .then(|| hits(Some(candidates.as_str())).map(|scoped| scoped.ctes))
            .flatten();
        let alternate = if let Some(languages) = alternate_languages {
            let alternates = format!("{cte}_alternates");
            self.ctes.push(format!("{alternates} AS MATERIALIZED (SELECT vt.alternate,vt.n,bm25(alternate_fts) AS rank FROM alternate_fts JOIN alternate_text vt ON vt.rowid=alternate_fts.rowid WHERE alternate_fts MATCH {param})"));
            let mut language = Vec::new();
            for lang in languages {
                language.push(self.text_param(lang));
            }
            let language = if language.is_empty() {
                String::new()
            } else {
                format!(" AND av.to_lang IN ({})", language.join(","))
            };
            let from = format!(
                "FROM {alternates} ah JOIN source_alternate_part ap ON ap.alternate=ah.alternate AND ap.n=ah.n JOIN source_alternate av ON av.id=ap.alternate WHERE {} AND {}{language}",
                crate::alternates::current_sql("av"),
                fields::message_alternate_match("av", "ap", &a.m)
            );
            Some((
                format!("EXISTS(SELECT 1 {from})"),
                format!("(SELECT min(ah.rank) {from})"),
            ))
        } else {
            None
        };
        let start = self.ctes.len();
        let leg = ranking::SearchLeg {
            index: snippet_index.unwrap_or(polylogue_query::search::SnippetIndex::Prose),
            expression: match_text,
            parameter: param,
            name: cte,
            ctes: start..start,
            scoped: scoped_ctes,
            spans: words.iter().map(|word| word.2).collect(),
        };
        self.ctes.push(leg.indexed(None));
        self.ctes.extend(hits_relation.ctes);
        if let Some(top) = top {
            let leg = ranking::SearchLeg {
                ctes: start..self.ctes.len(),
                ..leg
            };
            *top.ranking.borrow_mut() = Some(ranking::TextRank {
                expression: match &alternate {
                    Some((_, rank)) => format!(
                        "min(coalesce({},1e30),coalesce({rank},1e30))",
                        hits_relation.rank
                    ),
                    None => hits_relation.rank,
                },
                source: hits_relation.source,
                leg: plans.then_some(leg),
            });
        }
        Ok(match alternate {
            Some((condition, _)) => format!("({} OR {condition})", hits_relation.condition),
            None => hits_relation.condition,
        })
    }

    /// A word that names something: `latest`, `self`, `here`, or any other
    /// word, read as a native session ID or unique prefix.
    pub(super) fn reference_word(
        &mut self,
        word: &str,
        scope: &RowContext,
        span: Span,
    ) -> Result<String, QueryError> {
        let s = self.session_alias(scope, span)?;
        Ok(match word {
            "latest" => {
                let inner = self.fresh();
                self.newest_session(&s, &inner, None)
            }
            "here" => {
                return super::custom_fields::here(
                    self,
                    &super::custom_fields::Call {
                        unit: scope.unit,
                        name: "here",
                        test: &Test::Special(Special::One {
                            text: "here".into(),
                            quoted: false,
                        }),
                        scope,
                        span,
                        cx: Cx::plain(false),
                    },
                    false,
                );
            }
            "self" => {
                let Some(me) = self.opts.self_session.clone() else {
                    return Err(QueryError::NotFound {
                        message: "`self` names the calling session, and this caller is not one"
                            .into(),
                    });
                };
                let p = self.text_param(&me);
                format!("{s}.id = {p}")
            }
            id => self.session_reference(&format!("{s}.native_id"), "native_id", id, true),
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
    /// caller checks that at most one does before the query runs, so a
    /// prefix two sessions share is refused rather than answered with both,
    /// and that a `reference` names one. An ID that names no session
    /// compares with the empty string, which no ID is, so on a row that has
    /// the column the condition is false rather than unknown, and its
    /// negation holds.
    pub(super) fn session_reference(
        &mut self,
        expr: &str,
        column: &str,
        id: &str,
        reference: bool,
    ) -> String {
        let exact = self.text_param(id);
        let prefix = self.text_param(&like_prefix(id));
        self.id_checks.push(IdCheck {
            word: id.to_owned(),
            reference,
            sql: format!(
                "WITH exact AS MATERIALIZED (SELECT id FROM session WHERE {column}=?1) \
                 SELECT id FROM exact UNION SELECT s.id \
                 FROM (SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM exact)) missing \
                 CROSS JOIN session s WHERE s.{column} LIKE ?2 ESCAPE '\\' LIMIT 5"
            ),
            params: vec![Param::Text(id.to_owned()), Param::Text(like_prefix(id))],
        });
        // The check above refuses ambiguous and absent references. Resolve the
        // value once, with a short-circuited fallback, so a full ID remains
        // an indexed equality when it drives a scope.
        format!(
            "{expr}=coalesce(\
             (SELECT {column} FROM session WHERE {column}={exact} LIMIT 1),\
             (SELECT {column} FROM session WHERE {column} LIKE {prefix} ESCAPE '\\' LIMIT 1),\
             '')"
        )
    }

    /// `origin:native-id` (or a unique prefix of it) as a session condition.
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
                "" | "NULL" => {
                    return Ok(self.session_reference(&format!("{s}.id"), "id", id, true));
                }
                template => scope.a.expand(template),
            }
        };
        // Resolve the reference once, so the row's own session key can use its index.
        let resolved = self.session_reference("rs.id", "id", id, true);
        // One resolved session has at most one non-null key. For tool calls,
        // IS preserves a miss as false and lets SQLite prefer the session
        // index over the action index, whose common actions span the archive.
        let comparison = if scope.unit == Unit::ToolCalls {
            "IS"
        } else {
            "IN"
        };
        Ok(format!(
            "{key} {comparison} (SELECT rs.pk FROM session rs WHERE {resolved})"
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

    fn reference_membership(
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
        cx: Cx<'_>,
    ) -> Result<String, QueryError> {
        let def = t.field.declaration();
        // `F:*` names a field without restricting it; the one exception is
        // `queries where name:*`, which selects the named queries.
        if matches!(t.test, Test::Any) {
            return Ok(if def.unit == Unit::Queries && def.name == "name" {
                scope.a.expand("{r}.name IS NOT NULL")
            } else {
                "1".into()
            });
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
                cx,
            };
            return compile(self, &call);
        }
        let expr = rule
            .expr(&a, self.opts.clock.now.as_millisecond())
            .expect("non-special fields have SQL");
        if cx.is_top() && t.test.is_comparison() {
            self.show(def.name, &expr, def.column_kind(), def.column_reference());
        }
        if def.kind == Kind::Time
            && let Some(stored) = fields::stored_time(def.unit, def.name, &a)
        {
            return self.time_condition(&stored, &t.test, t.span);
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

    pub(super) fn time_test(
        &mut self,
        stored: &fields::StoredTime,
        predicate: time::TimePredicate,
    ) -> String {
        let broad = predicate.candidates(&stored.cache);
        let encoded = serde_json::to_string(&predicate).expect("typed time predicates serialize");
        let parameter = self.text_param(&encoded);
        let exact = format!(
            "pl_time_test({}, '{}', {parameter})",
            stored.bytes,
            stored.storage.as_str()
        );
        if broad == "1" {
            return exact;
        }
        // A period can meet a bound its cached start misses, so the rows
        // that may hold one are tested whatever their cache says. Where the
        // field names them (`periods`), no other row's stored time is read.
        let may_be_period = match &stored.periods {
            Some(periods) => periods.clone(),
            None => format!(
                "NOT pl_time_is_instant({}, '{}')",
                stored.bytes,
                stored.storage.as_str()
            ),
        };
        format!("(({broad} OR {may_be_period}) AND {exact})")
    }

    fn time_condition(
        &mut self,
        stored: &fields::StoredTime,
        test: &Test,
        span: Span,
    ) -> Result<String, QueryError> {
        use query_language::time::{TimeCondition as C, TimeRange};
        use time::TimePredicate as P;
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
                return Ok(format!("NOT ({})", self.time_test(stored, P::Known)));
            }
            Test::Known => return Ok(self.time_test(stored, P::Known)),
            Test::Time(TimeTest::Within(at)) => compare(self.resolved(at), Op::Eq)?,
            Test::Time(TimeTest::Compare(op, at)) => compare(self.resolved(at), *op)?,
            Test::Time(TimeTest::Between(from, to)) => {
                let a = self.resolved(from);
                let b = self.resolved(to);
                if matches!(b, Time::Instant { at } if at == jiff::Timestamp::MAX) {
                    return Ok(self.time_test(
                        stored,
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
        Ok(self.time_test(stored, predicate))
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
        let n = column_name(name);
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
        cx: Cx<'_>,
    ) -> Result<String, QueryError> {
        let cx = cx.inner();
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
                self.lower(filter, &inner, cx)
            }
            Via::Session(alias) => {
                let inner = RowContext {
                    unit: Unit::Sessions,
                    a: Aliases {
                        s: a.expand(alias),
                        ..a.clone()
                    },
                };
                self.lower(filter, &inner, cx)
            }
            Via::Exists { cond: template, .. } => {
                let n = self.fresh();
                let inner = RowContext {
                    unit: target,
                    a: n.clone(),
                };
                let cond = self.lower(filter, &inner, cx)?;
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
                        parts.push(self.session_reference(expr, "id", x, false));
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
                    parts.push(self.directory_range(expr, x));
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
            // Binding gives a special value only to a field with its own
            // rule, which reads it there.
            Test::Special(_) => unreachable!("{} has its own rule", def.name),
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

    /// A time literal in a computed value, resolved against the
    /// execution's clock and recorded. A computed value's operand types are
    /// fixed by the shared binder, so its literals take their stored form
    /// here, where its expression is lowered.
    pub(super) fn time(&self, text: &str, span: Span) -> Result<Time, QueryError> {
        let resolved =
            polylogue_query::time::resolve(text, &self.opts.clock).map_err(|mut error| {
                error.position = span.start;
                QueryError::language(error, self.text)
            })?;
        Ok(self.resolved(&resolved))
    }

    /// `path` and the directories below it, by whole components, as a key
    /// range over `expr` that an index on it serves: `path` itself, then
    /// every value that starts with `path/` (`0` is the character after `/`).
    pub(super) fn directory_range(&mut self, expr: &str, path: &str) -> String {
        crate::session_sql::path_under_sql(expr, path, |value| self.text_param(value))
    }
}

pub(super) use polylogue_query::filter::unknown_field;
