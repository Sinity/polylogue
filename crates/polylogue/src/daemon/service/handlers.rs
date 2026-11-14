//! The handlers of the archive's own operations: queries and their
//! conveniences, reads of state, authored writes and prepared context. Acts
//! on live sessions live in `acts`, the browser extension's in `browser`,
//! and reads in `read`.
use super::{
    Failure, advisory_silence, context, delivery_event, error, listed, ok, query_targets,
    query_targets_in, repeated_request, status, store_error, write, write_from_read,
};
use crate::briefing::Ask;
use crate::cli::print;
use crate::commands::{Caller, Command, LIVE, NOTICES, WAITING};
use crate::daemon::handler::{Call, Handler};
use crate::protocol;
use command_surface::{Access, Answer};
use polylogue_model::user::UserWrite;
use serde_json::Value;
use std::sync::mpsc::sync_channel;

/// What a query convenience asks for.
struct QueryInput<'a> {
    text: &'a str,
    page_size: Option<u64>,
    why: bool,
    session: Option<String>,
    fields: &'a [String],
}

/// Runs one query with the ranking the archive's settings allow.
fn run_query(
    call: &Call<'_>,
    input: QueryInput<'_>,
) -> Result<Answer<protocol::query::Answer>, Failure> {
    let reader = call.reader()?;
    let context = call.context()?;
    let d = call.daemon;
    let freshness = d.freshness(&reader)?;
    Ok(super::super::query::with_ranking(
        &reader,
        &d.warm,
        &d.config,
        |embeddings, ranking_identity, near| {
            let ask = super::super::query::Ask {
                text: input.text,
                cursor: call.cursor,
                page_size: input.page_size,
                why: input.why,
                session: input.session,
                context,
                embeddings,
                ranking_identity,
                near: Some(near),
                fields: input.fields,
            };
            match super::super::query::prepare(&reader, ask).and_then(|prepared| {
                super::super::handler::finish_read_phase("optional query count")?;
                prepared.complete()
            }) {
                Ok(answer) => super::query_outcome(answer, &freshness),
                Err(failure) => failure.answer(),
            }
        },
    ))
}

fn run_rows(
    call: &Call<'_>,
    input: QueryInput<'_>,
) -> Result<Answer<protocol::query::Rows>, Failure> {
    let answer = run_query(call, input)?;
    let mut problem = None;
    let rows = answer.map(|value| {
        value
            .into_rows()
            .map_err(|error| problem = Some(error))
            .ok()
    });
    Ok(match problem {
        Some(failure) => failure.answer(),
        None => rows.map(|value| value.expect("row query returned rows")),
    })
}

pub(crate) struct Query;
impl Handler for Query {
    type Arguments = protocol::query::Arguments;
    type Answer = protocol::query::Answer;
    const PAGES: bool = true;
    fn query_text(arguments: &Self::Arguments) -> Option<&str> {
        Some(&arguments.query)
    }
    const PRINT: fn(&Value, bool) -> String = print::query;
    fn ranking_query(arguments: &Self::Arguments, _row: &Command) -> Option<String> {
        Some(arguments.query.clone())
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        run_query(
            call,
            QueryInput {
                text: &arguments.query,
                page_size: arguments.limit,
                why: arguments.why,
                session: arguments.session,
                fields: &arguments.fields,
            },
        )
    }
}

/// The query; without one, the `sessions` unit with its defaults.
pub(crate) struct List;
impl Handler for List {
    type Arguments = protocol::query_aliases::List;
    type Answer = protocol::query::Answer;
    const PAGES: bool = true;
    fn query_text(arguments: &Self::Arguments) -> Option<&str> {
        arguments.query.as_deref()
    }
    const PRINT: fn(&Value, bool) -> String = print::query;
    fn ranking_query(arguments: &Self::Arguments, _row: &Command) -> Option<String> {
        Some(
            arguments
                .query
                .as_deref()
                .filter(|query| !query.trim().is_empty())
                .unwrap_or("sessions")
                .to_owned(),
        )
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let text = arguments
            .query
            .as_deref()
            .filter(|query| !query.trim().is_empty())
            .unwrap_or("sessions");
        run_query(
            call,
            QueryInput {
                text,
                page_size: arguments.limit,
                why: arguments.why,
                session: None,
                fields: &arguments.fields,
            },
        )
    }
}

/// The live view: the sessions running now.
pub(crate) struct Live;
impl Handler for Live {
    type Arguments = protocol::query_aliases::Live;
    type Answer = protocol::query::Rows;
    const PAGES: bool = true;
    const PRINT: fn(&Value, bool) -> String = print::query;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        run_rows(
            call,
            QueryInput {
                text: LIVE,
                page_size: None,
                why: arguments.why,
                session: None,
                fields: &arguments.fields,
            },
        )
    }
}

/// The sessions waiting on the operator, the longest wait first.
pub(crate) struct Waiting;
impl Handler for Waiting {
    type Arguments = protocol::query_aliases::Live;
    type Answer = protocol::query::Rows;
    const PAGES: bool = true;
    const PRINT: fn(&Value, bool) -> String = print::query;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        run_rows(
            call,
            QueryInput {
                text: WAITING,
                page_size: None,
                why: arguments.why,
                session: None,
                fields: &arguments.fields,
            },
        )
    }
}

/// Signal notices, the latest first.
pub(crate) struct Notices;
impl Handler for Notices {
    type Arguments = protocol::query_aliases::Notices;
    type Answer = protocol::query::Rows;
    const PAGES: bool = true;
    const PRINT: fn(&Value, bool) -> String = print::query;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        run_rows(
            call,
            QueryInput {
                text: NOTICES,
                page_size: Some(arguments.limit.unwrap_or(50)),
                why: arguments.why,
                session: None,
                fields: &arguments.fields,
            },
        )
    }
}

/// `measure WHAT`: the query `QUERY | measure WHAT`, one row per measurement.
pub(crate) struct Measure;
impl Handler for Measure {
    type Arguments = protocol::query_aliases::Measure;
    type Answer = protocol::query::Answer;
    const PAGES: bool = true;
    fn query_text(arguments: &Self::Arguments) -> Option<&str> {
        arguments.query.as_deref()
    }
    const PRINT: fn(&Value, bool) -> String = print::query;
    fn ranking_query(arguments: &Self::Arguments, row: &Command) -> Option<String> {
        Some(measure_query(row.name, arguments))
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let text = measure_query(call.row.name, &arguments);
        run_query(
            call,
            QueryInput {
                text: &text,
                page_size: arguments.limit,
                why: arguments.why,
                session: None,
                fields: &arguments.fields,
            },
        )
    }
}

/// `measure NAME` with its options as the query `QUERY | measure NAME:OPTION`.
pub(crate) fn measure_query(row: &str, arguments: &protocol::query_aliases::Measure) -> String {
    let name = row.trim_start_matches("measure ");
    let what = match arguments.metric.as_deref().or(arguments.option.as_deref()) {
        Some(metric) => format!("{name}:{metric}"),
        None => name.to_owned(),
    };
    let what = polylogue_query::quoted(&what);
    match arguments.query.as_deref().filter(|q| !q.trim().is_empty()) {
        Some(query) => format!("{query} | measure {what}"),
        None => format!("sessions | measure {what}"),
    }
}

pub(crate) struct Sql;
impl Handler for Sql {
    type Arguments = protocol::query_aliases::Sql;
    type Answer = protocol::query_aliases::SqlRows;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let reader = call.reader()?;
        Ok(
            // Compiled under a read-only authorizer: a statement that writes,
            // creates temporary objects, attaches databases or sets pragmas is
            // refused before it runs.
            match reader.query_untrusted(&arguments.statement) {
                Ok(rows) if rows.is_empty() => Answer::empty(rows),
                Ok(rows) => Answer::ok(rows),
                Err(e) => error("sql", e),
            },
        )
    }
}

/// Grammar completions at a UTF-8 byte offset in the full query text.
pub(crate) struct Complete;
impl Handler for Complete {
    type Arguments = protocol::complete::Arguments;
    type Answer = protocol::complete::Answer;
    const CLASS: crate::daemon::executor::Class = crate::daemon::executor::Class::Interactive;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let text = arguments.text;
        let position = match arguments.position.map(usize::try_from).transpose() {
            Ok(position) => position.unwrap_or(text.len()),
            Err(_) => {
                return Ok(error(
                    "bad_request",
                    "completion position cannot be represented on this host",
                ));
            }
        };
        if position > text.len() || !text.is_char_boundary(position) {
            return Ok(error(
                "bad_request",
                "completion position must be a UTF-8 byte boundary within the query",
            ));
        }
        let d = call.daemon;
        let token = match d.archive.scope_token() {
            Ok(token) => token,
            Err(error) => return Ok(store_error(&error)),
        };
        let candidates = crate::completion_catalog::complete_file(
            &d.completion_path,
            &crate::runtime_dir::archive_key(&d.archive.root),
            &text,
            position,
        );
        if let Err(error) = token.check() {
            return Ok(store_error(&error));
        }
        Ok(if candidates.is_empty() {
            Answer::empty(candidates)
        } else {
            Answer::ok(candidates)
        })
    }
}

pub(crate) struct Status;
impl Handler for Status {
    type Arguments = protocol::status::Arguments;
    type Answer = protocol::status::Answer;
    const CLASS: crate::daemon::executor::Class = crate::daemon::executor::Class::Interactive;
    fn run(_: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        Ok(match call.reader() {
            Ok(reader) => status::answer(call.daemon, &reader),
            Err(failure) => failure.answer(),
        })
    }
}

/// Explains a query without running it, or pages through the language's
/// syntax, units, fields, operators and capabilities.
pub(crate) struct Explain;
impl Handler for Explain {
    type Arguments = protocol::explain::Arguments;
    type Answer = Value;
    const PAGES: bool = true;
    fn query_text(arguments: &Self::Arguments) -> Option<&str> {
        arguments.expression.as_deref()
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let protocol::explain::Arguments {
            subject,
            expression,
            unit,
            page_size,
            session,
        } = arguments;
        let result = if subject == "query" {
            if unit.is_some() || page_size.is_some() || call.cursor.is_some() {
                return Ok(error(
                    "bad_request",
                    "unit, page_size and cursor apply to declaration pages",
                ));
            }
            let Some(text) = expression.as_deref().filter(|s| !s.trim().is_empty()) else {
                return Ok(error("bad_request", "expression is required for query"));
            };
            let reader = call.reader()?;
            let context = call.context()?;
            let d = call.daemon;
            super::super::query::with_ranking(
                &reader,
                &d.warm,
                &d.config,
                |embeddings, ranking_identity, _near| {
                    super::super::query::explain(
                        &reader,
                        super::super::query::Ask {
                            text,
                            cursor: None,
                            page_size: None,
                            why: false,
                            session,
                            context,
                            embeddings,
                            ranking_identity,
                            near: None,
                            fields: &[],
                        },
                    )
                },
            )
        } else {
            crate::explain::answer(
                &subject,
                expression.as_deref(),
                unit.as_deref(),
                page_size,
                call.cursor,
            )
        };
        Ok(match result {
            Ok(data) => ok(data),
            Err(f) => f.answer(),
        })
    }
}

/// Chronological recorded activity and a logical-day calendar.
pub(crate) struct Timeline;
impl Handler for Timeline {
    type Arguments = protocol::calendar::Timeline;
    type Answer = Value;
    const PAGES: bool = true;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let reader = call.reader()?;
        let context = call.context()?;
        Ok(
            match super::super::calendar::timeline(
                &reader,
                super::super::calendar::Timeline {
                    date: arguments.date.as_deref().filter(|date| !date.is_empty()),
                    across_years: arguments.across_years,
                    undated: arguments.undated,
                    cursor: call.cursor,
                    page_size: arguments.limit,
                },
                &context,
            ) {
                Ok(data) => ok(data),
                Err(f) => f.answer(),
            },
        )
    }
}

/// Session-scoped usage, optionally bounded by a logical day, with
/// response pages.
pub(crate) struct Cost;
impl Handler for Cost {
    type Arguments = protocol::calendar::Cost;
    type Answer = Value;
    const PAGES: bool = true;
    fn query_text(arguments: &Self::Arguments) -> Option<&str> {
        arguments.query.as_deref()
    }
    fn ranking_query(arguments: &Self::Arguments, _row: &Command) -> Option<String> {
        Some(arguments.query.clone().unwrap_or_else(|| "sessions".into()))
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let reader = call.reader()?;
        let context = call.context()?;
        let d = call.daemon;
        let nonempty = |value: &Option<String>| value.clone().filter(|value| !value.is_empty());
        let (month, date, model) = (
            nonempty(&arguments.month),
            nonempty(&arguments.date),
            nonempty(&arguments.model),
        );
        let query = arguments.query.as_deref().unwrap_or("sessions");
        Ok(super::super::query::with_ranking(
            &reader,
            &d.warm,
            &d.config,
            |embeddings, ranking_identity, near| match super::super::calendar::cost(
                &reader,
                super::super::calendar::Cost {
                    embeddings,
                    ranking_identity,
                    near: Some(near),
                    month: month.as_deref(),
                    query,
                    date: date.as_deref(),
                    model: model.as_deref(),
                    undated: arguments.undated,
                    unknown_model: arguments.unknown_model,
                    cursor: call.cursor,
                    page_size: arguments.limit,
                },
                &context,
            ) {
                Ok(data) => ok(data),
                Err(f) => f.answer(),
            },
        ))
    }
}

/// Observes stored wire structure at idle priority.
pub(crate) struct SchemaInfer;
impl Handler for SchemaInfer {
    type Arguments = protocol::schema::Infer;
    type Answer = super::super::schema::Inference;
    const CLASS: crate::daemon::executor::Class = crate::daemon::executor::Class::Background;
    fn default_deadline(_: &Self::Arguments) -> bool {
        false
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let protocol::schema::Infer {
            origin,
            all,
            input,
            since,
            revision,
            field,
        } = arguments;
        if origin.is_some() && all {
            return Ok(error("bad_request", "choose an origin or --all"));
        }
        let Ok(revision) = revision.map(i64::try_from).transpose() else {
            return Ok(error("bad_request", "revision exceeds signed range"));
        };
        Ok(super::super::schema::infer(
            call.daemon,
            super::super::schema::Infer {
                origin,
                inputs: input.into_iter().collect(),
                revision,
                since,
                fields: field.into_iter().collect(),
            },
        ))
    }
}

/// Stored wire fields no parser reads.
pub(crate) struct SchemaUnread;
impl Handler for SchemaUnread {
    type Arguments = protocol::schema::Unread;
    type Answer = polylogue_store::wire_shapes::Report;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        if arguments.origin.is_some() && arguments.all {
            return Ok(error("bad_request", "choose an origin or --all"));
        }
        Ok(match call.reader() {
            Ok(reader) => super::super::schema::unread(&reader, arguments.origin.as_deref()),
            Err(failure) => failure.answer(),
        })
    }
}

/// The on-demand check of stored bytes, with an optional manifest file.
pub(crate) struct Verify;
impl Handler for Verify {
    type Arguments = protocol::verify::Arguments;
    type Answer = protocol::verify::Answer;
    const CLASS: crate::daemon::executor::Class = crate::daemon::executor::Class::Background;
    fn default_deadline(_: &Self::Arguments) -> bool {
        false
    }
    const PRINT: fn(&Value, bool) -> String = print::verify;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let manifest = arguments
            .manifest
            .as_deref()
            .map(|path| call.path(path, "manifest path"))
            .transpose()?;
        Ok(super::super::verify::run(call.daemon, manifest.as_deref()))
    }
}

/// One attachment's descriptor and a checked range of its bytes.
pub(crate) struct Attachment;
impl Handler for Attachment {
    type Arguments = protocol::attachment::Arguments;
    type Answer = protocol::attachment::Answer;
    const PAGES: bool = true;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let index = |value: u64, key: &str| {
            usize::try_from(value).map_err(|_| format!("{key} exceeds address space"))
        };
        let read = (|| {
            let message = polylogue_model::MessageId::parse(
                arguments
                    .message
                    .strip_prefix("message:")
                    .unwrap_or(&arguments.message),
            )
            .map_err(|e| e.to_string())?;
            Ok::<_, String>(super::super::attachment::Range {
                message,
                block: index(arguments.block, "block")?,
                attachment: index(arguments.attachment, "attachment")?,
                input: arguments.input,
                length: index(arguments.length, "length")?,
                cursor: call.cursor.map(str::to_owned),
            })
        })();
        Ok(match read {
            Ok(range) => super::super::attachment::read(range, call.daemon),
            Err(message) => error("bad_request", message),
        })
    }
}

pub(crate) struct Import;
impl Handler for Import {
    type Arguments = protocol::import::Arguments;
    type Answer = protocol::import::Answer;
    const CLASS: crate::daemon::executor::Class = crate::daemon::executor::Class::Background;
    fn read_phase(_: &Self::Arguments, _: &Command, _: &Caller) -> bool {
        false
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let paths = arguments
            .paths
            .iter()
            .map(|path| call.path(path, "import path"))
            .collect::<Result<Vec<_>, _>>()?;
        let from_archive = match arguments.from_archive.as_deref() {
            Some(_) if !paths.is_empty() => {
                return Ok(error(
                    "bad_request",
                    "--from-archive names one archive and takes no other paths",
                ));
            }
            Some(_) if arguments.format.is_some() => {
                return Ok(error("bad_request", "--from-archive takes no --format"));
            }
            Some(archive) => {
                let archive = call.path(archive, "archive path")?;
                Some(archive)
            }
            None if arguments.dry_run => {
                return Ok(error("bad_request", "--dry-run applies to --from-archive"));
            }
            None => None,
        };
        let d = call.daemon;
        let invocation = call.invocation;
        let (reply, wait) = sync_channel(1);
        let tx = match d.loop_tx.lock() {
            Ok(tx) => tx,
            Err(_) => return Ok(error("internal", "the loop has stopped")),
        };
        if let Err(error) = invocation.check_parent_lifetime() {
            return Ok(store_error(&error));
        }
        let sent = tx
            .send(polylogue_engine::LoopMsg::Import {
                paths,
                format: arguments.format.clone(),
                from_archive,
                dry_run: arguments.dry_run,
                authorship: invocation.authorship.clone(),
                parent: invocation.parent_lifetime(),
                reply,
            })
            .is_ok();
        drop(tx);
        if !sent {
            return Ok(error("internal", "the loop has stopped"));
        }
        Ok(match wait.recv() {
            Ok(Ok(report)) => super::import_answer(report),
            Ok(Err(polylogue_engine::ImportFailure::Parent(error))) => store_error(&error),
            Ok(Err(polylogue_engine::ImportFailure::Import(error))) => {
                super::error("import", error)
            }
            Err(_) => error("internal", "the loop has stopped"),
        })
    }
}

pub(crate) struct Assert;
impl Handler for Assert {
    type Arguments = protocol::assertion::Assert;
    type Answer = polylogue_model::user::UserOutcome;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        if arguments.assertions.is_empty() {
            return Ok(error("bad_request", "assertions must not be empty"));
        }
        Ok(write(
            call.daemon,
            call.invocation,
            call.request_id.into(),
            UserWrite::Assert {
                assertions: arguments.assertions,
            },
        ))
    }
}

pub(crate) struct AssertQuery;
impl Handler for AssertQuery {
    type Arguments = protocol::assertion::AssertQuery;
    type Answer = polylogue_model::user::AssertQueryOutcome;
    const READS_FIRST: bool = true;
    fn query_text(arguments: &Self::Arguments) -> Option<&str> {
        Some(&arguments.query)
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let protocol::assertion::AssertQuery { query, assertion } = arguments;
        if query.trim().is_empty() {
            return Ok(error("bad_request", "query must not be empty"));
        }
        let (d, invocation) = (call.daemon, call.invocation);
        let reader = call.reader()?;
        let selected_scope = reader.scope_token();
        let intended = UserWrite::AssertQuery {
            query: query.clone(),
            assertion: assertion.clone(),
            resolved: Vec::new(),
        };
        let resolved = match repeated_request(&reader, call.request_id, &intended, invocation) {
            Ok(Some(previous)) => previous,
            Ok(None) => {
                if assertion
                    .links
                    .iter()
                    .any(|link| link.role == polylogue_model::assertion::LinkRole::About)
                {
                    return Ok(error(
                        "bad_request",
                        "a query assertion cannot supply its own about links",
                    ));
                }
                let targets = query_targets(d, &reader, &query, invocation)?;
                UserWrite::AssertQuery {
                    query,
                    assertion,
                    resolved: targets,
                }
            }
            Err(failure) => return Err(failure),
        };
        let targets = match &resolved {
            UserWrite::AssertQuery { resolved, .. } => resolved.len(),
            _ => return Ok(error("conflict", "stored request has another action")),
        };
        drop(reader);
        Ok(super::submit(
            d,
            invocation,
            call.request_id.into(),
            resolved,
            Some(selected_scope),
            |outcome| polylogue_model::user::AssertQueryOutcome::new(outcome, targets),
        ))
    }
}

pub(crate) struct Revise;
impl Handler for Revise {
    type Arguments = protocol::assertion::Revise;
    type Answer = polylogue_model::user::UserOutcome;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        if arguments.bases.is_empty() {
            return Ok(error("bad_request", "bases must not be empty"));
        }
        Ok(write(
            call.daemon,
            call.invocation,
            call.request_id.into(),
            UserWrite::ReviseAssertions {
                bases: arguments.bases,
                patch: arguments.patch,
            },
        ))
    }
}

pub(crate) struct Withdraw;
impl Handler for Withdraw {
    type Arguments = protocol::assertion::Withdraw;
    type Answer = polylogue_model::user::UserOutcome;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        if arguments.targets.is_empty() {
            return Ok(error("bad_request", "targets must not be empty"));
        }
        Ok(write(
            call.daemon,
            call.invocation,
            call.request_id.into(),
            UserWrite::WithdrawAssertions {
                targets: arguments.targets,
                reason: arguments.reason,
            },
        ))
    }
}

/// The operator accepts or rejects one revision of an agent's note.
pub(crate) struct Judge;
impl Handler for Judge {
    type Arguments = protocol::assertion::Judge;
    type Answer = polylogue_model::user::UserOutcome;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let protocol::assertion::Judge {
            note,
            accepted,
            reason,
        } = arguments;
        let reference = match polylogue_model::Reference::parse(&note) {
            Ok(polylogue_model::Reference::Assertion(reference)) => reference,
            Ok(_) => {
                return Ok(error(
                    "bad_request",
                    "a judgment needs a natural assertion reference",
                ));
            }
            Err(e) => return Ok(error("bad_request", e)),
        };
        let reader = call.reader()?;
        let selected_scope = reader.scope_token();
        let found = match reader.assertion(&reference) {
            Ok(Some(found)) => found,
            Ok(None) => return Ok(error("not_found", format!("no assertion {note}"))),
            Err(e) => return Ok(store_error(&e)),
        };
        if !found.fact.author.is_agent() {
            return Ok(error(
                "bad_request",
                "only assertions agents wrote are accepted or rejected; revise or withdraw the operator's own",
            ));
        }
        drop(reader);
        let judgment = polylogue_model::assertion::NewAssertion {
            kind: polylogue_model::assertion::AssertionKind::Judgment,
            key: Some(if accepted { "accepted" } else { "rejected" }.into()),
            value: reason.map(Value::String),
            links: vec![polylogue_model::assertion::NewAssertionLink {
                role: polylogue_model::assertion::LinkRole::About,
                reference: polylogue_model::Reference::Assertion(
                    polylogue_model::RevisionReference {
                        root: found.fact.root,
                        revision: Some(found.fact.id),
                    },
                ),
            }],
            valid_until: None,
        };
        Ok(write_from_read(
            call.daemon,
            call.invocation,
            call.request_id.into(),
            UserWrite::Assert {
                assertions: vec![judgment],
            },
            selected_scope,
        ))
    }
}

/// Saves a query under a name, to run as `query:NAME`; saving a name again
/// revises its definition.
pub(crate) struct Save;
impl Handler for Save {
    type Arguments = protocol::saved::Save;
    type Answer = polylogue_model::user::UserOutcome;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let reader = call.reader()?;
        let intended = UserWrite::SaveQuery {
            name: arguments.name,
            text: arguments.query,
            denominator: arguments.denominator,
        };
        let resolved = match repeated_request(&reader, call.request_id, &intended, call.invocation)
        {
            Ok(Some(previous)) => previous,
            Ok(None) => intended,
            Err(failure) => return Err(failure),
        };
        drop(reader);
        Ok(write(
            call.daemon,
            call.invocation,
            call.request_id.into(),
            resolved,
        ))
    }
}

/// Watches a query: each row that starts matching becomes a notice (a
/// signal rule delivering notices).
pub(crate) struct Watch;
impl Handler for Watch {
    type Arguments = protocol::saved::Watch;
    type Answer = polylogue_model::user::UserOutcome;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let protocol::saved::Watch {
            name,
            hold,
            notify,
            query,
        } = arguments;
        let reader = call.reader()?;
        let context = call.context()?;
        let selected_scope = reader.scope_token();
        if let Some(hold) = &hold
            && crate::signals::duration(hold).is_none()
        {
            return Ok(error(
                "invalid",
                format!("--for {hold:?} is not a duration such as 20m"),
            ));
        }
        // A watch whose query cannot run would only be refused later.
        let ask = super::super::query::Ask {
            text: &query,
            cursor: None,
            page_size: Some(1),
            why: false,
            session: None,
            context,
            embeddings: false,
            ranking_identity: None,
            near: None,
            fields: &[],
        };
        super::super::query::run(&reader, ask)?;
        drop(reader);
        let mut rule = serde_json::json!({
            "name": name.clone().unwrap_or_else(|| query.clone()),
            "query": query,
            "fires": "on_new",
            "deliver": "notice",
            "interrupt": notify,
        });
        if let Some(hold) = hold {
            rule["for"] = serde_json::json!(hold);
        }
        let assertion = polylogue_model::assertion::NewAssertion {
            kind: polylogue_model::assertion::AssertionKind::Signal,
            key: rule["name"].as_str().map(str::to_owned),
            value: Some(rule),
            links: vec![],
            valid_until: None,
        };
        Ok(write_from_read(
            call.daemon,
            call.invocation,
            call.request_id.into(),
            UserWrite::Assert {
                assertions: vec![assertion],
            },
            selected_scope,
        ))
    }
}

/// Runs a query and keeps the rows it returned, as `result:<id>`.
pub(crate) struct Keep;
impl Handler for Keep {
    type Arguments = protocol::saved::Keep;
    type Answer = polylogue_model::user::UserOutcome;
    fn query_text(arguments: &Self::Arguments) -> Option<&str> {
        Some(&arguments.query)
    }
    fn ranking_query(arguments: &Self::Arguments, _row: &Command) -> Option<String> {
        Some(arguments.query.clone())
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let protocol::saved::Keep { name, query } = arguments;
        let (d, invocation) = (call.daemon, call.invocation);
        let request_id: String = call.request_id.into();
        let reader = call.reader()?;
        let selected_scope = reader.scope_token();
        let intended = UserWrite::KeepResult {
            name: name.clone(),
            query: query.clone(),
            rows_json: String::new(),
            read_at: String::new(),
        };
        match repeated_request(&reader, &request_id, &intended, invocation) {
            Ok(Some(previous)) => {
                drop(reader);
                return Ok(write_from_read(
                    d,
                    invocation,
                    request_id,
                    previous,
                    selected_scope,
                ));
            }
            Err(failure) => return Err(failure),
            Ok(None) => {}
        }
        let context = call.context()?;
        let kept = super::super::query::with_ranking(
            &reader,
            &d.warm,
            &d.config,
            |embeddings, ranking_identity, near| {
                let ask = super::super::query::Ask {
                    text: &query,
                    cursor: None,
                    page_size: None,
                    why: false,
                    session: None,
                    context,
                    embeddings,
                    ranking_identity,
                    near: Some(near),
                    fields: &[],
                };
                super::super::query::keep(&reader, ask)
            },
        );
        let kept = kept?;
        drop(reader);
        let read_at = serde_json::json!({ "now": kept["now"] }).to_string();
        Ok(write_from_read(
            d,
            invocation,
            request_id,
            UserWrite::KeepResult {
                name,
                query,
                rows_json: kept.to_string(),
                read_at,
            },
            selected_scope,
        ))
    }
}

/// Reads a setting or the catalog without a value; writes one with a value.
pub(crate) struct Setting;
impl Handler for Setting {
    type Arguments = protocol::setting::Arguments;
    type Answer = protocol::setting::Answer;
    fn read_only(arguments: &Self::Arguments, _: &Command, _: &Caller) -> bool {
        arguments.value.is_none()
    }
    fn access(arguments: &Self::Arguments, _: Access) -> Access {
        if arguments.value.is_some() {
            Access::Write
        } else {
            Access::Read
        }
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let protocol::setting::Arguments {
            key,
            value,
            machine,
        } = arguments;
        let invocation = call.invocation;
        if let Some(value) = value {
            let Some(key) = key else {
                return Ok(error("bad_request", "a value requires a setting key"));
            };
            call.reader()?;
            return Ok(write(
                call.daemon,
                invocation,
                call.request_id.into(),
                UserWrite::SetSetting {
                    key,
                    value,
                    machine,
                },
            )
            .map(protocol::setting::Answer::Receipt));
        }
        let reader = call.reader()?;
        let selected = machine.as_deref().unwrap_or(&invocation.machine);
        if let Some(key) = key {
            if let Err(e) = polylogue_model::settings::target_machine(
                &key,
                machine.as_deref(),
                &invocation.machine,
            ) {
                return Ok(error("bad_value", e));
            }
            return Ok(match reader.setting_current(&key, selected) {
                Ok(current) => Answer::ok(protocol::setting::Answer::Current(Box::new(current))),
                Err(e) => store_error(&e),
            });
        }
        for spec in polylogue_model::settings::CATALOG
            .iter()
            .filter(|spec| spec.scope == polylogue_model::settings::Scope::Host)
        {
            if let Err(e) = polylogue_model::settings::target_machine(
                spec.key,
                machine.as_deref(),
                &invocation.machine,
            ) {
                return Ok(error("bad_value", e));
            }
        }
        Ok(match reader.settings_catalog(selected) {
            Ok(catalog) => Answer::ok(protocol::setting::Answer::Catalog(catalog)),
            Err(e) => store_error(&e),
        })
    }
}

/// With a session, its resume commands; without, the sessions in a
/// directory worth resuming.
pub(crate) struct Continue;
impl Handler for Continue {
    type Arguments = protocol::resume::Arguments;
    type Answer = Value;
    const PRINT: fn(&Value, bool) -> String = print::resume;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let reader = call.reader()?;
        if let Some(session) = arguments.session {
            return Ok(match crate::resume::plan(&reader, &session) {
                Ok(plan) => ok(plan),
                Err(e) => store_error(&e),
            });
        }
        let cwd = call.directory(arguments.cwd.as_deref())?;
        let context = call.context()?;
        Ok(
            match crate::resume::resumable(
                &reader,
                &call.daemon.scope,
                &cwd,
                arguments.limit.unwrap_or(10),
                &context,
            ) {
                Ok(sessions) => listed(sessions),
                Err(e) => store_error(&e),
            },
        )
    }
}

/// Embeddings: `preview` (what a build would embed and cost) or `build`,
/// named by the row.
pub(crate) struct Embed;
impl Handler for Embed {
    type Arguments = protocol::embed::Arguments;
    type Answer = Value;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let what = call.row.name.trim_start_matches("embed ");
        embed(
            call,
            what,
            Some(arguments.query.unwrap_or_default()),
            arguments.limit,
            arguments.yes,
        )
    }
}

/// The stored embedding vectors and what a build would add.
pub(crate) struct EmbedStatus;
impl Handler for EmbedStatus {
    type Arguments = protocol::embed::Status;
    type Answer = Value;
    fn run(_: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        embed(call, "status", None, None, false)
    }
}

fn embed(
    call: &Call<'_>,
    what: &str,
    query: Option<String>,
    limit: Option<u64>,
    yes: bool,
) -> Result<Answer<Value>, Failure> {
    let reader = call.reader()?;
    let context = call.context()?;
    let d = call.daemon;
    let settings = match reader.settings_snapshot(d.config.requests.machine()) {
        Ok(settings) => settings,
        Err(e) => return Ok(store_error(&e)),
    };
    Ok(crate::embeddings::handle(
        &reader,
        what,
        query.as_deref(),
        limit,
        yes,
        crate::embeddings::Context {
            settings: &settings,
            credentials: &d.config.embedding_credentials,
            operation: &context,
        },
    ))
}

/// The context rows: a request with a sender is recorded as a delivery.
fn prepare_context(
    call: &Call<'_>,
    ask: Ask,
) -> Result<Answer<protocol::context::Answer>, Failure> {
    let advisory = matches!(ask, Ask::Advisory { .. });
    match call.reader() {
        Ok(reader) => drop(reader),
        Err(failure) if failure.code() == "shutting_down" || !advisory => {
            return Err(failure);
        }
        Err(_) => return Ok(advisory_silence()),
    }
    let context_handle = call.context()?;
    Ok(context(
        call.daemon,
        call.request_id.into(),
        &ask,
        call.sender(),
        call.budget,
        &context_handle,
    ))
}

pub(crate) struct Brief;
impl Handler for Brief {
    type Arguments = protocol::context::Brief;
    type Answer = protocol::context::Answer;
    const CLASS: crate::daemon::executor::Class = crate::daemon::executor::Class::Interactive;
    const BUDGETED: bool = true;
    const RECORDS_DELIVERY: bool = true;
    const READS_FIRST: bool = true;
    const PRINT: fn(&Value, bool) -> String = print::context;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let cwd = call.directory(arguments.cwd.as_deref())?;
        prepare_context(
            call,
            Ask::SessionStart {
                cwd,
                session: arguments.session,
                source: arguments.source,
            },
        )
    }
}

/// One operator-judged file pointer, silent when unavailable within its cap.
pub(crate) struct Advisory;
impl Handler for Advisory {
    type Arguments = protocol::context::Advisory;
    type Answer = protocol::context::Answer;
    const FAILS_OPEN: bool = true;
    const RECORDS_DELIVERY: bool = true;
    const READS_FIRST: bool = true;
    fn read_cap() -> Option<std::time::Duration> {
        Some(crate::advisory::BUDGET)
    }
    const PRINT: fn(&Value, bool) -> String = print::context;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let cwd = call.directory(arguments.cwd.as_deref())?;
        prepare_context(
            call,
            Ask::Advisory {
                prompt: arguments.prompt,
                cwd,
                session: arguments.session,
            },
        )
    }
}

pub(crate) struct Takeover;
impl Handler for Takeover {
    type Arguments = protocol::context::Takeover;
    type Answer = protocol::context::Answer;
    const BUDGETED: bool = true;
    const RECORDS_DELIVERY: bool = true;
    const READS_FIRST: bool = true;
    const PRINT: fn(&Value, bool) -> String = print::context;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        prepare_context(
            call,
            Ask::Takeover {
                session: arguments.session,
                for_session: arguments.for_session,
            },
        )
    }
}

pub(crate) struct Recall;
impl Handler for Recall {
    type Arguments = protocol::context::Recall;
    type Answer = protocol::context::Answer;
    const PAGES: bool = true;
    const BUDGETED: bool = true;
    const RECORDS_DELIVERY: bool = true;
    const READS_FIRST: bool = true;
    fn query_text(arguments: &Self::Arguments) -> Option<&str> {
        Some(&arguments.query)
    }
    const PRINT: fn(&Value, bool) -> String = print::context;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        prepare_context(
            call,
            Ask::Recall {
                text: arguments.query,
                cwd: call.directory(arguments.cwd.as_deref()).ok(),
                for_session: arguments.for_session,
                cursor: call.cursor.map(str::to_owned),
            },
        )
    }
}

/// What happened to a delivery after it was handed out; queued at once.
pub(crate) struct DeliveryEvent;
impl Handler for DeliveryEvent {
    type Arguments = protocol::act::DeliveryEvent;
    type Answer = Value;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        Ok(delivery_event(
            call.daemon,
            call.invocation,
            call.request_id.into(),
            UserWrite::DeliveryEvent {
                delivery: arguments.delivery,
                what: arguments.what,
                detail: arguments.detail,
            },
        ))
    }
}

/// A browser selection saved as the operator's note on its message.
pub(crate) struct SelectionNote;
impl Handler for SelectionNote {
    type Arguments = protocol::browser::SelectionNote;
    type Answer = polylogue_model::user::UserOutcome;
    const READS_FIRST: bool = true;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
        let (d, invocation) = (call.daemon, call.invocation);
        let input = crate::browser::notes::SelectionNote {
            request_id: call.request_id.into(),
            body: arguments.body,
            quote: arguments.quote,
            source: crate::browser::notes::Source {
                provider: arguments.provider,
                conversation_id: arguments.conversation_id,
                message_id: arguments.message_id,
                page_url: arguments.page_url,
                selected_at: arguments.selected_at,
            },
        };
        let (session, message_id) = input.references()?;
        let reader = call.reader()?;
        let message = polylogue_model::Reference::Message(message_id.clone());
        let intended = UserWrite::Assert {
            assertions: vec![input.assertion(session.clone(), message.clone())],
        };
        let selected_scope = reader.scope_token();
        match repeated_request(&reader, &input.request_id, &intended, invocation) {
            Ok(Some(previous)) => {
                drop(reader);
                return Ok(write_from_read(
                    d,
                    invocation,
                    input.request_id,
                    previous,
                    selected_scope,
                ));
            }
            Ok(None) => {}
            Err(failure) => return Err(failure),
        }
        // A message reference names exactly one message whatever its
        // reachability; `id:` takes the native ID, not the reference text.
        let query = format!(
            "messages where message:{}",
            polylogue_query::quoted(&message_id.to_string())
        );
        let targets = query_targets_in(d, &reader, &query, invocation, Some(&session))?;
        if targets.as_slice() != [message] {
            return Ok(error(
                "conflict",
                "the supplied message is missing, ambiguous or belongs to another conversation; capture the conversation and retry",
            ));
        }
        drop(reader);
        Ok(write_from_read(
            d,
            invocation,
            input.request_id,
            intended,
            selected_scope,
        ))
    }
}
