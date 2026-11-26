//! The handlers of the archive's own commands: reads of state, authored
//! writes, imports and the advisory. The verbs of a line are in
//! `daemon::verbs`, acts on live sessions in `acts`, the browser extension's
//! in `browser`.
use super::{
    Failure, advisory_silence, context, delivery_event, error, ok, query_targets_in,
    repeated_request, status, store_error, write, write_from_read,
};
use crate::brief::Ask;
use crate::cli::print;
use crate::commands::ToolAccess;
use crate::daemon::handler::{Bound, Call, Lane, Late, Operation, Policy};
use crate::protocol;
use command_surface::{Answer, JsonSchema, TextStyle};
use polylogue_model::authorship::Channel;
use polylogue_model::user::UserWrite;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::mpsc::sync_channel;

/// The answer when the engine loop no longer takes requests.
fn loop_stopped<T>() -> Answer<T> {
    error(polylogue_model::Code::Internal, "the loop has stopped")
}

pub(crate) struct Sql;
impl Operation for Sql {
    type Arguments = protocol::query::Sql;
    type Reply = protocol::query::SqlRows;
    fn policy(_: &Self::Arguments, _: Channel) -> Result<Policy, Failure> {
        Ok(Policy::READ)
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Reply>, Failure> {
        let reader = call.reader()?;
        Ok(
            // Compiled under a read-only authorizer: a statement that writes,
            // creates temporary objects, attaches databases or sets pragmas is
            // refused before it runs.
            match reader.query_untrusted(&arguments.statement) {
                Ok(rows) if rows.is_empty() => Answer::empty(rows),
                Ok(rows) => Answer::ok(rows),
                Err(e) => error(polylogue_model::Code::Sql, e),
            },
        )
    }
}

/// Grammar completions at a UTF-8 byte offset in the full query text.
pub(crate) struct Complete;
impl Operation for Complete {
    type Arguments = protocol::complete::Arguments;
    type Reply = protocol::complete::Answer;
    fn policy(_: &Self::Arguments, _: Channel) -> Result<Policy, Failure> {
        Ok(Policy {
            lane: Lane::Interactive,
            ..Policy::READ
        })
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Reply>, Failure> {
        let text = arguments.text;
        let position = match arguments.position.map(usize::try_from).transpose() {
            Ok(position) => position.unwrap_or(text.len()),
            Err(_) => {
                return Ok(error(
                    polylogue_model::Code::BadRequest,
                    "completion position cannot be represented on this host",
                ));
            }
        };
        if position > text.len() || !text.is_char_boundary(position) {
            return Ok(error(
                polylogue_model::Code::BadRequest,
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

/// Whether a status request reads vector counts or runs verification: slow
/// work that takes the background class and no default deadline.
fn status_is_deep(arguments: &protocol::status::Arguments) -> bool {
    arguments.verify || arguments.vectors
}

pub(crate) struct Status;
impl Operation for Status {
    type Arguments = protocol::status::Arguments;
    type Reply = protocol::status::Answer;
    /// Status reads the daemon's snapshot and its own state and takes the
    /// reserved reader if any; vector counts and verification are slow work
    /// at idle priority with no default deadline, and so is a wait.
    fn policy(arguments: &Self::Arguments, channel: Channel) -> Result<Policy, Failure> {
        if arguments.manifest.is_some() && !arguments.verify {
            return Err(Failure::bad_request("--manifest requires --verify"));
        }
        if arguments.line && status_is_deep(arguments) {
            return Err(Failure::bad_request(
                "--line cannot be combined with --verify or --vectors",
            ));
        }
        if arguments.wait.is_some() && (arguments.watch || arguments.verify) {
            return Err(Failure::bad_request(
                "--wait cannot be combined with --watch or --verify",
            ));
        }
        if arguments.after.is_some() && arguments.wait.is_none() {
            return Err(Failure::bad_request("--after needs --wait"));
        }
        if arguments.watch && (arguments.line || arguments.verify) {
            return Err(Failure::bad_request(
                "--watch cannot be combined with --line or --verify",
            ));
        }
        if arguments.watch && !crate::operation::command_line(channel) {
            return Err(Failure::bad_request(
                "--watch redraws a terminal and is available only on the command line",
            ));
        }
        if arguments.verify && !crate::operation::command_line(channel) {
            return Err(Failure::bad_request(
                "archive verification is available only on the command line",
            ));
        }
        Ok(if status_is_deep(arguments) {
            Policy {
                bound: Bound::CallerOnly,
                lane: Lane::Background,
                ..Policy::READ
            }
        } else if arguments.wait.is_some() {
            Policy {
                bound: Bound::CallerOnly,
                lane: Lane::Interactive,
                ..Policy::READ
            }
        } else {
            Policy {
                lane: Lane::Interactive,
                ..Policy::READ
            }
        })
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Reply>, Failure> {
        let manifest = arguments
            .manifest
            .as_deref()
            .map(|path| call.path(path, "manifest path"))
            .transpose()?;
        // Only vector counts take a reader, dropped before waiting for
        // idle-priority verification; everything else is the daemon's
        // snapshot and its own state.
        let vectors = status_is_deep(&arguments).then(|| {
            call.reader()
                .map_err(|failure| failure.error.message)
                .and_then(|reader| {
                    reader
                        .vectors()
                        .and_then(|vectors| vectors.status())
                        .map_err(|error| error.to_string())
                })
        });
        let mut status = match &arguments.wait {
            Some(wait) => {
                let wait = super::line::listing_wait(wait, call.caller.channel())?;
                status::wait(call, wait, arguments.after, || {
                    status::answer(call.daemon, vectors.clone(), arguments.log)
                })?
            }
            None => status::answer(call.daemon, vectors, arguments.log),
        };
        if arguments.verify
            && status
                .value()
                .is_some_and(|status| status.store.is_none() && status.index_build.is_some())
        {
            status
                .envelope
                .notes
                .push("archive verification awaits a readable index".into());
            return Ok(status);
        }
        if !arguments.verify || status.value().is_none() {
            return Ok(status);
        }
        Ok(status::with_verification(
            status,
            super::super::verify::run(call.daemon, manifest.as_deref()),
        ))
    }
}

/// Pages of the language, a verb's fields, or the full reference.
pub(crate) struct Help;
impl Operation for Help {
    type Arguments = protocol::help::Arguments;
    type Reply = Value;
    fn policy(_: &Self::Arguments, _: Channel) -> Result<Policy, Failure> {
        Ok(Policy {
            pages: true,
            ..Policy::READ
        })
    }
    const PRINT: fn(&Value, TextStyle) -> String = |value, style| {
        value.get("manual").and_then(Value::as_str).map_or_else(
            || print::json(value, style),
            polylogue_render::terminal::escape,
        )
    };
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Value>, Failure> {
        if arguments.to.is_some() || arguments.format.is_some() {
            let option = if arguments.to.is_some() {
                "to"
            } else {
                "format"
            };
            return Ok(error(
                polylogue_model::Code::NotBuilt,
                format!("help --{option} is not built yet"),
            ));
        }
        if arguments.all {
            return Ok(ok(serde_json::json!({"manual": crate::manual::text()})));
        }
        let topic = arguments.topic.as_deref().unwrap_or("");
        let subject = match topic {
            "units" | "fields" | "transforms" => topic,
            "query" => "syntax",
            _ => {
                let steps = crate::daemon::verbs::declarations();
                if let Some(verb) = steps.verbs().iter().find(|verb| verb.verb() == topic) {
                    return Ok(ok(serde_json::json!({
                        "verb": topic,
                        "signature": verb.signature(),
                        "arguments": verb.arguments(),
                        "fields": verb.input_schema()["properties"],
                    })));
                }
                if let Some(line) = crate::explain::removed_word(topic) {
                    return Ok(ok(serde_json::json!({ "manual": line })));
                }
                return Ok(error(
                    polylogue_model::Code::NotBuilt,
                    format!("help {topic} is not built yet"),
                ));
            }
        };
        crate::explain::answer(
            subject,
            arguments.unit.as_deref(),
            arguments.page_size,
            call.cursor,
        )
        .map(ok)
    }
}

/// Observes stored wire structure at idle priority.
pub(crate) struct SchemaInfer;
impl Operation for SchemaInfer {
    type Arguments = protocol::schema::Infer;
    type Reply = super::super::schema::Inference;
    fn policy(_: &Self::Arguments, _: Channel) -> Result<Policy, Failure> {
        Ok(Policy {
            bound: Bound::CallerOnly,
            lane: Lane::Background,
            ..Policy::READ
        })
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Reply>, Failure> {
        let protocol::schema::Infer {
            origin,
            all,
            input,
            since,
            revision,
            field,
        } = arguments;
        if origin.is_some() && all {
            return Ok(error(
                polylogue_model::Code::BadRequest,
                "choose an origin or --all",
            ));
        }
        let Ok(revision) = revision.map(i64::try_from).transpose() else {
            return Ok(error(
                polylogue_model::Code::BadRequest,
                "revision exceeds signed range",
            ));
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
impl Operation for SchemaUnread {
    type Arguments = protocol::schema::Unread;
    type Reply = polylogue_store::wire_shapes::Report;
    fn policy(_: &Self::Arguments, _: Channel) -> Result<Policy, Failure> {
        Ok(Policy::READ)
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Reply>, Failure> {
        if arguments.origin.is_some() && arguments.all {
            return Ok(error(
                polylogue_model::Code::BadRequest,
                "choose an origin or --all",
            ));
        }
        Ok(match call.reader() {
            Ok(reader) => super::super::schema::unread(&reader, arguments.origin.as_deref()),
            Err(failure) => failure.answer(),
        })
    }
}

pub(crate) struct Import;
impl Operation for Import {
    type Arguments = protocol::import::Arguments;
    type Reply = protocol::import::Answer;
    const PRINT: fn(&Self::Reply, TextStyle) -> String = crate::cli::print::import;
    /// Files are imported at idle priority; a provider's import job is
    /// started, shown or awaited like any other request.
    fn policy(arguments: &Self::Arguments, _: Channel) -> Result<Policy, Failure> {
        Ok(if arguments.provider().is_some() {
            Policy::write(ToolAccess::Act)
        } else {
            Policy {
                lane: Lane::Background,
                ..Policy::write(ToolAccess::Write)
            }
        })
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Reply>, Failure> {
        if let Some(named) = &arguments.provider {
            if !arguments.paths.is_empty() {
                return Err(Failure::bad_request(
                    "an import takes paths or a provider, not both",
                ));
            }
            if arguments.provider().is_none() {
                return Ok(error(
                    polylogue_model::Code::UnsupportedProvider,
                    format!("{named} is not a web chat the browser extension knows"),
                ));
            }
        }
        if let Some(provider) = arguments.provider() {
            return super::browser::import_provider(call, &arguments, provider)
                .map(|answer| answer.map(protocol::import::Answer::Provider));
        }
        if let Some((_, option)) = [
            (arguments.since.is_some(), "--since"),
            (arguments.cancel, "--cancel"),
            (arguments.wait.is_some(), "--wait"),
        ]
        .into_iter()
        .find(|(used, _)| *used)
        {
            return Err(Failure::bad_request(format!(
                "{option} is for a provider import (`polylogue import chatgpt {option}`), not for files"
            )));
        }
        if arguments.withdraw_missing {
            return Ok(error(
                polylogue_model::Code::NotBuilt,
                "import --withdraw-missing is not built yet",
            ));
        }
        let paths = arguments
            .paths
            .iter()
            .map(|path| call.path(path, "import path"))
            .collect::<Result<Vec<_>, _>>()?;
        let d = call.daemon;
        let invocation = call.invocation;
        let (reply, wait) = sync_channel(1);
        let Ok(tx) = d.loop_tx.lock() else {
            return Ok(loop_stopped());
        };
        if let Err(error) = invocation.check_parent_lifetime() {
            return Ok(store_error(&error));
        }
        let sent = tx
            .send(polylogue_engine::LoopMsg::Import {
                paths,
                format: arguments.origin.clone(),
                dry_run: arguments.dry_run,
                authorship: invocation.authorship.clone(),
                parent: invocation.parent_lifetime(),
                reply,
            })
            .is_ok();
        drop(tx);
        if !sent {
            return Ok(loop_stopped());
        }
        Ok(match wait.recv() {
            Ok(Ok(report)) => {
                super::import_answer(report).map(|r| protocol::import::Answer::Files(Box::new(r)))
            }
            Ok(Err(polylogue_engine::ImportFailure::Parent(error))) => store_error(&error),
            Ok(Err(polylogue_engine::ImportFailure::Import(
                polylogue_engine::EngineError::Store(error),
            ))) => store_error(&error),
            Ok(Err(polylogue_engine::ImportFailure::Import(
                polylogue_engine::EngineError::Stopped,
            ))) => store_error(&polylogue_store::StoreError::Stopped),
            Ok(Err(polylogue_engine::ImportFailure::Import(error))) => {
                super::error(polylogue_model::Code::Import, error)
            }
            Err(_) => loop_stopped(),
        })
    }
}

/// Submit the complete Save as one authored write and one retry receipt.
pub(crate) fn save_selection(
    step: protocol::steps::SaveStep,
    query: String,
    selection: &crate::daemon::verbs::Selection,
    call: &Call<'_>,
) -> Result<Answer<(protocol::line::SaveReport, Vec<polylogue_model::Reference>)>, Failure> {
    let (d, invocation) = (call.daemon, call.invocation);
    let request_id: String = call.request_id.into();
    let reader = call.reader()?;
    let selected_scope = reader.scope_token();
    let mut save = polylogue_model::user::Save {
        name: step.name,
        query,
        denominator: step.denominator,
        rows_json: String::new(),
        read_at: String::new(),
        outcomes: None,
        watch: step.watch.then_some(polylogue_model::user::SaveWatch {
            hold: step.hold,
            notify: step.notify,
        }),
    };
    let intended = UserWrite::Save(Box::new(save.clone()));
    let present = |receipt: polylogue_store::user_writes::WriteResult| {
        let mut report = receipt
            .save
            .expect("Save transaction returns its resolved outcomes");
        let written = receipt.written;
        report.pending = receipt.outcome.pending;
        report.repeat |= receipt.outcome.repeat;
        (report, written)
    };
    match repeated_request(&reader, &request_id, &intended, invocation) {
        Ok(Some(previous)) => {
            drop(reader);
            return Ok(super::answers::submit_result(
                d,
                invocation,
                request_id,
                previous,
                Some(selected_scope),
                present,
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
                text: &save.query,
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
            super::super::query::save_selected(
                &reader,
                ask,
                selection
                    .retained
                    .as_deref()
                    .ok_or_else(|| Failure::internal("Save ran without its accepted selection"))?,
                selection.resolved.as_deref(),
            )
        },
    );
    let kept = kept?;
    drop(reader);
    let read_at = serde_json::json!({ "now": kept["now"], "position":kept["read_at"] }).to_string();
    save.rows_json = kept.to_string();
    save.read_at = read_at;
    Ok(super::answers::submit_result(
        d,
        invocation,
        request_id,
        UserWrite::Save(Box::new(save)),
        Some(selected_scope),
        present,
    ))
}

/// A setting request with a value writes; without one it reads.
fn setting_writes(arguments: &protocol::setting::Arguments) -> bool {
    arguments.value.is_some()
}

/// Reads a setting or the catalog without a value; writes one with a value.
pub(crate) struct Setting;
impl Operation for Setting {
    type Arguments = protocol::setting::Arguments;
    type Reply = protocol::setting::Answer;
    fn policy(arguments: &Self::Arguments, _: Channel) -> Result<Policy, Failure> {
        Ok(if setting_writes(arguments) {
            Policy::write(ToolAccess::Write)
        } else {
            Policy::READ
        })
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Reply>, Failure> {
        let protocol::setting::Arguments {
            key,
            value,
            machine,
        } = arguments;
        let invocation = call.invocation;
        if let Some(value) = value {
            let Some(key) = key else {
                return Ok(error(
                    polylogue_model::Code::BadRequest,
                    "a value requires a setting key",
                ));
            };
            if let Some(hint) = polylogue_model::settings::historical_hint(&key) {
                return Ok(error(polylogue_model::Code::BadValue, hint));
            }
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
            .map(protocol::setting::Answer::Outcome));
        }
        let reader = call.reader()?;
        let selected = machine.as_deref().unwrap_or(&invocation.machine);
        if let Some(key) = key {
            if let Err(e) = polylogue_model::settings::target_machine(
                &key,
                machine.as_deref(),
                &invocation.machine,
            ) {
                return Ok(error(polylogue_model::Code::BadValue, e));
            }
            return Ok(match reader.setting_details(&key, selected) {
                Ok(current) => {
                    let mut answer =
                        Answer::ok(protocol::setting::Answer::Current(Box::new(current)));
                    if let Some(hint) = polylogue_model::settings::historical_hint(&key) {
                        answer.envelope.notes.push(hint.into());
                    }
                    answer
                }
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
                return Ok(error(polylogue_model::Code::BadValue, e));
            }
        }
        Ok(match reader.settings_catalog(selected) {
            Ok(catalog) => Answer::ok(protocol::setting::Answer::Catalog(catalog)),
            Err(e) => store_error(&e),
        })
    }
}

/// Previews embedding work, or builds it when the caller confirms.
pub(crate) struct Embed;
impl Operation for Embed {
    type Arguments = protocol::embed::Arguments;
    type Reply = crate::embeddings::Answer;
    /// The act tool runs it: a confirmed build spends on a paid provider.
    fn policy(arguments: &Self::Arguments, _: Channel) -> Result<Policy, Failure> {
        Ok(if arguments.yes {
            Policy::write(ToolAccess::Act)
        } else {
            Policy {
                access: ToolAccess::Act,
                ..Policy::READ
            }
        })
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Reply>, Failure> {
        let reader = call.reader()?;
        let context = call.context()?;
        let d = call.daemon;
        let settings = match reader.settings_snapshot(d.config.requests.machine()) {
            Ok(settings) => settings,
            Err(e) => return Ok(store_error(&e)),
        };
        Ok(crate::embeddings::handle(
            &reader,
            arguments.query.as_deref(),
            arguments.yes,
            crate::embeddings::Context {
                clock: &d.clock,
                settings: &settings,
                credentials: &d.config.embedding_credentials,
                operation: &context,
            },
        ))
    }
}

/// Prepared context reads under a deadline; for a caller with a sender
/// (MCP, a hook) it then records a delivery, so it changes the archive.
fn context_policy(channel: Channel) -> Policy {
    if crate::operation::sender(channel).is_some() {
        Policy {
            access: ToolAccess::Read,
            bound: Bound::UntilHandOver,
            ..Policy::write(ToolAccess::Read)
        }
    } else {
        Policy::READ
    }
}

/// The context rows: a request with a sender is recorded as a delivery.
pub(crate) fn prepare_context(
    call: &Call<'_>,
    ask: Ask,
) -> Result<Answer<crate::brief::BriefReply>, Failure> {
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

/// What `internal advisory` asks about: the prompt a session is about to
/// run.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct AdvisoryArguments {
    /// The prompt naming a file
    #[schemars(extend("argument_form" = "words"))]
    pub prompt: String,
    /// The directory (default: the caller's working directory)
    pub cwd: Option<std::path::PathBuf>,
    /// The receiving session
    pub session: polylogue_model::SessionId,
}

/// One operator-judged file pointer, silent when unavailable within its cap.
pub(crate) struct Advisory;
impl Operation for Advisory {
    type Arguments = AdvisoryArguments;
    type Reply = crate::brief::BriefReply;
    const PRINT: fn(&Self::Reply, TextStyle) -> String = print::brief;
    fn policy(_: &Self::Arguments, channel: Channel) -> Result<Policy, Failure> {
        Ok(Policy {
            late: Late::AnswerEmpty,
            ..context_policy(channel)
        })
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Reply>, Failure> {
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

/// What a hook reports about a delivery it was handed.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct DeliveryEventArguments {
    pub delivery: String,
    pub what: polylogue_model::DeliveryOutcome,
    pub detail: Option<String>,
}

/// What happened to a delivery after it was handed out; queued at once.
pub(crate) struct DeliveryEvent;
impl Operation for DeliveryEvent {
    type Arguments = DeliveryEventArguments;
    type Reply = Value;
    fn policy(_: &Self::Arguments, _: Channel) -> Result<Policy, Failure> {
        Ok(Policy::write(ToolAccess::Write))
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Reply>, Failure> {
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
impl Operation for SelectionNote {
    type Arguments = protocol::browser::SelectionNote;
    type Reply = polylogue_model::user::UserOutcome;
    fn policy(_: &Self::Arguments, _: Channel) -> Result<Policy, Failure> {
        Ok(Policy {
            bound: Bound::UntilHandOver,
            ..Policy::write(ToolAccess::Write)
        })
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Reply>, Failure> {
        let (d, invocation) = (call.daemon, call.invocation);
        let input = arguments;
        let (session, message_id) = input.references(call.request_id)?;
        let reader = call.reader()?;
        let message = polylogue_model::Reference::Message(message_id.clone());
        let intended = UserWrite::Assert {
            annotations: vec![input.annotation(session.clone(), message.clone())],
        };
        let selected_scope = reader.scope_token();
        match repeated_request(&reader, call.request_id, &intended, invocation) {
            Ok(Some(previous)) => {
                drop(reader);
                return Ok(write_from_read(
                    d,
                    invocation,
                    call.request_id.into(),
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
                polylogue_model::Code::Conflict,
                "the supplied message is missing, ambiguous or belongs to another conversation; capture the conversation and retry",
            ));
        }
        drop(reader);
        Ok(write_from_read(
            d,
            invocation,
            call.request_id.into(),
            intended,
            selected_scope,
        ))
    }
}

#[cfg(test)]
mod tests;
