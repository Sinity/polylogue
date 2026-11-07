//! Typed adapters registered from the owning command declaration.
use crate::commands::{Action, Caller, Command, Surface, Values};
use crate::daemon::service::Daemon;
use command_surface::{
    Answer, Failure, FailureKind, JsonSchema, LatencyClass, Operation, OperationFuture,
    OperationSpec, OperationTable, RequestContext,
};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::{marker::PhantomData, sync::Arc};

/// An MCP tool call: the operation of the tool it runs (needed when the
/// tool has several), with the same arguments and admission fields as the
/// socket.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    pub arguments: Value,
    pub request_id: String,
    #[serde(default, skip_serializing_if = "default_deadline")]
    pub deadline: command_surface::Deadline,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "command_surface::TokenBudget::is_default"
    )]
    pub budget_tokens: command_surface::TokenBudget,
}

fn default_deadline(deadline: &command_surface::Deadline) -> bool {
    matches!(deadline, command_surface::Deadline::Default)
}

pub const PRODUCT: &str = "polylogue";

pub trait CommandType: Send + Sync + 'static {
    type Arguments: Serialize + DeserializeOwned + JsonSchema + Send + 'static;
    type Answer: Serialize + DeserializeOwned + JsonSchema + Send + 'static;
    const NAME: &'static str;
    fn required_access(_arguments: &Self::Arguments) -> command_surface::Access {
        declared_access(Self::NAME)
    }
}

pub fn declared_access(name: &str) -> command_surface::Access {
    if crate::commands::command(name)
        .expect("typed command exists")
        .writes
    {
        command_surface::Access::Write
    } else {
        command_surface::Access::Read
    }
}

/// Mixed read/write rows are classified from their admitted request, not row metadata.
pub(crate) fn read_scope_required(
    command: Option<&Command>,
    admitted: Option<&crate::daemon::Request>,
) -> bool {
    admitted.map_or_else(
        || command.is_some_and(|command| !command.writes),
        crate::daemon::Request::read_only,
    )
}

/// Pure writes retain their result instead of inheriting a read deadline.
/// Mixed operations use the admitted request, including bounded preparation.
pub(crate) fn admitted_deadline(
    command: Option<&Command>,
    admitted: Option<&crate::daemon::Request>,
    deadline: command_surface::Deadline,
) -> command_surface::Deadline {
    if admitted.map_or_else(
        || command.is_none_or(|command| !command.writes),
        crate::daemon::Request::has_read_phase,
    ) {
        deadline
    } else {
        command_surface::Deadline::None
    }
}

pub(crate) fn specification(command: &Command) -> OperationSpec {
    use command_surface::{ArgumentForm, ArgumentSpec, OperationKind};
    let tool = command.tool.expect("typed request declares its tool");
    OperationSpec {
        name: command.name.into(),
        command: command.cli_words(),
        summary: command.help.into(),
        reason: command.reason.into(),
        kind: OperationKind::Request {
            tool: tool
                .name()
                .strip_prefix("polylogue_")
                .expect("product tool prefix")
                .into(),
        },
        http_route: Some(format!("/api/v1{}", command.http_path())),
        latency: command.latency,
        failures: command.failures.to_vec(),
        access: declared_access(command.name),
        arguments: command
            .args
            .iter()
            .filter(|arg| !matches!(arg.name, "request_id" | "cursor" | "budget"))
            .map(|arg| ArgumentSpec {
                name: arg.name.into(),
                form: match arg.kind {
                    crate::commands::Kind::Bool => ArgumentForm::Flag,
                    crate::commands::Kind::Words => ArgumentForm::Words,
                    crate::commands::Kind::Rest => ArgumentForm::Rest,
                    crate::commands::Kind::List if arg.positional => ArgumentForm::Words,
                    crate::commands::Kind::List => ArgumentForm::Repeated,
                    _ if arg.positional => ArgumentForm::Positional,
                    _ => ArgumentForm::Named,
                },
                help: arg.help.into(),
                short: arg.short,
                calling_process: arg.cli_only,
                listed: !arg.unlisted,
            })
            .collect(),
        listed: true,
        default_budget_tokens: None,
    }
}

pub(crate) fn cursor(request: &crate::daemon::Request) -> Option<&str> {
    match request {
        crate::daemon::Request::Query { cursor, .. }
        | crate::daemon::Request::Read { cursor, .. }
        | crate::daemon::Request::Attachment { cursor, .. }
        | crate::daemon::Request::Explain { cursor, .. } => cursor.as_deref(),
        crate::daemon::Request::Context {
            ask: crate::briefing::Ask::Recall { cursor, .. },
            ..
        } => cursor.as_deref(),
        _ => None,
    }
}

pub(crate) fn with_cursor(
    mut request: crate::daemon::Request,
    cursor: Option<&str>,
) -> Result<crate::daemon::Request, Failure> {
    if let Some(cursor) = cursor {
        match &mut request {
            crate::daemon::Request::Context {
                ask: crate::briefing::Ask::Recall { cursor: target, .. },
                ..
            }
            | crate::daemon::Request::Query { cursor: target, .. }
            | crate::daemon::Request::Read { cursor: target, .. }
            | crate::daemon::Request::Attachment { cursor: target, .. }
            | crate::daemon::Request::Explain { cursor: target, .. } => {
                *target = Some(cursor.into())
            }
            _ => {
                return Err(Failure::new(
                    FailureKind::BadRequest,
                    "operation does not continue a page",
                ));
            }
        }
    }
    Ok(request)
}

pub(crate) fn with_budget(
    mut request: crate::daemon::Request,
    budget: command_surface::TokenBudget,
) -> Result<crate::daemon::Request, Failure> {
    if !budget.is_default() {
        match &mut request {
            crate::daemon::Request::Context { budget_tokens, .. } => *budget_tokens = budget,
            _ => {
                return Err(Failure::new(
                    FailureKind::BadRequest,
                    "operation does not prepare budgeted context",
                ));
            }
        }
    }
    Ok(request)
}

pub(crate) fn admit(
    command: &Command,
    envelope: &command_surface::RunnableRequest,
    caller: &Caller,
) -> Result<crate::daemon::Request, Failure> {
    let build = command
        .build_typed
        .ok_or_else(|| Failure::new(FailureKind::BadRequest, "operation has no typed builder"))?;
    with_budget(
        with_cursor(
            build(envelope.arguments.clone(), caller, &envelope.request_id)?,
            envelope.cursor.as_deref(),
        )?,
        envelope.budget_tokens,
    )
}

pub fn register<C: CommandType>(daemon: Arc<Daemon>, table: &mut OperationTable) {
    table
        .register(TypedCommand::<C> {
            daemon,
            marker: PhantomData,
        })
        .expect("declared command is unique");
}

struct TypedCommand<C> {
    daemon: Arc<Daemon>,
    marker: PhantomData<C>,
}

/// A dropped shared operation cancels its blocking reader before releasing ownership.
struct CancelRead(polylogue_store::read::limit::ReadCancellation);
impl Drop for CancelRead {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

impl<C: CommandType> Operation for TypedCommand<C> {
    type Request = C::Arguments;
    type Response = C::Answer;

    fn spec(&self) -> OperationSpec {
        specification(crate::commands::command(C::NAME).expect("typed command exists"))
    }

    fn required_access(&self, arguments: &Self::Request) -> command_surface::Access {
        C::required_access(arguments)
    }

    fn execute(
        &self,
        arguments: Self::Request,
        context: RequestContext,
    ) -> OperationFuture<'_, Answer<Self::Response>> {
        Box::pin(async move {
            let command = crate::commands::command(C::NAME).expect("typed command exists");
            let caller = Caller {
                surface: if context
                    .caller()
                    .identity::<crate::protocol::caller::Accepted>()
                    .is_some_and(|accepted| {
                        accepted.authorship.via == polylogue_model::authorship::Channel::Hook
                    }) {
                    Surface::Hook
                } else {
                    match context.via {
                        command_surface::Via::Web => Surface::Http,
                        command_surface::Via::Mcp => Surface::Mcp,
                        _ => Surface::Cli,
                    }
                },
                cwd: context
                    .caller()
                    .identity::<crate::protocol::caller::Accepted>()
                    .and_then(|caller| caller.working_directory.clone()),
                json: true,
            };
            let values = match checked_values::<C>(arguments, &caller, &context.request_id) {
                Ok(values) => values,
                Err(error) => return Answer::error(error),
            };
            let narrower = values
                .text("query")
                .and_then(|text| crate::daemon::execution::narrower(&text))
                .map(|text| {
                    let mut arguments = values.0.clone();
                    arguments.insert("query".into(), Value::String(text));
                    command_surface::Request {
                        operation: C::NAME.into(),
                        arguments: Value::Object(arguments),
                        request_id: context.request_id.clone(),
                        deadline: command_surface::Deadline::Default,
                        budget_tokens: Default::default(),
                        cursor: None,
                        fresh: false,
                        format: Default::default(),
                    }
                });
            if let Err(error) = context.progress_unknown(C::NAME, narrower) {
                return Answer::error(error);
            }
            let build = match command.action {
                Action::Request(build) | Action::Stream { build, .. } => build,
                Action::Local(_) | Action::ConfiguredLocal(_) => {
                    return Answer::error(Failure::new(
                        FailureKind::BadRequest,
                        "command runs locally",
                    ));
                }
            };
            let accepted = context
                .caller()
                .identity::<crate::protocol::caller::Accepted>();
            let admitted = accepted
                .filter(|accepted| {
                    accepted
                        .invocation
                        .as_ref()
                        .and_then(|invocation| invocation.accepted_request())
                        .is_some_and(|request| {
                            request.operation == C::NAME && request.request_id == context.request_id
                        })
                })
                .and_then(|accepted| accepted.admitted_request.clone());
            let request = match admitted {
                Some(request) => request,
                None => match build(command, &values, &caller) {
                    Ok(request) => request,
                    Err(error) => {
                        return Answer::error(Failure::new(FailureKind::BadRequest, error));
                    }
                },
            };
            let request = match with_cursor(request, context.cursor()) {
                Ok(request) => request,
                Err(error) => return Answer::error(error),
            };
            if let Err(error) = context.check_cancelled() {
                return Answer::error(error);
            }
            let cancellation = accepted
                .map(|caller| caller.cancellation.child())
                .unwrap_or_default();
            let authorship = accepted
                .map(|caller| caller.authorship.clone())
                .unwrap_or_else(|| {
                    let via = match context.via {
                        command_surface::Via::Web => polylogue_model::authorship::Channel::Web,
                        command_surface::Via::Mcp => polylogue_model::authorship::Channel::Mcp,
                        _ => polylogue_model::authorship::Channel::Cli,
                    };
                    polylogue_model::authorship::compute_authorship(via, &Default::default(), None)
                });
            let invocation = accepted.and_then(|caller| caller.invocation.clone());
            let parent = accepted.and_then(|caller| caller.parent.clone());
            let guard = CancelRead(cancellation.clone());
            let daemon = Arc::clone(&self.daemon);
            let worker = tokio::task::spawn_blocking(move || {
                let limit = match &parent {
                    Some(parent) => parent.read_limit(cancellation),
                    None => polylogue_store::read::limit::ReadLimit::without_deadline(cancellation),
                };
                let invocation = invocation
                    .unwrap_or_else(|| daemon.invocation(&authorship))
                    .with_parent(parent);
                crate::daemon::service::handle_with_invocation(
                    request,
                    &daemon,
                    &limit,
                    &invocation,
                )
            });
            let answer = match worker.await {
                Ok(answer) => answer,
                Err(error) => {
                    return Answer::error(Failure::new(FailureKind::Io, error.to_string()));
                }
            };
            drop(guard);
            typed(answer, C::NAME)
        })
    }

    fn render_text(&self, answer: &Self::Response) -> String {
        let command = crate::commands::command(C::NAME).expect("typed command exists");
        (command.print)(
            &serde_json::to_value(answer).expect("typed answer serializes"),
            false,
        )
    }
}

/// The daemon's answer decoded into the operation's declared answer type;
/// the outcome, notes and next actions pass through unchanged.
fn typed<T: DeserializeOwned>(answer: Answer<Value>, name: &str) -> Answer<T> {
    let mut problem = None;
    let decoded = answer.map(|value| {
        serde_json::from_value::<T>(value)
            .map_err(|error| problem = Some(error))
            .ok()
    });
    match problem {
        Some(error) => Answer::error(crate::protocol::protocol_mismatch(name, error)),
        None => decoded.map(|value| value.expect("decoded without a problem")),
    }
}

fn checked_values<C: CommandType>(
    arguments: C::Arguments,
    caller: &Caller,
    request_id: &str,
) -> Result<Values, Failure> {
    if request_id.is_empty() {
        return Err(Failure::new(
            FailureKind::BadRequest,
            "request_id must not be empty",
        ));
    }
    let command = crate::commands::command(C::NAME).expect("typed command exists");
    let values = serde_json::to_value(arguments)
        .expect("typed arguments serialize")
        .as_object()
        .expect("command arguments are objects")
        .clone();
    let mut values = Values::from_json(command, caller.surface, values)
        .map_err(|error| Failure::new(FailureKind::BadRequest, error))?;
    // Durable identity belongs to the accepted envelope, outside operation arguments.
    values
        .0
        .insert("request_id".into(), Value::String(request_id.into()));
    Ok(values)
}

/// Streams and finite answers validate through the same declaration adapter.
pub fn build<C: CommandType>(
    arguments: Value,
    caller: &Caller,
    request_id: &str,
) -> Result<crate::daemon::Request, Failure> {
    let arguments: C::Arguments = serde_json::from_value(arguments)
        .map_err(|error| Failure::new(FailureKind::BadRequest, error.to_string()))?;
    let values = checked_values::<C>(arguments, caller, request_id)?;
    let command = crate::commands::command(C::NAME).expect("typed command exists");
    match command.action {
        Action::Request(build) | Action::Stream { build, .. } => build(command, &values, caller)
            .map_err(|error| Failure::new(FailureKind::BadRequest, error)),
        Action::Local(_) | Action::ConfiguredLocal(_) => Err(Failure::new(
            FailureKind::BadRequest,
            "command runs locally",
        )),
    }
}

pub fn deadlines() -> command_surface::Deadlines {
    let mut deadlines = command_surface::Deadlines::default();
    for (class, milliseconds) in [
        (LatencyClass::Status, 80),
        (LatencyClass::Read, 250),
        (LatencyClass::Search, 2000),
        (LatencyClass::Aggregate, 400),
    ] {
        deadlines.set(class, Some(std::time::Duration::from_millis(milliseconds)));
    }
    deadlines
}

pub fn table(daemon: Arc<Daemon>) -> OperationTable {
    let mut table = OperationTable::new(PRODUCT, deadlines());
    table.accept(command_surface::RequestOption::Cursor);
    table.accept(command_surface::RequestOption::BudgetTokens);
    for (kind, codes) in super::FAILURE_CLASSES {
        for code in *codes {
            table
                .register_failure_code(*code, *kind)
                .expect("failure classification is unique");
        }
    }
    for tool in crate::commands::Tool::ALL {
        if crate::commands::COMMANDS
            .iter()
            .any(|command| command.register.is_some() && command.tool == Some(tool))
        {
            table
                .declare_tool(command_surface::ToolFamily {
                    name: tool
                        .name()
                        .strip_prefix("polylogue_")
                        .expect("product tool prefix")
                        .into(),
                    description: tool.description().into(),
                    open_world: tool.open_world(),
                    idempotent: tool.idempotent(),
                })
                .expect("declared tool is unique");
        }
    }
    for command in crate::commands::COMMANDS.iter() {
        if let Some(register) = command.register {
            register(Arc::clone(&daemon), &mut table);
        }
    }
    table
}

pub(crate) fn parse_budget(value: Option<&str>) -> Result<command_surface::TokenBudget, String> {
    match value {
        None => Ok(command_surface::TokenBudget::Default),
        Some("none") => Ok(command_surface::TokenBudget::Unlimited),
        Some(value) => value
            .parse::<u32>()
            .map(command_surface::TokenBudget::Tokens)
            .map_err(|_| "budget must be a token count or none".into()),
    }
}

pub fn request(
    command: &Command,
    arguments: Values,
    request_id: String,
    deadline: command_surface::Deadline,
) -> command_surface::RunnableRequest {
    let cursor = arguments.text("cursor");
    let budget_tokens =
        parse_budget(arguments.text("budget").as_deref()).expect("CLI token budget was validated");
    command_surface::Request {
        operation: command.name.into(),
        arguments: Value::Object(
            arguments
                .0
                .into_iter()
                .filter(|(name, _)| {
                    command.args.iter().any(|argument| {
                        argument.name == name && !argument.cli_only && name != "cursor"
                    })
                })
                .collect(),
        ),
        request_id,
        deadline,
        budget_tokens,
        cursor,
        fresh: false,
        format: Default::default(),
    }
}

#[cfg(test)]
mod tests;
