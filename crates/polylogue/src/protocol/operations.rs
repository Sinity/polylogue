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

pub trait CommandType: Send + Sync + 'static {
    type Arguments: Serialize + DeserializeOwned + JsonSchema + Send + 'static;
    type Answer: Serialize + DeserializeOwned + JsonSchema + Send + 'static;
    const NAME: &'static str;
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
        let command = crate::commands::command(C::NAME).expect("typed command exists");
        OperationSpec {
            name: C::NAME.into(),
            command: C::NAME.split(' ').map(str::to_owned).collect(),
            summary: command.help.into(),
            reason: command.reason.into(),
            http_route: format!("/api/v1{}", command.http_path()),
            latency: command.latency,
            failures: command.failures.to_vec(),
            write: command.writes,
        }
    }

    fn execute(
        &self,
        arguments: Self::Request,
        context: RequestContext,
    ) -> OperationFuture<'_, Answer<Self::Response>> {
        Box::pin(async move {
            let command = crate::commands::command(C::NAME).expect("typed command exists");
            let caller = Caller {
                surface: match context.origin {
                    command_surface::Origin::Http => Surface::Http,
                    command_surface::Origin::Mcp => Surface::Mcp,
                    _ => Surface::Cli,
                },
                cwd: None,
                json: true,
            };
            let values = serde_json::to_value(arguments).expect("typed arguments serialize");
            let mut values = values
                .as_object()
                .expect("command arguments are objects")
                .clone();
            if command.writes {
                values.insert(
                    "request_id".into(),
                    Value::String(context.request_id.clone()),
                );
            }
            let values = match Values::from_json(command, caller.surface, values) {
                Ok(values) => values,
                Err(error) => return Answer::error(Failure::new(FailureKind::BadRequest, error)),
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
                    }
                });
            if let Err(error) = context.progress(C::NAME, 0, narrower) {
                return Answer::error(error);
            }
            let Action::Request(build) = command.action else {
                return Answer::error(Failure::new(
                    FailureKind::BadRequest,
                    "command is not a finite daemon operation",
                ));
            };
            let request = match build(command, &values, &caller) {
                Ok(request) => request,
                Err(error) => return Answer::error(Failure::new(FailureKind::BadRequest, error)),
            };
            if let Err(error) = context.check_cancelled() {
                return Answer::error(error);
            }
            let accepted = context
                .caller()
                .identity::<crate::protocol::caller::Accepted>();
            let cancellation = accepted
                .map(|caller| caller.cancellation.clone())
                .unwrap_or_default();
            let authorship = accepted
                .map(|caller| caller.authorship.clone())
                .unwrap_or_else(|| {
                    let via = match context.origin {
                        command_surface::Origin::Http => polylogue_model::authorship::Channel::Web,
                        command_surface::Origin::Mcp => polylogue_model::authorship::Channel::Mcp,
                        _ => polylogue_model::authorship::Channel::Cli,
                    };
                    polylogue_model::authorship::compute_authorship(via, &Default::default(), None)
                });
            let guard = CancelRead(cancellation.clone());
            let daemon = Arc::clone(&self.daemon);
            let worker = tokio::task::spawn_blocking(move || {
                let limit = polylogue_store::read::limit::ReadLimit::without_deadline(cancellation);
                crate::daemon::service::handle_with_caller(request, &daemon, &limit, &authorship)
                    .answer()
            });
            let answer = match worker.await {
                Ok(answer) => answer,
                Err(error) => {
                    return Answer::error(Failure::new(FailureKind::Io, error.to_string()));
                }
            };
            drop(guard);
            match answer {
                Answer::Error { error, notes, next } => Answer::Error { error, notes, next },
                answer => {
                    let converted = answer.map(serde_json::from_value::<C::Answer>);
                    match converted {
                        Answer::Ok {
                            value: Ok(value),
                            notes,
                            next,
                        } => Answer::Ok { value, notes, next },
                        Answer::Empty {
                            value: Ok(value),
                            notes,
                            next,
                        } => Answer::Empty { value, notes, next },
                        Answer::Degraded {
                            value: Ok(value),
                            notes,
                            next,
                        } => Answer::Degraded { value, notes, next },
                        Answer::Ok {
                            value: Err(error), ..
                        }
                        | Answer::Empty {
                            value: Err(error), ..
                        }
                        | Answer::Degraded {
                            value: Err(error), ..
                        } => Answer::error(Failure::new(
                            FailureKind::Io,
                            format!("invalid {} answer: {error}", C::NAME),
                        )),
                        Answer::Error { .. } => unreachable!(),
                    }
                }
            }
        })
    }

    fn render_text(&self, answer: &Self::Response) -> String {
        let command = crate::commands::command(C::NAME).expect("typed command exists");
        (command.print)(&serde_json::to_value(answer).expect("typed answer serializes"))
    }
}

pub fn table(daemon: Arc<Daemon>) -> OperationTable {
    let mut deadlines = command_surface::Deadlines::default();
    for (class, milliseconds) in [
        (LatencyClass::Status, 80),
        (LatencyClass::Read, 250),
        (LatencyClass::Search, 2000),
        (LatencyClass::Aggregate, 400),
    ] {
        deadlines.set(class, Some(std::time::Duration::from_millis(milliseconds)));
    }
    let mut table = OperationTable::new("polylogue", deadlines);
    for (kind, codes) in super::FAILURE_CLASSES {
        for code in *codes {
            table
                .register_failure_code(*code, *kind)
                .expect("failure classification is unique");
        }
    }
    for command in crate::commands::COMMANDS.iter() {
        if let Some(register) = command.register {
            register(Arc::clone(&daemon), &mut table);
        }
    }
    table
}

pub fn request(
    command: &Command,
    arguments: Values,
    request_id: String,
    deadline: command_surface::Deadline,
) -> command_surface::RunnableRequest {
    command_surface::Request {
        operation: command.name.into(),
        arguments: Value::Object(arguments.0),
        request_id,
        deadline,
    }
}

#[cfg(test)]
mod tests;
