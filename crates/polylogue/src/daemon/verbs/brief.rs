//! `brief`: packaged context, chosen by the selection's form. No selection
//! is the directory brief, one session is its takeover, three or more named
//! sessions are an index of them, search words are a recall.
use super::{Given, RowRule, Selection, StepCall, StepDone, Verb, delegated, not_built_use};
use crate::daemon::Failure;
use crate::daemon::handler::{Call, Operation};
use crate::daemon::service::handlers;
use crate::protocol::context;
use crate::protocol::line::StepAnswer;
use crate::protocol::steps::BriefStep;
use command_surface::TokenBudget;
use polylogue_model::{Reference, SessionId};

#[cfg(test)]
mod tests;

pub(crate) struct Brief;
impl Verb for Brief {
    type Step = BriefStep;
    const NAME: &'static str = "brief";
    const ROWS: RowRule = RowRule::Any;
    fn run(
        step: BriefStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let for_session = step
            .for_session
            .as_deref()
            .map(|text| SessionId::parse(text).map_err(Failure::bad_value))
            .transpose()?;
        let budget =
            match step.budget {
                Some(tokens) => TokenBudget::Tokens(u32::try_from(tokens).map_err(|_| {
                    Failure::bad_value(format!("brief --budget exceeds {}", u32::MAX))
                })?),
                None => call.call.budget,
            };
        let call = Call {
            cursor: step.cursor.as_deref(),
            budget,
            ..call.call.copy()
        };
        let form = match form(selection) {
            Form::SessionWord(text) => Form::Session(one_session(&call, &text)?),
            other => other,
        };
        let answer = match form {
            Form::Directory => handlers::Brief::run(
                context::Brief {
                    cwd: step.cwd,
                    session: for_session,
                    source: step.source,
                },
                &call,
            )?,
            Form::Session(session) => handlers::Takeover::run(
                context::Takeover {
                    session,
                    for_session,
                },
                &call,
            )?,
            Form::Sessions(sessions) => handlers::prepare_context(
                &call,
                crate::brief::Ask::Sessions {
                    sessions,
                    for_session,
                },
            )?,
            Form::SessionWord(_) => unreachable!("a session word is resolved to a session above"),
            Form::Words(query) => handlers::Recall::run(
                context::Recall {
                    query,
                    cwd: step.cwd,
                    for_session,
                },
                &call,
            )?,
            Form::Other => return Err(not_built_use("brief", "over these rows")),
        };
        delegated(answer, |prepared| StepAnswer::Brief(Box::new(prepared)))
    }
    fn would(_: &BriefStep, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        Ok(match form(selection) {
            Form::Directory => "would brief this directory".into(),
            Form::Session(session) => format!("would brief a takeover of {session}"),
            Form::SessionWord(text) => format!("would brief a takeover of {text}"),
            Form::Sessions(sessions) => {
                format!("would brief an index of {} sessions", sessions.len())
            }
            Form::Words(query) => format!("would recall what is known about {query:?}"),
            Form::Other => return Err(not_built_use("brief", "over these rows")),
        })
    }
}

/// Which brief the selection asks for.
enum Form {
    Directory,
    Session(SessionId),
    /// `latest`, `self` or a bare ID: one session the query names.
    SessionWord(String),
    /// Three or more sessions named together.
    Sessions(Vec<SessionId>),
    /// Only words and quoted phrases.
    Words(String),
    Other,
}

fn form(selection: &Selection) -> Form {
    match &selection.given {
        Given::None => Form::Directory,
        Given::Reference(Reference::Session(session)) => Form::Session(session.clone()),
        Given::Query(text) => {
            let text = text.trim();
            if matches!(text, "latest" | "self" | "last")
                || polylogue_query::defaults::id_shaped(text)
            {
                Form::SessionWord(text.to_owned())
            } else if search_words(text) {
                Form::Words(text.to_owned())
            } else {
                Form::Other
            }
        }
        Given::References(references) if references.len() >= 3 => references
            .iter()
            .map(|reference| match reference {
                Reference::Session(session) => Some(session.clone()),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .map_or(Form::Other, Form::Sessions),
        Given::Reference(_) | Given::References(_) => Form::Other,
    }
}

/// Words and quoted phrases only: no condition, unit or stage.
fn search_words(text: &str) -> bool {
    let first = text.split_whitespace().next().unwrap_or_default();
    !text.is_empty()
        && !text.contains([':', '|', '(', ')'])
        && polylogue_query::ast::Unit::all().all(|unit| unit.name() != first)
}

/// The one session a session word names.
fn one_session(call: &Call<'_>, text: &str) -> Result<SessionId, Failure> {
    let reader = call.reader()?;
    let rows = crate::daemon::service::query_targets(call.daemon, &reader, text, call.invocation)?;
    match rows.as_slice() {
        [Reference::Session(session)] => Ok(session.clone()),
        _ => Err(Failure::new(
            polylogue_model::Code::Ambiguous,
            format!("{text:?} names {} rows, not one session", rows.len()),
        )),
    }
}
