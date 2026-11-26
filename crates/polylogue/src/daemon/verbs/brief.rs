//! `brief`: packaged context, chosen by the selection's form. No selection
//! is the directory's brief, one session is that session's brief (to take
//! it over), three or more named sessions are an index of them, and search
//! words are what is already known about them.
use super::{Given, RowRule, Selection, StepCall, StepDone, Verb, not_built_use};
use crate::brief::{Ask, BriefReply};
use crate::commands::ToolAccess;
use crate::daemon::Failure;
use crate::daemon::handler::Call;
use crate::daemon::service::handlers::prepare_context;
use crate::protocol::line::StepAnswer;
use crate::protocol::steps::BriefStep;
use command_surface::{Answer, TokenBudget};
use polylogue_model::read::ReadBudget;
use polylogue_model::{Reason, Reference, SessionId};

#[cfg(test)]
mod tests;

pub(crate) struct Brief;
impl Verb for Brief {
    type Step = BriefStep;
    const NAME: &'static str = "brief";
    const ROWS: RowRule = RowRule::Any;
    const ACCESS: ToolAccess = ToolAccess::Read;
    fn run(
        step: BriefStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let for_session = step
            .for_session
            .as_deref()
            .map(|text| {
                SessionId::parse(text).map_err(|problem| {
                    Failure::refusal(Reason::BadReference {
                        text: text.into(),
                        problem: problem.to_string(),
                    })
                })
            })
            .transpose()?;
        let budget = match step.budget {
            ReadBudget::Tokens(tokens) => {
                TokenBudget::Tokens(u32::try_from(tokens).map_err(|_| {
                    Failure::refusal(Reason::BudgetTooLarge {
                        most: u64::from(u32::MAX),
                    })
                })?)
            }
            ReadBudget::Unlimited => TokenBudget::Unlimited,
            ReadBudget::Default => call.call.budget,
        };
        let cursor = step.cursor.clone();
        let call = Call {
            cursor: cursor.as_deref(),
            budget,
            ..call.call.copy()
        };
        let answer = match form(selection) {
            Form::Directory => directory(step, for_session, &call)?,
            Form::Session(session) => self::session(session, for_session, &call)?,
            Form::SessionWord(text) => {
                self::session(one_session(&call, &text)?, for_session, &call)?
            }
            Form::Sessions(sessions) => prepare_context(
                &call,
                Ask::Sessions {
                    sessions,
                    for_session,
                },
            )?,
            Form::Words(text) => words(text, step.cwd, for_session, &call)?,
            Form::Other => return Err(not_built_use("brief", "over these rows")),
        };
        Ok(answer
            .map(|reply| StepAnswer::Brief(Box::new(reply)))
            .into())
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

/// The brief of the caller's directory, at session start or later.
fn directory(
    step: BriefStep,
    for_session: Option<SessionId>,
    call: &Call<'_>,
) -> Result<Answer<BriefReply>, Failure> {
    let cwd = call.directory(step.cwd.as_deref())?;
    prepare_context(
        call,
        Ask::SessionStart {
            cwd,
            session: for_session,
            source: step.source,
        },
    )
}

/// One session's brief, for the session that takes it over.
fn session(
    session: SessionId,
    for_session: Option<SessionId>,
    call: &Call<'_>,
) -> Result<Answer<BriefReply>, Failure> {
    prepare_context(
        call,
        Ask::Session {
            session,
            for_session,
        },
    )
}

/// What is already known about search words, a page at a time.
fn words(
    text: String,
    cwd: Option<std::path::PathBuf>,
    for_session: Option<SessionId>,
    call: &Call<'_>,
) -> Result<Answer<BriefReply>, Failure> {
    call.runs_query(&text);
    prepare_context(
        call,
        Ask::Words {
            text,
            cwd: call.directory(cwd.as_deref()).ok(),
            for_session,
            cursor: call.cursor.map(str::to_owned),
        },
    )
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
        _ => Err(Failure::refusal(Reason::NotOneSession {
            text: text.into(),
            rows: rows.len(),
        })),
    }
}
