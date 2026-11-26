//! The verbs of a line. Each verb is a [`Verb`]: its step type, which rows
//! it accepts, what it does on the selection the line resolved once, and
//! what it would do under a dry run. The bodies delegate to the handlers of
//! the operations the verbs replace, or refuse an option whose body is not
//! built yet with a typed `not_built`.
pub(crate) mod act;
pub(crate) mod annotate;
pub(crate) mod brief;
pub(crate) mod read;
pub(crate) mod resume;
pub(crate) mod save;

use crate::daemon::Failure;
use crate::daemon::handler::Call;
use crate::protocol::line::{Step, StepAnswer};
use command_surface::{Answer, Envelope, Outcome};
use polylogue_model::{Code, Reference};

/// One verb of a line.
pub(crate) trait Verb {
    type Step;
    const NAME: &'static str;
    const ROWS: RowRule;
    fn run(
        step: Self::Step,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure>;
    /// What the step would do on these rows; the dry run writes and sends nothing.
    fn would(
        step: &Self::Step,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<String, Failure> {
        let _ = (step, call);
        Ok(format!("would {} {}", Self::NAME, selection.described()))
    }
}

/// Which rows a verb acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowRule {
    /// Grouped and measured rows too: `read`, `brief`, `save`.
    Any,
    /// Archive rows only.
    Things,
    /// Exactly one archive row: `resume`.
    One,
}

/// A finished step: its reply, the lines that take its writes back, and the
/// notes and next calls of the handler it delegated to.
pub(crate) struct StepDone {
    pub reply: StepAnswer,
    pub undo: Vec<String>,
    pub envelope: Envelope,
    pub outcome: Outcome<()>,
}

/// What a step receives: the line's call with the step's own request ID.
pub(crate) struct StepCall<'a> {
    pub call: Call<'a>,
}

impl<'a> StepCall<'a> {
    /// The step's call with a request ID of its own; the line's cursor
    /// belongs to its listing, so a step starts without it. The line's budget
    /// carries over unless the step sets its own.
    pub(crate) fn new(line: &Call<'a>, request_id: &'a str) -> Self {
        Self {
            call: Call {
                request_id,
                cursor: None,
                budget: line.budget,
                ..line.copy()
            },
        }
    }
}

/// The selection as the line gave it and the rows it resolved to.
pub(crate) struct Selection {
    pub given: Given,
    /// The rows, resolved once before the first step; absent when the line
    /// does not resolve (a single read-only step streams as it reads).
    pub resolved: Option<Vec<Reference>>,
    /// The query text when the selection is a query.
    pub question: Option<String>,
    pub retained: Option<Box<polylogue_model::user::LineSelection>>,
}

/// What the line names.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Given {
    None,
    Query(String),
    Reference(Reference),
    References(Vec<Reference>),
}

impl Selection {
    /// The query a delegated handler runs: the resolved rows, the named
    /// references, or the query, as one query (references joined with OR).
    pub fn query_text(&self) -> Option<String> {
        if let Some(resolved) = &self.resolved {
            return Some(joined(resolved));
        }
        match &self.given {
            Given::None => None,
            Given::Query(text) => Some(text.clone()),
            Given::Reference(reference) => Some(reference.to_string()),
            Given::References(references) => Some(joined(references)),
        }
    }

    /// The rows a step acts on one by one.
    pub fn references(&self) -> Result<&[Reference], Failure> {
        match (&self.resolved, &self.given) {
            (Some(resolved), _) => Ok(resolved),
            (None, Given::Reference(reference)) => Ok(std::slice::from_ref(reference)),
            (None, Given::References(references)) => Ok(references),
            (None, Given::None) => Err(Failure::bad_request("this step needs rows")),
            (None, Given::Query(_)) => Err(Failure::internal(
                "a step over a query's rows ran before the line resolved them",
            )),
        }
    }

    /// How many rows, when known.
    pub fn count(&self) -> Option<u64> {
        self.references()
            .ok()
            .map(|references| references.len() as u64)
    }

    /// The rows in words, for what a step would do.
    pub fn described(&self) -> String {
        let rows = |count: u64| match count {
            1 => "1 row".to_owned(),
            count => format!("{count} rows"),
        };
        match (&self.given, self.count()) {
            (Given::None, _) => "with no selection".into(),
            (Given::Reference(reference), _) => reference.to_string(),
            (Given::Query(text), Some(count)) => format!("{} of {text:?}", rows(count)),
            (Given::Query(text), None) => format!("the rows of {text:?}"),
            (Given::References(_), Some(count)) => rows(count),
            (Given::References(_), None) => "the named rows".into(),
        }
    }
}

fn joined(references: &[Reference]) -> String {
    references
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// The rule of the step's verb.
pub(crate) fn rows_rule(step: &Step) -> RowRule {
    match step {
        Step::Read(_) => read::Read::ROWS,
        Step::Pick(_) => Pick::ROWS,
        Step::Brief(_) => brief::Brief::ROWS,
        Step::Note(_) => annotate::Note::ROWS,
        Step::Tag(_) => annotate::Tag::ROWS,
        Step::Edit(_) => annotate::Edit::ROWS,
        Step::Withdraw(_) => annotate::Withdraw::ROWS,
        Step::Judge(_) => annotate::Judge::ROWS,
        Step::Save(_) => save::Save::ROWS,
        Step::Send(_) => act::Send::ROWS,
        Step::Answer(_) => act::Answer::ROWS,
        Step::Interrupt(_) => act::Interrupt::ROWS,
        Step::Resume(_) => resume::Resume::ROWS,
    }
}

/// Runs one step.
pub(crate) fn run_step(
    step: Step,
    selection: &Selection,
    call: &StepCall<'_>,
) -> Result<StepDone, Failure> {
    match step {
        Step::Read(step) => read::Read::run(*step, selection, call),
        Step::Pick(step) => Pick::run(step, selection, call),
        Step::Brief(step) => brief::Brief::run(step, selection, call),
        Step::Note(step) => annotate::Note::run(*step, selection, call),
        Step::Tag(step) => annotate::Tag::run(step, selection, call),
        Step::Edit(step) => annotate::Edit::run(step, selection, call),
        Step::Withdraw(step) => annotate::Withdraw::run(step, selection, call),
        Step::Judge(step) => annotate::Judge::run(step, selection, call),
        Step::Save(step) => save::Save::run(*step, selection, call),
        Step::Send(step) => act::Send::run(step, selection, call),
        Step::Answer(step) => act::Answer::run(step, selection, call),
        Step::Interrupt(step) => act::Interrupt::run(step, selection, call),
        Step::Resume(step) => resume::Resume::run(step, selection, call),
    }
}

/// What one step would do, or the refusal it would meet.
pub(crate) fn would(
    step: &Step,
    selection: &Selection,
    call: &StepCall<'_>,
) -> Result<String, Failure> {
    match step {
        Step::Read(step) => read::Read::would(step, selection, call),
        Step::Pick(step) => Pick::would(step, selection, call),
        Step::Brief(step) => brief::Brief::would(step, selection, call),
        Step::Note(step) => annotate::Note::would(step, selection, call),
        Step::Tag(step) => annotate::Tag::would(step, selection, call),
        Step::Edit(step) => annotate::Edit::would(step, selection, call),
        Step::Withdraw(step) => annotate::Withdraw::would(step, selection, call),
        Step::Judge(step) => annotate::Judge::would(step, selection, call),
        Step::Save(step) => save::Save::would(step, selection, call),
        Step::Send(step) => act::Send::would(step, selection, call),
        Step::Answer(step) => act::Answer::would(step, selection, call),
        Step::Interrupt(step) => act::Interrupt::would(step, selection, call),
        Step::Resume(step) => resume::Resume::would(step, selection, call),
    }
}

/// The command that withdraws exactly the revisions this write created.
pub(crate) fn undo_line(
    request_id: &str,
    committed: &[Reference],
) -> Result<Option<String>, Failure> {
    if committed.is_empty() {
        return Ok(None);
    }
    let rows = crate::protocol::line::Rows {
        references: committed.iter().map(ToString::to_string).collect(),
        steps: vec![Step::Withdraw(
            crate::protocol::steps::WithdrawStep::default(),
        )],
        all: committed.len() > 1,
        ..Default::default()
    };
    let request = crate::client::request(
        "rows",
        serde_json::to_value(rows).map_err(Failure::internal)?,
        format!("{request_id}-undo"),
    );
    crate::commands::declarations()
        .cli_command(&request)
        .map(Some)
        .map_err(|mut error| {
            error.message = format!(
                "the write committed, but its undo command could not be written: {}",
                error.message
            );
            error.hint = Some(format!(
                "written revisions: {}",
                committed
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            Failure::from(error)
        })
}

/// The refusal of an option whose body is not built yet.
pub(crate) fn not_built(verb: &str, option: &str) -> Failure {
    Failure::new(
        Code::NotBuilt,
        format!("{verb} --{option} is not built yet"),
    )
}

/// The refusal of a use of a verb whose body is not built yet.
pub(crate) fn not_built_use(verb: &str, what: &str) -> Failure {
    Failure::new(Code::NotBuilt, format!("{verb} {what} is not built yet"))
}

/// A delegated handler's answer as a finished step: its failure is the
/// step's; its notes and next calls go to the line.
pub(crate) fn delegated<T>(
    answer: Answer<T>,
    reply: impl FnOnce(T) -> StepAnswer,
) -> Result<StepDone, Failure> {
    let Answer { outcome, envelope } = answer;
    let (value, outcome) = match outcome {
        Outcome::Ok(value) => (value, Outcome::Ok(())),
        Outcome::Empty(value) => (value, Outcome::Empty(())),
        Outcome::Degraded { value, missing } => (value, Outcome::Degraded { value: (), missing }),
        Outcome::Error(error) => return Err(Failure::from(error)),
    };
    Ok(StepDone {
        reply: reply(value),
        undo: Vec::new(),
        envelope,
        outcome,
    })
}

/// The sessions among the rows, refusing any other row.
pub(crate) fn sessions(
    verb: &str,
    selection: &Selection,
) -> Result<Vec<polylogue_model::SessionId>, Failure> {
    selection
        .references()?
        .iter()
        .map(|reference| match reference {
            Reference::Session(session) => Ok(session.clone()),
            other => Err(Failure::bad_request(format!(
                "{verb} acts on sessions; {other} is not one"
            ))),
        })
        .collect()
}

/// `pick`: the command line opens the picker and sends the chosen
/// references; the daemon has no person to ask.
pub(crate) struct Pick;
impl Verb for Pick {
    type Step = crate::protocol::steps::PickStep;
    const NAME: &'static str = "pick";
    const ROWS: RowRule = RowRule::Any;
    fn run(_: Self::Step, _: &Selection, _: &StepCall<'_>) -> Result<StepDone, Failure> {
        Err(refused_pick())
    }
    fn would(_: &Self::Step, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        Ok(format!(
            "would open the picker over {}",
            selection.described()
        ))
    }
}

pub(crate) fn refused_pick() -> Failure {
    Failure::bad_request("pick runs on the command line; send the chosen references")
}

/// A line is empty only when all its completed steps are empty; missing
/// parts survive a later successful step.
pub(crate) fn merge_outcome(current: &mut Outcome<()>, next: Outcome<()>) {
    match next {
        Outcome::Degraded { missing, .. } => match current {
            Outcome::Degraded {
                missing: previous, ..
            } => previous.extend(missing),
            _ => *current = Outcome::Degraded { value: (), missing },
        },
        Outcome::Ok(()) if matches!(current, Outcome::Empty(())) => *current = Outcome::Ok(()),
        Outcome::Error(error) => *current = Outcome::Error(error),
        _ => {}
    }
}
