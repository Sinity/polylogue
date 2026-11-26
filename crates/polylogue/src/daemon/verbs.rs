//! The verbs of a line. Each verb is a [`Verb`]: its step type, which rows
//! it accepts, the smallest tool that may run it, what it does on the
//! selection the line resolved once, and what it would do under a dry run.
//! [`verbs!`] declares the line's [`Step`] from the list of verbs, so a verb
//! is added in one place.
pub(crate) mod act;
pub(crate) mod annotate;
pub(crate) mod brief;
pub(crate) mod read;
pub(crate) mod resume;
pub(crate) mod save;

use crate::commands::ToolAccess;
use crate::daemon::Failure;
use crate::daemon::handler::Call;
use crate::protocol::line::StepAnswer;
use crate::protocol::steps::*;
use command_surface::{Answer, JsonSchema, Outcome, StepDeclarations};
use polylogue_model::{Code, Reason, Reference};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// One verb of a line.
pub(crate) trait Verb {
    type Step;
    const NAME: &'static str;
    const ROWS: RowRule;
    /// The smallest tool that may run the step.
    const ACCESS: ToolAccess;
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

/// Declares the line's [`Step`] from its verbs, in the order of the verb
/// table: each variant's step type and the [`Verb`] that runs it. A boxed
/// step is marked `boxed`.
macro_rules! verbs {
    ($($variant:ident($step:ty $(, $boxed:ident)?) => $verb:ty;)+) => {
        /// One verb of a line with its options.
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
        #[serde(tag = "verb", rename_all = "snake_case")]
        pub enum Step {
            $($variant(verbs!(@type $step $(, $boxed)?)),)+
        }

        impl Step {
            /// The verb as written on every surface.
            pub fn verb(&self) -> &'static str {
                match self {
                    $(Self::$variant(_) => <$verb as Verb>::NAME,)+
                }
            }

            /// The smallest tool that may run this step.
            pub fn access(&self) -> ToolAccess {
                match self {
                    $(Self::$variant(_) => <$verb as Verb>::ACCESS,)+
                }
            }

            /// Which rows the step's verb acts on.
            pub(crate) fn rows(&self) -> RowRule {
                match self {
                    $(Self::$variant(_) => <$verb as Verb>::ROWS,)+
                }
            }

            /// Runs the step; an answer that failed is the step's failure.
            pub(crate) fn run(
                self,
                selection: &Selection,
                call: &StepCall<'_>,
            ) -> Result<StepDone, Failure> {
                match self {
                    $(Self::$variant(step) => {
                        <$verb as Verb>::run(verbs!(@own step $(, $boxed)?), selection, call)
                    })+
                }
                .and_then(StepDone::checked)
            }

            /// What the step would do, or the refusal it would meet.
            pub(crate) fn would(
                &self,
                selection: &Selection,
                call: &StepCall<'_>,
            ) -> Result<String, Failure> {
                match self {
                    $(Self::$variant(step) => <$verb as Verb>::would(step, selection, call),)+
                }
            }
        }

        /// Every verb's declaration, in the order of the verb table.
        pub fn declarations() -> StepDeclarations {
            fn declare<A: DeserializeOwned + JsonSchema>(steps: &mut StepDeclarations, verb: &str) {
                steps
                    .declare::<A>(
                        verb,
                        crate::commands::arguments::from_schema(&schemars::schema_for!(A), false),
                    )
                    .unwrap_or_else(|failure| panic!("{verb}: {}", failure.message));
            }
            let mut steps = StepDeclarations::default();
            $(declare::<$step>(&mut steps, <$verb as Verb>::NAME);)+
            steps.cli_selection("selection", "references")
        }
    };
    (@type $step:ty, boxed) => { Box<$step> };
    (@type $step:ty) => { $step };
    (@own $value:ident, boxed) => { *$value };
    (@own $value:ident) => { $value };
}

verbs! {
    Read(ReadStep, boxed) => read::Read;
    Pick(PickStep) => Pick;
    Brief(BriefStep) => brief::Brief;
    Note(NoteStep, boxed) => annotate::Note;
    Tag(TagStep) => annotate::Tag;
    Edit(EditStep) => annotate::Edit;
    Withdraw(WithdrawStep) => annotate::Withdraw;
    Judge(JudgeStep) => annotate::Judge;
    Save(SaveStep, boxed) => save::Save;
    Send(SendStep) => act::Send;
    Answer(AnswerStep) => act::Answer;
    Interrupt(InterruptStep) => act::Interrupt;
    Resume(ResumeStep) => resume::Resume;
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

/// A finished step: its answer (the reply with its notes, next calls and
/// how complete it is) and the lines that take its writes back.
pub(crate) struct StepDone {
    pub answer: Answer<StepAnswer>,
    pub undo: Vec<String>,
}

impl StepDone {
    /// A failed answer is the step's failure.
    pub(crate) fn checked(self) -> Result<Self, Failure> {
        match self.answer.outcome {
            Outcome::Error(error) => Err(Failure::from(error)),
            _ => Ok(self),
        }
    }

    /// The reply alone, and how complete it is.
    pub(crate) fn outcome(&self) -> Outcome<()> {
        match &self.answer.outcome {
            Outcome::Ok(_) => Outcome::Ok(()),
            Outcome::Empty(_) => Outcome::Empty(()),
            Outcome::Degraded { missing, .. } => Outcome::Degraded {
                value: (),
                missing: missing.clone(),
            },
            Outcome::Error(error) => Outcome::Error(error.clone()),
        }
    }
}

impl From<Answer<StepAnswer>> for StepDone {
    fn from(answer: Answer<StepAnswer>) -> Self {
        Self {
            answer,
            undo: Vec::new(),
        }
    }
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
    pub query: Option<String>,
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
            (None, Given::None) => Err(Reason::StepNeedsRows {}.into()),
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
        steps: vec![Step::Withdraw(WithdrawStep::default())],
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
            other => Err(Failure::refusal(Reason::ActsOnSessions {
                verb: verb.into(),
                reference: other.to_string(),
            })),
        })
        .collect()
}

/// `pick`: the command line opens the picker and sends the chosen
/// references; the daemon has no person to ask.
pub(crate) struct Pick;
impl Verb for Pick {
    type Step = PickStep;
    const NAME: &'static str = "pick";
    const ROWS: RowRule = RowRule::Any;
    const ACCESS: ToolAccess = ToolAccess::Read;
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
    Reason::PickOnCommandLine {}.into()
}

/// One recipient's answer within a step over several: its value, its
/// notes and next calls added to the step's, its failure the step's.
pub(crate) fn gather<T>(
    answer: Answer<T>,
    outcome: &mut Outcome<()>,
    envelope: &mut command_surface::Envelope,
) -> Result<T, Failure> {
    envelope.notes.extend(answer.envelope.notes);
    envelope.next.extend(answer.envelope.next);
    match answer.outcome {
        Outcome::Error(error) => Err(Failure::from(error)),
        Outcome::Ok(value) => {
            merge_outcome(outcome, Outcome::Ok(()));
            Ok(value)
        }
        Outcome::Empty(value) => Ok(value),
        Outcome::Degraded { value, missing } => {
            merge_outcome(outcome, Outcome::Degraded { value: (), missing });
            Ok(value)
        }
    }
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
