//! One line: a selection and the verbs that act on its rows, and its answer.
//! The `rows` operation carries it on every surface.
use crate::commands::ToolAccess;
use command_surface::JsonSchema;
use polylogue_model::user::{AssertQueryOutcome, UserOutcome};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::num::NonZeroU64;

/// One line: a selection and the verbs that act on its rows, in order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Rows {
    /// One query or reference: the text inside the quotes on the command line
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<String>,
    /// References a person named one by one (row numbers, standard input, the picker's choice); used instead of a selection and needing no `all`
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(extend("unlisted" = true))]
    pub references: Vec<String>,
    /// The verbs, in order; none lists the rows
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<Step>,
    /// Act on every row a query matches
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub all: bool,
    /// Resolve and count the rows; each step says what it would do; nothing is written or sent
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dry_run: bool,
    /// Say how the line is read and what each step would do; run nothing
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub explain: bool,
    /// Rows per page of the listing
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_size: Option<NonZeroU64>,
    /// Columns of the listing
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<String>,
    /// Say why rows matched
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub why: bool,
    /// Return when rows match or this long passes (`30m`)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wait: Option<String>,
    /// Return the complete accepted population for a client picker boundary
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    #[schemars(extend("unlisted" = true))]
    pub return_selection: bool,
}

pub use crate::daemon::verbs::Step;

impl Rows {
    /// The highest step's access; a line without steps reads.
    pub fn access(&self) -> ToolAccess {
        self.steps
            .iter()
            .map(Step::access)
            .max()
            .unwrap_or(ToolAccess::Read)
    }
}

/// What a line answers: the listing, the reports of its steps, or how it is read.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "reply", rename_all = "snake_case")]
pub enum LineAnswer {
    Listing(Box<super::query::Answer>),
    Steps(StepsAnswer),
    Explained(LineExplanation),
}

impl LineAnswer {
    /// The failure of the step that stopped the line, when one did.
    pub fn failed_step(&self) -> Option<&command_surface::Failure> {
        let Self::Steps(steps) = self else {
            return None;
        };
        steps.steps.iter().find_map(|step| match &step.outcome {
            StepOutcome::Failed { error } => Some(error),
            _ => None,
        })
    }
}

/// The rows a line resolved and what each step did with them.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StepsAnswer {
    pub resolved: Option<Resolved>,
    pub steps: Vec<StepReport>,
}

/// The rows the line acted on, resolved once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Resolved {
    /// The selection's query text, when it is a query
    pub query: Option<String>,
    pub count: u64,
    /// The rows were named one by one rather than matched by a query
    pub named: bool,
    /// The first rows' references, at most twenty
    pub first: Vec<String>,
    /// The requested complete population, keeping its original grain and bindings
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population: Option<Box<command_surface::Answer<super::query::Answer>>>,
}

/// One step's request and what came of it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StepReport {
    pub verb: String,
    /// The step's own request ID, which a retried line repeats
    pub request_id: Option<String>,
    pub outcome: StepOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StepOutcome {
    Done {
        reply: StepAnswer,
        /// The lines that take the step's writes back, one per request
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        undo: Vec<String>,
    },
    Failed {
        error: command_surface::Failure,
    },
    /// An earlier step failed.
    NotRun,
    /// A dry run: what the step would do, and the refusal it would meet.
    WouldDo {
        would: String,
        refusal: Option<command_surface::Failure>,
    },
}

/// A finished step's reply.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "reply", rename_all = "snake_case")]
pub enum StepAnswer {
    Read(Box<polylogue_model::read::ReadPage>),
    /// One checked range of an attachment's bytes.
    Attachment(Box<crate::daemon::attachment::BytePage>),
    /// The read's records were delivered before the next step began.
    Streamed(Box<polylogue_model::read::ReadFooter>),
    /// The follow's events were delivered through these final marks.
    Followed(Box<super::follow::Marks>),
    Picked(Vec<String>),
    Brief(Box<crate::brief::BriefReply>),
    Wrote(UserOutcome),
    WroteEach(AssertQueryOutcome),
    Saved(SaveReport),
    Sent(super::acts::SendReport),
    /// What `answer`, `interrupt` and `resume --in` did to each session.
    Acted(Vec<super::acts::Acted>),
    Resume(Box<super::resume::Answer>),
}

pub use polylogue_model::user::SaveReport;

/// The attachment range a line's first step answered with.
#[cfg(test)]
pub(crate) fn attachment_page(answer: &Value) -> crate::daemon::attachment::BytePage {
    let line: LineAnswer = serde_json::from_value(answer.clone()).expect("a line answer");
    let LineAnswer::Steps(steps) = line else {
        panic!("not a step answer: {answer}");
    };
    match steps.steps.into_iter().next().map(|step| step.outcome) {
        Some(StepOutcome::Done {
            reply: StepAnswer::Attachment(page),
            ..
        }) => *page,
        _ => panic!("no attachment range: {answer}"),
    }
}

/// How a line is read, without running it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LineExplanation {
    pub form: SelectionForm,
    /// The selection query's plan
    pub plan: Option<Value>,
    pub steps: Vec<StepExplanation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StepExplanation {
    pub verb: String,
    pub would: String,
    pub access: ToolAccess,
}

polylogue_model::stored_names! {
    /// What the selection is.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SelectionForm {
        None => "none",
        Query => "query",
        Reference => "reference",
        References => "references",
        KeptResult => "kept_result",
    }
}
