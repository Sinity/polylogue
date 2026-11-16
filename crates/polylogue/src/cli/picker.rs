//! Inline selection for the command line. The caller supplies row pages,
//! overview readings and actions; the picker owns keys and terminal drawing.

use crate::daemon::Failure;
use std::{future::Future, sync::Arc, time::Duration};

pub type Result<T> = std::result::Result<T, Failure>;

/// One selectable row, including the text used by fuzzy filtering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub reference: String,
    pub title: String,
    pub first_prompt: String,
    pub repository: String,
    pub description: String,
}

/// A transfer page, not a limit on the selection's size.
#[derive(Clone, Debug, Default)]
pub struct Page {
    pub rows: Vec<Candidate>,
    pub count: Option<u64>,
    pub next: Option<String>,
}

#[derive(Clone, Debug)]
pub enum Selection {
    Query(String),
    Ambiguous { question: String, candidates: Page },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Fuzzy,
    Query,
}

/// Query mode replaces the selection; fuzzy mode pages the original selection.
#[derive(Clone, Debug)]
pub struct PageRequest {
    pub selection: String,
    pub cursor: Option<String>,
}

/// The source renders an overview at this width and honours this deadline.
#[derive(Clone, Debug)]
pub struct PreviewRequest {
    pub reference: String,
    pub width: u16,
    pub deadline: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Enter,
    Read,
    Browser,
    Copy,
}

/// Pages and previews are cancellable futures. Actions run with the terminal
/// restored, so a pager can take input and return to this same selection.
pub trait Source: Send + Sync + 'static {
    fn page(&self, request: PageRequest) -> impl Future<Output = Result<Page>> + Send;
    fn preview(&self, request: PreviewRequest) -> impl Future<Output = Result<String>> + Send;
    fn act(&self, action: Action, references: &[String]) -> Result<()>;
    /// Replaces the terminal's row list, in displayed order, with its question.
    fn shown(&self, question: &str, rows: &[Candidate]) -> Result<()>;
}

#[derive(Clone, Debug)]
pub struct Options {
    pub selection: Selection,
    /// The words Enter will run, or empty to print references.
    pub enter: String,
    pub multiple: bool,
    pub json: bool,
    pub format: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Choice {
    Selected(Vec<String>),
    Cancelled,
}

/// Opens over query rows or an ambiguity's candidates and returns references.
/// A complete one-row selection runs the supplied Enter action directly.
pub fn open<S: Source>(source: Arc<S>, options: Options) -> Result<Choice> {
    let _ = (source, options);
    Err(Failure::new(
        polylogue_model::Code::NotBuilt,
        "picker implementation is being assembled",
    ))
}
