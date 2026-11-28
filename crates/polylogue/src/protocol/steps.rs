//! The verbs of a line, one typed step each. A field is one option
//! of its verb on every surface: `--name` on the command line (with `-` for
//! `_`), the key beside `verb` in an MCP or HTTP step. Its doc comment is
//! its help; the schema key `argument_form = "positional"` marks the one
//! word a verb takes after its name. The first sentence of a step type's
//! doc comment is the verb's line in the MCP tool descriptions. An option
//! that is designed and not built has no field here; it is on the list in
//! `crate::unbuilt`.
use crate::brief::StartSource;
use command_surface::JsonSchema;
use polylogue_model::read::{ReadBudget, ReadDelivery, ReadOptions};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The shipped selection for a read invoked without explicit rows.
pub const fn usual_read_selection(follow: bool) -> &'static str {
    if follow { "running" } else { "recent" }
}

/// `read`: show the rows' content in a view, a page at a time. How to read
/// them is [`ReadOptions`], the type the reading receives; the rest says how
/// the page is delivered.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadStep {
    /// The command-line consumer displays this page
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    #[schemars(extend("unlisted" = true))]
    pub displayed: bool,
    #[serde(flatten)]
    pub options: ReadOptions,
    /// Keep printing what the rows' sessions do; the last step of its line
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub follow: bool,
    /// The document encoding
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Tokens in one page; none lifts the token cap while transfer bounds remain
    #[serde(default, skip_serializing_if = "ReadBudget::is_default")]
    pub budget: ReadBudget,
    /// Continue from a previous page's cursor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    /// A page, a stream the command line writes a file from, or a bundle's stream
    #[serde(default, skip_serializing_if = "is_page")]
    #[schemars(extend("unlisted" = true))]
    pub delivery: ReadDelivery,
    /// With an attachment: the stored input to read when several hold its bytes
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
}

fn is_page(delivery: &ReadDelivery) -> bool {
    *delivery == ReadDelivery::Page
}

/// `pick`: choose rows in the picker; the verbs after it act on the chosen ones.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(extend("properties" = {}))]
pub struct PickStep {}

/// `brief`: package context: with no selection for the directory `cwd`, of a session to take it over, of search words what is already known.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BriefStep {
    /// The directory (default: the caller's working directory)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// The session that will continue from the brief
    #[serde(rename = "for", default, skip_serializing_if = "Option::is_none")]
    pub for_session: Option<String>,
    /// Tokens the brief may hold; none lifts the token cap
    #[serde(default, skip_serializing_if = "ReadBudget::is_default")]
    pub budget: ReadBudget,
    /// Continue from a previous page's cursor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    /// Why the session starts, as the session-start hook reports it
    #[serde(default)]
    #[schemars(extend("unlisted" = true))]
    pub source: StartSource,
}

/// `note TEXT`: write text about each row, of a kind.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NoteStep {
    /// The text: Markdown with references, or `-` for standard input
    #[schemars(extend("argument_form" = "positional"))]
    pub text: String,
    /// The kind of note (default `note`): decision, finding, title, outcome, translation:LANG, page, heading, …
    #[serde(rename = "as", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// The key of a kind keyed by name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// References the note rests on
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<String>,
    /// The note it answers
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub replies_to: Vec<String>,
    /// Records it supports
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supports: Vec<String>,
    /// Records it challenges
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub challenges: Vec<String>,
    /// Records it qualifies
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub qualifies: Vec<String>,
    /// Records it corrects
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub corrects: Vec<String>,
    /// Records it is derived from
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub derived_from: Vec<String>,
    /// Records it blocks
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocks: Vec<String>,
    /// Where it applies
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scope: Vec<String>,
    /// When it stops holding
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<String>,
}

/// `tag NAME`: label each row.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TagStep {
    /// The tag: NAME or NAME:VALUE
    #[schemars(extend("argument_form" = "positional"))]
    pub name: String,
}

/// `edit [TEXT]`: write a new revision of each annotation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EditStep {
    /// The new text, or `-` for standard input; without it a terminal opens your editor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("argument_form" = "positional"))]
    pub text: Option<String>,
    /// A new key
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
}

/// `withdraw`: take each annotation back; nothing is deleted.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WithdrawStep {
    /// Why
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// `judge VERDICT`: accept or reject an annotation an agent wrote.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JudgeStep {
    /// accept, reject or defer; on several rows the preferred row, an order (`B>A>C`), tie, incomparable, abstain or insufficient-evidence
    #[schemars(extend("argument_form" = "positional"))]
    pub verdict: String,
    /// Why
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// `save [NAME]`: name the question as query:NAME and keep this line's rows as result:ID; watch tells you when rows start matching.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SaveStep {
    /// The saved query's name, or page:NAME to place the rows on a page
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("argument_form" = "positional"))]
    pub name: Option<String>,
    /// Tell me when rows start matching
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub watch: bool,
    /// How long a watched row must keep matching, such as 20m
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hold: Option<String>,
    /// Also show each notice as a desktop notification
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub notify: bool,
    /// The denominator query of a measure or aggregate figure
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub denominator: Option<String>,
}

/// `send TEXT`: send text to each session: a running one gets it at its next tool call or prompt, one ended session is resumed with it, and a web chat turn waits for the operator's approval.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SendStep {
    /// The message, or `-` for standard input
    #[schemars(extend("argument_form" = "positional"))]
    pub text: String,
    /// A reference the message is about
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub about: Option<String>,
    /// Files sent with the message
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attach: Vec<String>,
}

/// `answer VALUE`: allow or deny what a session is waiting for.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnswerStep {
    /// allow, deny, or the text of an answer
    #[schemars(extend("argument_form" = "positional"))]
    pub value: String,
    /// A reason given with the answer
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Answer the same tool and input whenever the session asks next
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub standing: bool,
}

/// `interrupt`: stop a session's current turn.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(extend("properties" = {}))]
pub struct InterruptStep {}

/// `resume`: reopen a session where it ran, or with `in` continue its work in a new session of another harness.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResumeStep {
    /// Continue the work in a new session of this harness
    #[serde(rename = "in", default, skip_serializing_if = "Option::is_none")]
    pub harness: Option<String>,
    /// With in: the message to continue from (default the latest)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<String>,
    /// With in: the directory, when not the session's own
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// With in: interrupt the session once the new one runs
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stop_source: bool,
}

#[cfg(test)]
mod tests;
