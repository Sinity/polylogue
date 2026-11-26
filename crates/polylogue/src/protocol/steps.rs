//! The verbs of a line, one typed step each. A field is one option
//! of its verb on every surface: `--name` on the command line (with `-` for
//! `_`), the key beside `verb` in an MCP or HTTP step. Its doc comment is
//! its help; the schema key `argument_form = "positional"` marks the one
//! word a verb takes after its name.
use crate::brief::StartSource;
use command_surface::JsonSchema;
use polylogue_model::read::{ReadBudget, ReadDelivery, ReadOptions};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The shipped selection for a read invoked without explicit rows.
pub const fn usual_read_selection(follow: bool) -> &'static str {
    if follow { "running" } else { "recent" }
}

/// `read`: show the rows' content. How to read them is [`ReadOptions`], the
/// type the reading receives; the rest says how the page is delivered.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadStep {
    /// The command-line consumer displays this page
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    #[schemars(extend("unlisted" = true))]
    pub displayed: bool,
    #[serde(flatten)]
    pub options: ReadOptions,
    /// End before this message, time or `compaction[:N]`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub until: Option<String>,
    /// The first N messages of each row
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first: Option<u64>,
    /// The last N messages of each row
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last: Option<u64>,
    /// Show the alternate beside its original
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub beside: bool,
    /// Compare a kept result's rows with their identities removed
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub blind: bool,
    /// The seed of a blind reading's labels
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Keep printing what the rows' sessions do; the last step of its line
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub follow: bool,
    /// With follow: stop when every selected session settles
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub until_idle: bool,
    /// How the compare view lines rows up: prompt, lineage, time or file
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align: Option<String>,
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

/// `brief`: packaged context for this directory, a session, search words or other rows.
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
    /// One part per message, as lines of a file, or `-` for standard input
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parts: Option<String>,
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
    /// A query whose result checks the note
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub check: Option<String>,
    /// What the check query is expected to return
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expect: Option<String>,
    /// The page it is written on
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page: Option<String>,
    /// The page section it is placed in
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
    /// A heading's level
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<u64>,
    /// The template a new page starts from
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// A template parameter, KEY=VALUE
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub param: Vec<String>,
    /// A page's layout: linear, board or table
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<String>,
    /// A page's subject or topic
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub about: Option<String>,
    /// A member's role on a page
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// How a page enters briefs: off, contents or full
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brief: Option<String>,
    /// Also rename the web chat in its provider
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub rename: bool,
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

/// `judge VERDICT`: accept, reject or defer each annotation, or say which of several rows is better.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JudgeStep {
    /// accept, reject or defer; on several rows the preferred row, an order (`B>A>C`), tie, incomparable, abstain or insufficient-evidence
    #[schemars(extend("argument_form" = "positional"))]
    pub verdict: String,
    /// The dimension of a comparison
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on: Option<String>,
    /// The seed of the blind reading whose labels the verdict names
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Why
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// `save [NAME]`: name the question and keep this line's rows.
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
    /// The page section the rows go in
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
    /// The slot the rows go after
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
    /// Place these rows as they are now
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub fixed: bool,
    /// Place the question, run again when the page is read
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub live: bool,
    /// Place the kept result
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub kept: bool,
    /// The template a new page starts from
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// How the rows show on the page: card, excerpt, text, full, table, list, count, value, chart or compare
    #[serde(rename = "as", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

/// `send TEXT`: send text to each session.
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
    /// A page sent with the message
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page: Option<String>,
    /// The page section sent with the message
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
}

/// `answer VALUE`: respond to what a session is waiting for.
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

/// `resume`: reopen a session where it ran, or continue its work in another harness.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResumeStep {
    /// Fork the session instead of continuing it
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub fork: bool,
    /// Reopen a running session in a second client where its harness allows
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub anyway: bool,
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
