//! The options of a reading: its view, what it shows and hides, the budgets
//! of a page, and the place a context or history reading is taken at.
use super::ReadError;
use crate::{BlockKind, Message, PromptKind, Role};
use serde::{Deserialize, Serialize};

/// The moment a reading's annotations and pages are taken at: an exact local
/// archive position or a time. The wire form is one string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum ReadAt {
    Position(crate::ChangePosition),
    Time(crate::Timestamp),
    /// A time in the shared time grammar, resolved against the accepted
    /// read's clock.
    TimeExpression(String),
}
impl TryFrom<String> for ReadAt {
    type Error = String;
    fn try_from(text: String) -> Result<Self, String> {
        // Local clock and date-time words can also have three colon-separated
        // parts, so the shared time grammar reads the word first.
        if query_language::time::is_time_expression(&text) {
            return Ok(match crate::Timestamp::parse_rfc3339(&text) {
                Some(at) => Self::Time(at),
                None => Self::TimeExpression(text),
            });
        }
        if let Some(position) = crate::ChangePosition::parse(&text) {
            return Ok(Self::Position(position));
        }
        if text.trim().is_empty() {
            return Err("read --at requires a time or a local position".into());
        }
        Ok(Self::TimeExpression(text))
    }
}
impl From<ReadAt> for String {
    fn from(at: ReadAt) -> Self {
        at.to_string()
    }
}
impl std::fmt::Display for ReadAt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Position(position) => position.fmt(f),
            Self::Time(at) => at.fmt(f),
            Self::TimeExpression(text) => f.write_str(text),
        }
    }
}

stored_names! {
    name, parse;
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub enum View {
        #[default]
        Transcript => "transcript",
        Dialogue => "dialogue",
        Overview => "overview",
        Context => "context",
        Tree => "tree",
        Changes => "changes",
        Live => "live",
        Raw => "raw",
        Trace => "trace",
    }
}
impl View {
    /// The metadata filter and layout consult this same rule.
    pub const fn block_kinds(self) -> &'static [BlockKind] {
        use BlockKind::*;
        match self {
            Self::Transcript | Self::Context | Self::Raw => {
                &[Text, Thinking, ToolCall, ToolResult, Image, File, Other]
            }
            Self::Dialogue => &[Text],
            Self::Changes => &[ToolCall, ToolResult],
            Self::Overview | Self::Tree | Self::Live | Self::Trace => &[],
        }
    }
    pub fn keeps_message(self, message: &Message) -> bool {
        self.keeps_message_metadata(message.role, message.prompt_kind)
    }
    pub fn keeps_message_metadata(self, role: Role, prompt_kind: Option<PromptKind>) -> bool {
        match self {
            Self::Dialogue => !crate::is_harness_message(role, prompt_kind),
            Self::Transcript | Self::Context | Self::Changes | Self::Raw => true,
            Self::Overview | Self::Tree | Self::Live | Self::Trace => false,
        }
    }
}
impl std::fmt::Display for View {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
impl std::str::FromStr for View {
    type Err = ReadError;
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::parse(name).ok_or_else(|| ReadError::UnknownView { name: name.into() })
    }
}

/// The resolved visibility bound for an annotation revision graph.
#[derive(Debug, Clone, Copy)]
pub enum AnnotationAt {
    Position(i64),
    Time(crate::Timestamp),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Expansion {
    None,
    #[serde(rename = "headers")]
    Collapsed,
    Full,
}

/// How to read the selected rows, as the caller writes it: the same names
/// on every surface. `show` and `hide` stay the words the caller typed; the
/// accessors below expand them where items are filtered.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadOptions {
    /// The reading view
    #[serde(default)]
    pub view: View,
    /// Item kinds to show, beyond the view's own
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub show: Vec<String>,
    /// Item kinds to hide, from the same table as show
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hide: Vec<String>,
    /// Start at this message, time within the session or absolute time; `seen` where this reader stopped
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// Show the first N messages of the selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first: Option<u64>,
    /// Show the last N messages of the selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last: Option<u64>,
    /// N messages either side of the selected item
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub around: Option<u64>,
    /// With the context view: a message or compaction[:N] to compare with
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<String>,
    /// The alternate to read (`translation:pl`), or `none` for the originals
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alternate: Option<String>,
    /// Read only the named physical session
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub physical: bool,
    /// Label the human `user` instead of `you`, for excerpts shared with others
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub neutral_names: bool,
    /// Annotations and pages as they stood at this time or position
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<String>")]
    pub at: Option<ReadAt>,
}

/// Head and tail lines of a tool's output when `tools` names no number.
const OUTPUT_LINES: u64 = 100;

/// One row per show kind: the alternate, its word and its levels, fullest
/// last. The table gives the enum, its words and the `--show` help text, so
/// the CLI, MCP and the web's show and hide keys cannot drift from it.
macro_rules! show_hide_table {
    ($($kind:ident $name:literal [$($level:literal),+] $number:tt;)+) => {
        /// The closed set of item kinds a reading shows or hides, one table for the
        /// CLI's `--show`/`--hide`, MCP's `show`/`hide` and the web's show and hide keys.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum ShowKind {
            $($kind,)+
        }
        impl ShowKind {
            pub const ALL: [Self; [$(stringify!($kind)),+].len()] = [$(Self::$kind,)+];
            /// The `--show` help: every kind with its levels.
            pub const HELP: &'static str = concat!(
                "Item kinds to show, each with an optional level (the fullest when omitted):"
                $(, " ", $name, "[:", show_hide_table!(@levels $($level),+), show_hide_table!(@number $number), "]")+
            );
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$kind => $name,)+
                }
            }
            /// The levels a kind takes, fullest last; `--show KIND` uses the
            /// last.
            pub const fn levels(self) -> &'static [&'static str] {
                match self {
                    $(Self::$kind => &[$($level),+],)+
                }
            }
            /// Whether the kind also takes a positive number as its level.
            pub const fn takes_number(self) -> bool {
                match self {
                    $(Self::$kind => $number,)+
                }
            }
        }
    };
    (@levels $first:literal $(, $rest:literal)*) => {
        concat!($first $(, "|", $rest)*)
    };
    (@number true) => {
        "|N"
    };
    (@number false) => {
        ""
    };
}
show_hide_table! {
    Tools "tools" ["line", "full"] true;
    Thinking "thinking" ["line", "full"] false;
    Harness "harness" ["line", "full"] false;
    Inherited "inherited" ["full"] false;
    Branches "branches" ["full"] false;
    Spawns "spawns" ["card", "full"] true;
    Annotations "annotations" ["header"] false;
    Turns "turns" ["dividers"] false;
    Context "context" ["meter"] false;
}
impl ShowKind {
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }
}

/// How spawned sessions appear at their spawning call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SpawnDisplay {
    Hidden,
    /// A delegation card with the call that reads the child.
    Card,
    /// The child's reading inline after the card, to this depth of nested
    /// spawns; `None` expands every level.
    Inline {
        depth: Option<u32>,
    },
}

/// What the show and hide words ask of one item kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Asked<'a> {
    Hidden,
    /// The level a show word names, its fullest when it names none.
    Level(&'a str),
}

impl ReadOptions {
    /// What the words ask of `kind`: a hide word wins, then the last show
    /// word; none when no word names the kind.
    fn asked(&self, kind: ShowKind) -> Option<Asked<'_>> {
        if self.hide.iter().any(|word| word == kind.name()) {
            return Some(Asked::Hidden);
        }
        self.show.iter().rev().find_map(|word| {
            let (name, level) = word
                .split_once(':')
                .map_or((word.as_str(), None), |(name, level)| (name, Some(level)));
            (name == kind.name())
                .then(|| Asked::Level(level.unwrap_or(kind.levels()[kind.levels().len() - 1])))
        })
    }

    /// A show word's positive number, for the kinds that take one.
    fn number(level: &str) -> Option<u64> {
        level.parse::<u64>().ok().filter(|number| *number > 0)
    }

    fn expansion(&self, kind: ShowKind) -> Expansion {
        match self.asked(kind) {
            Some(Asked::Hidden) => Expansion::None,
            Some(Asked::Level("full")) => Expansion::Full,
            Some(Asked::Level(level)) if Self::number(level).is_some() => Expansion::Full,
            Some(Asked::Level(_)) | None => Expansion::Collapsed,
        }
    }

    fn shown(&self, kind: ShowKind, default: bool) -> bool {
        match self.asked(kind) {
            Some(Asked::Hidden) => false,
            Some(Asked::Level(_)) => true,
            None => default,
        }
    }

    pub fn tools(&self) -> Expansion {
        self.expansion(ShowKind::Tools)
    }
    pub fn thinking(&self) -> Expansion {
        self.expansion(ShowKind::Thinking)
    }
    pub fn harness(&self) -> Expansion {
        self.expansion(ShowKind::Harness)
    }
    /// Head and tail lines of expanded tool output; zero keeps all of it.
    pub fn output_lines(&self) -> u64 {
        match self.asked(ShowKind::Tools) {
            Some(Asked::Level("full")) => 0,
            Some(Asked::Level(level)) => Self::number(level).unwrap_or(OUTPUT_LINES),
            _ => OUTPUT_LINES,
        }
    }
    /// The parent's copied prefix.
    pub fn inherited(&self) -> bool {
        self.shown(ShowKind::Inherited, false)
    }
    /// Abandoned versions under their split point.
    pub fn branches(&self) -> bool {
        self.shown(ShowKind::Branches, false)
    }
    /// Annotations listed under each message.
    pub fn annotations(&self) -> bool {
        self.shown(ShowKind::Annotations, true)
    }
    /// A divider before each prompt.
    pub fn turns(&self) -> bool {
        self.shown(ShowKind::Turns, false)
    }
    /// The context meter on model message headers.
    pub fn context_meter(&self) -> bool {
        self.shown(ShowKind::Context, false)
    }
    /// Spawned sessions at their calls: hidden, a card, or inline.
    pub fn spawns(&self) -> SpawnDisplay {
        match self.asked(ShowKind::Spawns) {
            Some(Asked::Hidden) => SpawnDisplay::Hidden,
            Some(Asked::Level("card")) | None => SpawnDisplay::Card,
            Some(Asked::Level("full")) => SpawnDisplay::Inline { depth: None },
            Some(Asked::Level(level)) => SpawnDisplay::Inline {
                // A depth past u32::MAX reaches every level a session tree has.
                depth: Some(
                    Self::number(level).map_or(u32::MAX, |n| u32::try_from(n).unwrap_or(u32::MAX)),
                ),
            },
        }
    }

    /// Refuse a show or hide word outside the table, with the table's words.
    fn check_words(&self) -> Result<(), ReadError> {
        let unknown = |word: &str| ReadError::UnknownShowWord {
            word: word.into(),
            kinds: ShowKind::ALL
                .iter()
                .map(|kind| {
                    format!(
                        "{}:{}{}",
                        kind.name(),
                        kind.levels().join("|"),
                        if kind.takes_number() { "|N" } else { "" }
                    )
                })
                .collect::<Vec<_>>()
                .join(", "),
        };
        for word in &self.show {
            let (name, level) = word
                .split_once(':')
                .map_or((word.as_str(), None), |(name, level)| (name, Some(level)));
            let kind = ShowKind::parse(name).ok_or_else(|| unknown(word))?;
            if let Some(level) = level
                && !kind.levels().contains(&level)
                && !(kind.takes_number() && Self::number(level).is_some())
            {
                return Err(unknown(word));
            }
        }
        for word in &self.hide {
            ShowKind::parse(word).ok_or_else(|| unknown(word))?;
        }
        Ok(())
    }

    /// Whether any show or hide word names `kind`.
    fn names(&self, kind: ShowKind) -> bool {
        self.asked(kind).is_some()
    }

    /// Visibility of an available original reference-text body. The caller
    /// decides whether the body is available.
    pub fn text_kind_visible(&self, kind: BlockKind) -> bool {
        self.view.block_kinds().contains(&kind)
            && match kind {
                BlockKind::Text => true,
                BlockKind::Thinking => self.thinking() == Expansion::Full,
                BlockKind::ToolResult => self.tools() == Expansion::Full,
                _ => false,
            }
    }

    pub fn annotation_position(&self) -> Option<&crate::ChangePosition> {
        match &self.at {
            Some(ReadAt::Position(position)) => Some(position),
            _ => None,
        }
    }

    pub fn annotation_at(&self) -> Option<AnnotationAt> {
        match self.at.as_ref()? {
            ReadAt::Position(position) => Some(AnnotationAt::Position(position.archive)),
            ReadAt::Time(at) => Some(AnnotationAt::Time(*at)),
            ReadAt::TimeExpression(_) => None,
        }
    }

    pub fn validate(&self) -> Result<(), ReadError> {
        self.check_words()?;
        let prose = matches!(self.view, View::Transcript | View::Context);
        for (supplied, option, accepted) in [
            (
                self.names(ShowKind::Tools) && !prose,
                "tools",
                "transcript, context",
            ),
            (
                self.names(ShowKind::Thinking) && !prose,
                "thinking",
                "transcript, context",
            ),
            (
                self.names(ShowKind::Harness) && !prose,
                "harness",
                "transcript, context",
            ),
            (self.turns() && !prose, "turns", "transcript, context"),
            (
                self.inherited()
                    && matches!(
                        self.view,
                        View::Overview | View::Tree | View::Live | View::Raw | View::Trace
                    ),
                "inherited",
                "transcript, dialogue, context, changes",
            ),
            (
                self.branches()
                    && matches!(
                        self.view,
                        View::Overview | View::Tree | View::Live | View::Trace
                    ),
                "branches",
                "transcript, dialogue, context, changes, raw",
            ),
            (
                matches!(self.spawns(), SpawnDisplay::Inline { .. }) && self.view == View::Context,
                "spawns",
                "transcript, dialogue, changes, raw",
            ),
            (
                self.context_meter()
                    && matches!(
                        self.view,
                        View::Overview | View::Tree | View::Live | View::Trace
                    ),
                "context_meter",
                "transcript, dialogue, context, changes, raw",
            ),
            (
                self.diff.is_some() && self.view != View::Context,
                "diff",
                "context",
            ),
        ] {
            if supplied {
                return Err(ReadError::UnsupportedOption {
                    view: self.view,
                    option: option.into(),
                    accepted_by: accepted.into(),
                });
            }
        }
        Ok(())
    }
}

/// A read step's token choice. Absence selects the default; JSON null lifts it.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(untagged)]
#[schemars(extend("cli_null" = "none"))]
pub enum ReadBudget {
    Tokens(u64),
    Unlimited,
    #[default]
    #[serde(skip)]
    Default,
}
impl ReadBudget {
    pub fn is_default(&self) -> bool {
        matches!(self, Self::Default)
    }
}

/// Logical delivery caps captured before per-page response framing is reserved.
/// A continuation reuses all caps; an explicit override changes only that page.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct PageBudget {
    pub items: std::num::NonZeroUsize,
    pub tokens: std::num::NonZeroU64,
    pub bytes: std::num::NonZeroUsize,
}
impl PageBudget {
    /// Resolve only the token cap. Every page still has its item/byte bounds
    /// and must make minimum progress, including an explicit zero-token read.
    pub fn with_tokens(mut self, choice: ReadBudget) -> Self {
        self.tokens = match choice {
            ReadBudget::Default => self.tokens,
            ReadBudget::Tokens(tokens) => {
                std::num::NonZeroU64::new(tokens).unwrap_or(std::num::NonZeroU64::MIN)
            }
            // The fixed byte bound always ends a page before this cap binds.
            ReadBudget::Unlimited => std::num::NonZeroU64::MAX,
        };
        self
    }
}
impl Default for PageBudget {
    fn default() -> Self {
        Self {
            items: std::num::NonZeroUsize::new(50).unwrap(),
            tokens: std::num::NonZeroU64::new(4000).unwrap(),
            bytes: std::num::NonZeroUsize::new(21_000).unwrap(),
        }
    }
}
