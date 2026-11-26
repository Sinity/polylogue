//! The options of a reading: its view, what it shows and hides, the budgets
//! of a page, and the place a context or history reading is taken at.
use super::ReadError;
use crate::{BlockKind, Message, PromptKind, Role};
use serde::{Deserialize, Serialize};

/// The moment a reading's annotations and pages are taken at: an exact local
/// archive position or a time. The wire form is one string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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

/// Absent expansions distinguish defaults from explicit unsupported options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadOptions {
    #[serde(default)]
    pub view: View,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Expansion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Expansion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness: Option<Expansion>,
    /// Head and tail lines of expanded tool output. Zero keeps the complete output.
    #[serde(default = "output_lines")]
    pub output_lines: u64,
    #[serde(default)]
    pub inherited: bool,
    #[serde(default)]
    pub branches: bool,
    #[serde(default)]
    pub turns: bool,
    #[serde(default = "enabled")]
    pub annotations: bool,
    #[serde(default)]
    pub for_others: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternates: Vec<String>,
    /// Spawned sessions at their calls: hidden, a card, or inline.
    #[serde(default = "spawn_cards")]
    pub spawns: SpawnDisplay,
    /// The context meter on model message headers.
    #[serde(default)]
    pub context_meter: bool,
    /// For `root:REF`: read only the named physical session.
    #[serde(default)]
    pub physical: bool,
    /// The time or exact local position annotations and pages are read at;
    /// now when absent.
    // Polylogue's Sinex kind declarations hash this field's published schema,
    // so its declared type and text change only with a new kind version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(
        with = "Option<crate::MessageId>",
        description = "The message a context reading is taken at; the latest when absent."
    )]
    pub at: Option<ReadAt>,
    /// The other place of a context difference: a message, or `compaction`
    /// or `compaction:N` for just before the last or the Nth-last boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<String>,
    /// For a message read: this many messages before and after it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub around: Option<u64>,
    /// Start at the last page instead of the first.
    #[serde(default)]
    pub latest: bool,
}
fn output_lines() -> u64 {
    100
}
fn enabled() -> bool {
    true
}
fn spawn_cards() -> SpawnDisplay {
    SpawnDisplay::Card
}
impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            view: View::Transcript,
            tools: None,
            thinking: None,
            harness: None,
            output_lines: output_lines(),
            inherited: false,
            branches: false,
            turns: false,
            annotations: enabled(),
            for_others: false,
            alternates: vec![],
            spawns: spawn_cards(),
            physical: false,
            context_meter: false,
            at: None,
            diff: None,
            around: None,
            latest: false,
        }
    }
}

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

impl ReadOptions {
    /// Apply `--show KIND[:LEVEL]` and `--hide KIND` words. A kind or level
    /// outside the table is refused with the table's words.
    pub fn apply_show_hide(&mut self, show: &[String], hide: &[String]) -> Result<(), ReadError> {
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
        for word in show {
            let (name, level) = word
                .split_once(':')
                .map_or((word.as_str(), None), |(n, l)| (n, Some(l)));
            let kind = ShowKind::parse(name).ok_or_else(|| unknown(word))?;
            let level = level.unwrap_or(kind.levels()[kind.levels().len() - 1]);
            let number = kind
                .takes_number()
                .then(|| level.parse::<u64>().ok().filter(|number| *number > 0))
                .flatten();
            if number.is_none() && !kind.levels().contains(&level) {
                return Err(unknown(word));
            }
            let expansion = if level == "full" {
                Expansion::Full
            } else {
                Expansion::Collapsed
            };
            match kind {
                ShowKind::Tools => {
                    self.tools = Some(if number.is_some() {
                        Expansion::Full
                    } else {
                        expansion
                    });
                    self.output_lines = match (level, number) {
                        ("full", _) => 0,
                        (_, Some(lines)) => lines,
                        _ => output_lines(),
                    };
                }
                ShowKind::Thinking => self.thinking = Some(expansion),
                ShowKind::Harness => self.harness = Some(expansion),
                ShowKind::Inherited => self.inherited = true,
                ShowKind::Branches => self.branches = true,
                ShowKind::Spawns => {
                    self.spawns = match (level, number) {
                        ("card", _) => SpawnDisplay::Card,
                        (_, Some(depth)) => SpawnDisplay::Inline {
                            // A depth past u32::MAX reaches every level a session tree has.
                            depth: Some(u32::try_from(depth).unwrap_or(u32::MAX)),
                        },
                        _ => SpawnDisplay::Inline { depth: None },
                    }
                }
                ShowKind::Annotations => self.annotations = true,
                ShowKind::Turns => self.turns = true,
                ShowKind::Context => self.context_meter = true,
            }
        }
        for word in hide {
            let kind = ShowKind::parse(word).ok_or_else(|| unknown(word))?;
            match kind {
                ShowKind::Tools => self.tools = Some(Expansion::None),
                ShowKind::Thinking => self.thinking = Some(Expansion::None),
                ShowKind::Harness => self.harness = Some(Expansion::None),
                ShowKind::Inherited => self.inherited = false,
                ShowKind::Branches => self.branches = false,
                ShowKind::Spawns => self.spawns = SpawnDisplay::Hidden,
                ShowKind::Annotations => self.annotations = false,
                ShowKind::Turns => self.turns = false,
                ShowKind::Context => self.context_meter = false,
            }
        }
        Ok(())
    }

    /// The `show` and `hide` words that give these options from the
    /// defaults: the inverse of [`Self::apply_show_hide`].
    pub fn show_hide_words(&self) -> (Vec<String>, Vec<String>) {
        let mut show = Vec::new();
        let mut hide = Vec::new();
        let tools = ShowKind::Tools.name();
        match self.tools {
            Some(Expansion::Full) if self.output_lines == 0 => show.push(format!("{tools}:full")),
            Some(Expansion::Full) => show.push(format!("{tools}:{}", self.output_lines)),
            Some(Expansion::Collapsed) => show.push(format!("{tools}:line")),
            Some(Expansion::None) => hide.push(tools.into()),
            None => {}
        }
        for (name, level) in [
            (ShowKind::Thinking.name(), self.thinking),
            (ShowKind::Harness.name(), self.harness),
        ] {
            match level {
                Some(Expansion::Full) => show.push(format!("{name}:full")),
                Some(Expansion::Collapsed) => show.push(format!("{name}:line")),
                Some(Expansion::None) => hide.push(name.to_owned()),
                None => {}
            }
        }
        for (name, value, default) in [
            (ShowKind::Inherited.name(), self.inherited, false),
            (ShowKind::Branches.name(), self.branches, false),
            (ShowKind::Annotations.name(), self.annotations, enabled()),
            (ShowKind::Turns.name(), self.turns, false),
            (ShowKind::Context.name(), self.context_meter, false),
        ] {
            if value != default {
                if value {
                    show.push(name.to_owned());
                } else {
                    hide.push(name.to_owned());
                }
            }
        }
        let spawns = ShowKind::Spawns.name();
        match self.spawns {
            SpawnDisplay::Card => {}
            SpawnDisplay::Hidden => hide.push(spawns.into()),
            SpawnDisplay::Inline { depth: None } => show.push(format!("{spawns}:full")),
            // Depth zero places no inline reading: the same as cards, and the
            // table accepts only positive depths.
            SpawnDisplay::Inline { depth: Some(0) } => show.push(format!("{spawns}:card")),
            SpawnDisplay::Inline { depth: Some(depth) } => show.push(format!("{spawns}:{depth}")),
        }
        (show, hide)
    }
}
impl ReadOptions {
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
    pub fn tools(&self) -> Expansion {
        self.tools.unwrap_or(Expansion::Collapsed)
    }
    pub fn thinking(&self) -> Expansion {
        self.thinking.unwrap_or(Expansion::Collapsed)
    }
    pub fn harness(&self) -> Expansion {
        self.harness.unwrap_or(Expansion::Collapsed)
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
            _ => None,
        }
    }

    pub fn validate(&self) -> Result<(), ReadError> {
        let prose = matches!(self.view, View::Transcript | View::Context);
        for (supplied, option, accepted) in [
            (
                self.tools.is_some() && !prose,
                "tools",
                "transcript, context",
            ),
            (
                self.thinking.is_some() && !prose,
                "thinking",
                "transcript, context",
            ),
            (
                self.harness.is_some() && !prose,
                "harness",
                "transcript, context",
            ),
            (self.turns && !prose, "turns", "transcript, context"),
            (
                self.output_lines != output_lines() && !prose,
                "output_lines",
                "transcript, context",
            ),
            (
                self.inherited
                    && matches!(
                        self.view,
                        View::Overview | View::Tree | View::Live | View::Raw | View::Trace
                    ),
                "inherited",
                "transcript, dialogue, context, changes",
            ),
            (
                self.branches
                    && matches!(
                        self.view,
                        View::Overview | View::Tree | View::Live | View::Trace
                    ),
                "branches",
                "transcript, dialogue, context, changes, raw",
            ),
            (
                matches!(self.spawns, SpawnDisplay::Inline { .. }) && self.view == View::Context,
                "spawns",
                "transcript, dialogue, changes, raw",
            ),
            (
                self.context_meter
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
