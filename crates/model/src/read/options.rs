use super::ReadError;
use crate::{Block, BlockKind, Message, PromptKind, Role};
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum View {
    #[default]
    Transcript,
    Dialogue,
    Summary,
    Context,
    Links,
    Changes,
    Live,
    Raw,
    Trace,
}
stored_names!(View, name, parse {
    Transcript => "transcript",
    Dialogue => "dialogue",
    Summary => "summary",
    Context => "context",
    Links => "links",
    Changes => "changes",
    Live => "live",
    Raw => "raw",
    Trace => "trace",
});
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
            Self::Summary | Self::Links | Self::Live | Self::Trace => &[],
        }
    }
    pub fn keeps_message(self, message: &Message) -> bool {
        self.keeps_message_metadata(message.role, message.prompt_kind)
    }
    pub fn keeps_message_metadata(self, role: Role, prompt_kind: Option<PromptKind>) -> bool {
        match self {
            Self::Dialogue => {
                role != Role::System && !prompt_kind.is_some_and(PromptKind::is_harness)
            }
            Self::Transcript | Self::Context | Self::Changes | Self::Raw => true,
            Self::Summary | Self::Links | Self::Live | Self::Trace => false,
        }
    }
    pub fn keeps(self, message: &Message, block: &Block) -> bool {
        self.keeps_message(message) && self.block_kinds().contains(&block.kind())
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
    pub assertions: bool,
    #[serde(default)]
    pub for_others: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variants: Vec<String>,
    /// Spawned sessions at their calls: hidden, a card, or inline.
    #[serde(default = "spawn_cards")]
    pub spawns: SpawnFold,
    /// The context meter on model message headers.
    #[serde(default)]
    pub context_meter: bool,
    /// For `root:REF`: read only the named physical session.
    #[serde(default)]
    pub physical: bool,
    /// The message a context reading is taken at; the latest when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<crate::MessageId>,
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
fn spawn_cards() -> SpawnFold {
    SpawnFold::Card
}
impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            view: View::Transcript,
            tools: None,
            thinking: None,
            harness: None,
            output_lines: 100,
            inherited: false,
            branches: false,
            turns: false,
            assertions: true,
            for_others: false,
            variants: vec![],
            spawns: SpawnFold::Card,
            physical: false,
            context_meter: false,
            at: None,
            diff: None,
            around: None,
            latest: false,
        }
    }
}

/// One row per show kind: the variant, its word and its levels, fullest
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
pub enum SpawnFold {
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
                        ("card", _) => SpawnFold::Card,
                        (_, Some(depth)) => SpawnFold::Inline {
                            depth: Some(u32::try_from(depth).map_err(|_| unknown(word))?),
                        },
                        _ => SpawnFold::Inline { depth: None },
                    }
                }
                ShowKind::Annotations => self.assertions = true,
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
                ShowKind::Spawns => self.spawns = SpawnFold::Hidden,
                ShowKind::Annotations => self.assertions = false,
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
        match self.tools {
            Some(Expansion::Full) if self.output_lines == 0 => show.push("tools:full".into()),
            Some(Expansion::Full) => show.push(format!("tools:{}", self.output_lines)),
            Some(Expansion::Collapsed) => show.push("tools:line".into()),
            Some(Expansion::None) => hide.push("tools".into()),
            None => {}
        }
        for (name, level) in [("thinking", self.thinking), ("harness", self.harness)] {
            match level {
                Some(Expansion::Full) => show.push(format!("{name}:full")),
                Some(Expansion::Collapsed) => show.push(format!("{name}:line")),
                Some(Expansion::None) => hide.push(name.to_owned()),
                None => {}
            }
        }
        for (name, value, default) in [
            ("inherited", self.inherited, false),
            ("branches", self.branches, false),
            (ShowKind::Annotations.name(), self.assertions, true),
            ("turns", self.turns, false),
            ("context", self.context_meter, false),
        ] {
            if value != default {
                if value {
                    show.push(name.to_owned());
                } else {
                    hide.push(name.to_owned());
                }
            }
        }
        match self.spawns {
            SpawnFold::Card => {}
            SpawnFold::Hidden => hide.push("spawns".into()),
            SpawnFold::Inline { depth: None } => show.push("spawns:full".into()),
            SpawnFold::Inline { depth: Some(depth) } => show.push(format!("spawns:{depth}")),
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
                self.output_lines != 100 && !prose,
                "output_lines",
                "transcript, context",
            ),
            (
                self.inherited
                    && matches!(
                        self.view,
                        View::Summary | View::Links | View::Live | View::Raw | View::Trace
                    ),
                "inherited",
                "transcript, dialogue, context, changes",
            ),
            (
                self.branches
                    && matches!(
                        self.view,
                        View::Summary | View::Links | View::Live | View::Trace
                    ),
                "branches",
                "transcript, dialogue, context, changes, raw",
            ),
            (
                matches!(self.spawns, SpawnFold::Inline { .. }) && self.view == View::Context,
                "spawns",
                "transcript, dialogue, changes, raw",
            ),
            (
                self.at.is_some() && self.view != View::Context,
                "at",
                "context",
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
impl Default for PageBudget {
    fn default() -> Self {
        Self {
            items: std::num::NonZeroUsize::new(50).unwrap(),
            tokens: std::num::NonZeroU64::new(4000).unwrap(),
            bytes: std::num::NonZeroUsize::new(21_000).unwrap(),
        }
    }
}
