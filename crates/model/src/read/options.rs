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
}
impl View {
    pub const ALL: [Self; 8] = [
        Self::Transcript,
        Self::Dialogue,
        Self::Summary,
        Self::Context,
        Self::Links,
        Self::Changes,
        Self::Live,
        Self::Raw,
    ];
    pub const NAMES: &[&str] = &[
        "transcript",
        "dialogue",
        "summary",
        "context",
        "links",
        "changes",
        "live",
        "raw",
    ];
    pub const fn name(self) -> &'static str {
        match self {
            Self::Transcript => "transcript",
            Self::Dialogue => "dialogue",
            Self::Summary => "summary",
            Self::Context => "context",
            Self::Links => "links",
            Self::Changes => "changes",
            Self::Live => "live",
            Self::Raw => "raw",
        }
    }
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|view| view.name() == name)
    }
    /// The metadata filter and layout consult this same rule.
    pub const fn block_kinds(self) -> &'static [BlockKind] {
        use BlockKind::*;
        match self {
            Self::Transcript | Self::Context | Self::Raw => {
                &[Text, Thinking, ToolCall, ToolResult, Image, File, Other]
            }
            Self::Dialogue => &[Text],
            Self::Changes => &[ToolCall, ToolResult],
            Self::Summary | Self::Links | Self::Live => &[],
        }
    }
    pub fn keeps_message(self, message: &Message) -> bool {
        self.keeps_message_metadata(message.role, message.prompt_kind)
    }
    pub fn keeps_message_metadata(self, role: Role, prompt_kind: Option<PromptKind>) -> bool {
        match self {
            Self::Dialogue => role != Role::System && prompt_kind != Some(PromptKind::Harness),
            Self::Transcript | Self::Context | Self::Changes | Self::Raw => true,
            Self::Summary | Self::Links | Self::Live => false,
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

/// One row per fold kind: the variant, its word and its levels, fullest
/// last. The table gives the enum, its words and the `--show` help text, so
/// the CLI, MCP and the web's fold keys cannot drift from it.
macro_rules! fold_table {
    ($($kind:ident $name:literal [$($level:literal),+] $depth:tt;)+) => {
        /// The closed set of item kinds a reading folds, one table for the
        /// CLI's `--show`/`--hide`, MCP's `show`/`hide` and the web's fold keys.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum FoldKind {
            $($kind,)+
        }
        impl FoldKind {
            pub const ALL: [Self; [$(stringify!($kind)),+].len()] = [$(Self::$kind,)+];
            /// The `--show` help: every kind with its levels.
            pub const HELP: &'static str = concat!(
                "Item kinds to show, each with an optional level (the fullest when omitted):"
                $(, " ", $name, "[:", fold_table!(@levels $($level),+), fold_table!(@depth $depth), "]")+
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
            /// Whether the kind also takes a depth `N` (1 or more) as its level.
            pub const fn takes_depth(self) -> bool {
                match self {
                    $(Self::$kind => $depth,)+
                }
            }
        }
    };
    (@levels $first:literal $(, $rest:literal)*) => {
        concat!($first $(, "|", $rest)*)
    };
    (@depth true) => {
        "|N"
    };
    (@depth false) => {
        ""
    };
}
fold_table! {
    Tools "tools" ["line", "full"] false;
    Thinking "thinking" ["line", "full"] false;
    Harness "harness" ["line", "full"] false;
    Inherited "inherited" ["full"] false;
    Branches "branches" ["full"] false;
    Spawns "spawns" ["card", "full"] true;
    Assertions "assertions" ["header"] false;
    Turns "turns" ["dividers"] false;
    Context "context" ["meter"] false;
}
impl FoldKind {
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
    pub fn apply_fold(&mut self, show: &[String], hide: &[String]) -> Result<(), ReadError> {
        let unknown = |word: &str| ReadError::UnknownFold {
            word: word.into(),
            kinds: FoldKind::ALL
                .iter()
                .map(|kind| {
                    format!(
                        "{}:{}{}",
                        kind.name(),
                        kind.levels().join("|"),
                        if kind.takes_depth() { "|N" } else { "" }
                    )
                })
                .collect::<Vec<_>>()
                .join(", "),
        };
        for word in show {
            let (name, level) = word
                .split_once(':')
                .map_or((word.as_str(), None), |(n, l)| (n, Some(l)));
            let kind = FoldKind::parse(name).ok_or_else(|| unknown(word))?;
            let level = level.unwrap_or(kind.levels()[kind.levels().len() - 1]);
            let depth = kind
                .takes_depth()
                .then(|| level.parse::<u32>().ok().filter(|depth| *depth > 0))
                .flatten();
            if depth.is_none() && !kind.levels().contains(&level) {
                return Err(unknown(word));
            }
            let expansion = if level == "full" {
                Expansion::Full
            } else {
                Expansion::Collapsed
            };
            match kind {
                FoldKind::Tools => self.tools = Some(expansion),
                FoldKind::Thinking => self.thinking = Some(expansion),
                FoldKind::Harness => self.harness = Some(expansion),
                FoldKind::Inherited => self.inherited = true,
                FoldKind::Branches => self.branches = true,
                FoldKind::Spawns => {
                    self.spawns = match (level, depth) {
                        ("card", _) => SpawnFold::Card,
                        (_, Some(depth)) => SpawnFold::Inline { depth: Some(depth) },
                        _ => SpawnFold::Inline { depth: None },
                    }
                }
                FoldKind::Assertions => self.assertions = true,
                FoldKind::Turns => self.turns = true,
                FoldKind::Context => self.context_meter = true,
            }
        }
        for word in hide {
            let kind = FoldKind::parse(word).ok_or_else(|| unknown(word))?;
            match kind {
                FoldKind::Tools => self.tools = Some(Expansion::None),
                FoldKind::Thinking => self.thinking = Some(Expansion::None),
                FoldKind::Harness => self.harness = Some(Expansion::None),
                FoldKind::Inherited => self.inherited = false,
                FoldKind::Branches => self.branches = false,
                FoldKind::Spawns => self.spawns = SpawnFold::Hidden,
                FoldKind::Assertions => self.assertions = false,
                FoldKind::Turns => self.turns = false,
                FoldKind::Context => self.context_meter = false,
            }
        }
        Ok(())
    }

    /// The `show` and `hide` words that give these options from the
    /// defaults: the inverse of [`Self::apply_fold`].
    pub fn fold_words(&self) -> (Vec<String>, Vec<String>) {
        let mut show = Vec::new();
        let mut hide = Vec::new();
        for (name, level) in [
            ("tools", self.tools),
            ("thinking", self.thinking),
            ("harness", self.harness),
        ] {
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
            ("assertions", self.assertions, true),
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
    /// Visibility of an available original reference-text body. Availability
    /// itself remains evidence owned by the caller.
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
                        View::Summary | View::Links | View::Live | View::Raw
                    ),
                "inherited",
                "transcript, dialogue, context, changes",
            ),
            (
                self.branches && matches!(self.view, View::Summary | View::Links | View::Live),
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
