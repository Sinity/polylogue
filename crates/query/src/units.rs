//! Pure query units and relations. Store owns their row lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    Sessions,
    Documents,
    Messages,
    Blocks,
    ToolCalls,
    Files,
    Events,
    Assertions,
    Links,
    Turns,
    Deliveries,
    Changes,
    LiveIntervals,
    Processes,
}

impl Unit {
    /// Every unit, in the registry's order.
    pub fn all() -> impl Iterator<Item = Unit> {
        crate::units::UNITS.iter().map(|s| s.unit)
    }

    pub fn name(self) -> &'static str {
        crate::units::spec(self).name
    }

    pub fn from_name(name: &str) -> Option<Unit> {
        let name = name.to_ascii_lowercase();
        Unit::all().find(|u| u.name() == name)
    }
}

use crate::ColKind;
use crate::search::SnippetIndex;
use ColKind::*;
pub struct ColumnSpec {
    pub name: &'static str,
    pub kind: ColKind,
}
pub struct UnitSpec {
    pub unit: Unit,
    pub name: &'static str,
    pub in_session: bool,
    pub identity: &'static [&'static str],
    pub main_time: Option<&'static str>,
    pub columns: &'static [ColumnSpec],
    /// How search words find its rows (docs/search.md: Terms).
    pub words: Words,
    pub relations: &'static [Relation],
}
/// How search words find a unit's rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Words {
    /// Words do not search it.
    None,
    /// Hits in the full-text indexes: plain words search it when `plain`
    /// (otherwise only `output:` and `thinking:` do), and the hits of the
    /// `ranks` indexes rank its rows by relevance.
    Index {
        plain: bool,
        ranks: &'static [SnippetIndex],
    },
    /// Each word is a substring of a stored value; it filters and does not
    /// rank.
    Substring,
}
impl Words {
    /// Plain words search the unit.
    pub fn plain(self) -> bool {
        match self {
            Words::Index { plain, .. } => plain,
            Words::Substring => true,
            Words::None => false,
        }
    }
    /// The hits of `index` rank the unit's rows.
    pub fn ranks(self, index: SnippetIndex) -> bool {
        matches!(self, Words::Index { ranks, .. } if ranks.contains(&index))
    }
}
pub struct Relation {
    pub prefix: &'static str,
    pub target: Unit,
}
impl UnitSpec {
    pub fn relation(&self, prefix: &str) -> Option<&Relation> {
        self.relations.iter().find(|r| r.prefix == prefix)
    }
}
/// A unit the design declares whose rows have no producer yet. The catalog
/// accepts it with its designed fields, so a query over it answers
/// `not_built` instead of "not a unit" or an empty result.
pub struct DesignedUnit {
    pub name: &'static str,
    pub columns: &'static [ColumnSpec],
}
impl DesignedUnit {
    pub fn named(name: &str) -> Option<&'static DesignedUnit> {
        let name = name.strip_prefix("pl.").unwrap_or(name);
        DESIGNED_UNITS
            .iter()
            .find(|unit| unit.name.eq_ignore_ascii_case(name))
    }
}
const fn column(name: &'static str, kind: ColKind) -> ColumnSpec {
    ColumnSpec { name, kind }
}
pub static DESIGNED_UNITS: &[DesignedUnit] = &[
    DesignedUnit {
        name: "pastes",
        columns: &[
            column("hash", Text),
            column("source", Text),
            column("state", Text),
            column("message", Text),
            column("session", Text),
        ],
    },
    DesignedUnit {
        name: "variants",
        columns: &[
            column("id", Text),
            column("root", Text),
            column("base", Text),
            column("source", Text),
            column("kind", Text),
            column("from_lang", Text),
            column("to_lang", Text),
            column("author", Text),
            column("via", Text),
            column("method", Text),
            column("created_at", Time),
        ],
    },
    DesignedUnit {
        name: "attachments",
        columns: &[
            column("message", Text),
            column("block", Text),
            column("position", Number),
            column("supplied_by", Text),
            column("name", Text),
            column("media_type", Text),
            column("size", Number),
            column("availability", Text),
            column("blob", Text),
            column("source_ref", Text),
            column("session", Text),
        ],
    },
];
pub fn spec(unit: Unit) -> &'static UnitSpec {
    UNITS
        .iter()
        .find(|s| s.unit == unit)
        .expect("declared unit")
}
pub static UNITS: &[UnitSpec] = &[
    UnitSpec {
        unit: Unit::Sessions,
        name: "sessions",
        in_session: false,
        identity: &["id"],
        main_time: Some("last_active"),
        columns: &[
            ColumnSpec {
                name: "id",
                kind: Text,
            },
            ColumnSpec {
                name: "origin",
                kind: Text,
            },
            ColumnSpec {
                name: "kind",
                kind: Text,
            },
            ColumnSpec {
                name: "title",
                kind: Text,
            },
            ColumnSpec {
                name: "started",
                kind: Time,
            },
            ColumnSpec {
                name: "last_active",
                kind: Time,
            },
            ColumnSpec {
                name: "cwd",
                kind: Text,
            },
            ColumnSpec {
                name: "repo",
                kind: Text,
            },
        ],
        words: Words::Index {
            plain: true,
            ranks: &[SnippetIndex::Prose, SnippetIndex::Output],
        },
        relations: &[],
    },
    UnitSpec {
        unit: Unit::Documents,
        name: "documents",
        in_session: false,
        identity: &["id"],
        main_time: Some("source_time"),
        columns: &[
            ColumnSpec {
                name: "id",
                kind: Text,
            },
            ColumnSpec {
                name: "origin",
                kind: Text,
            },
            ColumnSpec {
                name: "purpose",
                kind: Text,
            },
            ColumnSpec {
                name: "title",
                kind: Text,
            },
            ColumnSpec {
                name: "source_time",
                kind: Time,
            },
            ColumnSpec {
                name: "authorship",
                kind: Text,
            },
        ],
        words: Words::Index {
            plain: true,
            ranks: &[SnippetIndex::Prose],
        },
        relations: &[],
    },
    UnitSpec {
        unit: Unit::Messages,
        name: "messages",
        in_session: true,
        identity: &["id"],
        main_time: Some("at"),
        columns: &[
            ColumnSpec {
                name: "id",
                kind: Text,
            },
            ColumnSpec {
                name: "session",
                kind: Text,
            },
            ColumnSpec {
                name: "role",
                kind: Text,
            },
            ColumnSpec {
                name: "written_by",
                kind: Text,
            },
            ColumnSpec {
                name: "at",
                kind: Time,
            },
            ColumnSpec {
                name: "model",
                kind: Text,
            },
        ],
        words: Words::Index {
            plain: true,
            ranks: &[SnippetIndex::Prose, SnippetIndex::Output],
        },
        relations: &[Relation {
            prefix: "session",
            target: Unit::Sessions,
        }],
    },
    UnitSpec {
        unit: Unit::Blocks,
        name: "blocks",
        in_session: true,
        identity: &["message", "idx"],
        main_time: Some("at"),
        columns: &[
            ColumnSpec {
                name: "session",
                kind: Text,
            },
            ColumnSpec {
                name: "message",
                kind: Text,
            },
            ColumnSpec {
                name: "kind",
                kind: Text,
            },
            ColumnSpec {
                name: "idx",
                kind: Number,
            },
            ColumnSpec {
                name: "at",
                kind: Time,
            },
        ],
        words: Words::Index {
            plain: true,
            ranks: &[SnippetIndex::Prose, SnippetIndex::Output],
        },
        relations: &[
            Relation {
                prefix: "session",
                target: Unit::Sessions,
            },
            Relation {
                prefix: "message",
                target: Unit::Messages,
            },
            Relation {
                prefix: "call",
                target: Unit::ToolCalls,
            },
        ],
    },
    UnitSpec {
        unit: Unit::ToolCalls,
        name: "tool_calls",
        in_session: true,
        identity: &["message", "idx"],
        main_time: Some("at"),
        columns: &[
            ColumnSpec {
                name: "session",
                kind: Text,
            },
            ColumnSpec {
                name: "message",
                kind: Text,
            },
            ColumnSpec {
                name: "tool",
                kind: Text,
            },
            ColumnSpec {
                name: "action",
                kind: Text,
            },
            ColumnSpec {
                name: "command",
                kind: Text,
            },
            ColumnSpec {
                name: "path",
                kind: Text,
            },
            ColumnSpec {
                name: "outcome",
                kind: Text,
            },
            ColumnSpec {
                name: "exit_code",
                kind: Number,
            },
            ColumnSpec {
                name: "at",
                kind: Time,
            },
        ],
        words: Words::Index {
            plain: false,
            ranks: &[SnippetIndex::Output],
        },
        relations: &[
            Relation {
                prefix: "session",
                target: Unit::Sessions,
            },
            Relation {
                prefix: "message",
                target: Unit::Messages,
            },
            Relation {
                prefix: "child",
                target: Unit::Sessions,
            },
            Relation {
                prefix: "next",
                target: Unit::ToolCalls,
            },
            Relation {
                prefix: "previous",
                target: Unit::ToolCalls,
            },
        ],
    },
    UnitSpec {
        unit: Unit::Files,
        name: "files",
        in_session: true,
        identity: &["session", "path"],
        main_time: Some("first_at"),
        columns: &[
            ColumnSpec {
                name: "session",
                kind: Text,
            },
            ColumnSpec {
                name: "path",
                kind: Text,
            },
            ColumnSpec {
                name: "reads",
                kind: Number,
            },
            ColumnSpec {
                name: "writes",
                kind: Number,
            },
            ColumnSpec {
                name: "edits",
                kind: Number,
            },
            ColumnSpec {
                name: "first_at",
                kind: Time,
            },
            ColumnSpec {
                name: "last_at",
                kind: Time,
            },
        ],
        words: Words::None,
        relations: &[Relation {
            prefix: "session",
            target: Unit::Sessions,
        }],
    },
    UnitSpec {
        unit: Unit::Events,
        name: "events",
        in_session: true,
        identity: &["id"],
        main_time: Some("at"),
        columns: &[
            ColumnSpec {
                name: "id",
                kind: Text,
            },
            ColumnSpec {
                name: "session",
                kind: Text,
            },
            ColumnSpec {
                name: "input",
                kind: Text,
            },
            ColumnSpec {
                name: "source",
                kind: Text,
            },
            ColumnSpec {
                name: "kind",
                kind: Text,
            },
            ColumnSpec {
                name: "at",
                kind: Time,
            },
            ColumnSpec {
                name: "data",
                kind: ColKind::Json,
            },
        ],
        words: Words::None,
        relations: &[Relation {
            prefix: "session",
            target: Unit::Sessions,
        }],
    },
    UnitSpec {
        unit: Unit::Assertions,
        name: "assertions",
        in_session: false,
        identity: &["id", "root"],
        main_time: Some("created_at"),
        columns: &[
            ColumnSpec {
                name: "reference",
                kind: Text,
            },
            ColumnSpec {
                name: "id",
                kind: Text,
            },
            ColumnSpec {
                name: "kind",
                kind: Text,
            },
            ColumnSpec {
                name: "key",
                kind: Text,
            },
            ColumnSpec {
                name: "value",
                kind: Json,
            },
            ColumnSpec {
                name: "author",
                kind: Text,
            },
            ColumnSpec {
                name: "created_at",
                kind: Time,
            },
            ColumnSpec {
                name: "root",
                kind: Text,
            },
            ColumnSpec {
                name: "about",
                kind: Text,
            },
        ],
        words: Words::Substring,
        relations: &[],
    },
    UnitSpec {
        unit: Unit::Links,
        name: "links",
        in_session: false,
        identity: &["parent", "child", "kind", "how"],
        main_time: None,
        columns: &[
            ColumnSpec {
                name: "parent",
                kind: Text,
            },
            ColumnSpec {
                name: "child",
                kind: Text,
            },
            ColumnSpec {
                name: "kind",
                kind: Text,
            },
            ColumnSpec {
                name: "how",
                kind: Text,
            },
            ColumnSpec {
                name: "evidence",
                kind: Text,
            },
            ColumnSpec {
                name: "pending",
                kind: Bool,
            },
        ],
        words: Words::None,
        relations: &[
            Relation {
                prefix: "parent",
                target: Unit::Sessions,
            },
            Relation {
                prefix: "child",
                target: Unit::Sessions,
            },
        ],
    },
    UnitSpec {
        unit: Unit::Turns,
        name: "turns",
        in_session: true,
        identity: &["id"],
        main_time: Some("start"),
        columns: &[
            ColumnSpec {
                name: "id",
                kind: Text,
            },
            ColumnSpec {
                name: "session",
                kind: Text,
            },
            ColumnSpec {
                name: "start",
                kind: Time,
            },
            ColumnSpec {
                name: "end",
                kind: Time,
            },
            ColumnSpec {
                name: "burst",
                kind: Number,
            },
            ColumnSpec {
                name: "prompts",
                kind: Text,
            },
            ColumnSpec {
                name: "slash_command",
                kind: Bool,
            },
            ColumnSpec {
                name: "answer",
                kind: Text,
            },
            ColumnSpec {
                name: "attempts",
                kind: Number,
            },
            ColumnSpec {
                name: "path",
                kind: Text,
            },
            ColumnSpec {
                name: "latency",
                kind: Duration,
            },
            ColumnSpec {
                name: "tool_calls",
                kind: Number,
            },
            ColumnSpec {
                name: "errors",
                kind: Number,
            },
            ColumnSpec {
                name: "input_tokens",
                kind: Number,
            },
            ColumnSpec {
                name: "output_tokens",
                kind: Number,
            },
            ColumnSpec {
                name: "cost",
                kind: Money,
            },
            ColumnSpec {
                name: "abandoned",
                kind: Bool,
            },
        ],
        words: Words::None,
        relations: &[Relation {
            prefix: "session",
            target: Unit::Sessions,
        }],
    },
    UnitSpec {
        unit: Unit::Deliveries,
        name: "deliveries",
        in_session: true,
        identity: &["id"],
        main_time: Some("at"),
        columns: &[
            ColumnSpec {
                name: "id",
                kind: Text,
            },
            ColumnSpec {
                name: "recipient",
                kind: Text,
            },
            ColumnSpec {
                name: "event",
                kind: Text,
            },
            ColumnSpec {
                name: "outcome",
                kind: Text,
            },
            ColumnSpec {
                name: "at",
                kind: Time,
            },
            ColumnSpec {
                name: "bytes",
                kind: Number,
            },
        ],
        words: Words::None,
        relations: &[
            Relation {
                prefix: "session",
                target: Unit::Sessions,
            },
            Relation {
                prefix: "recipient",
                target: Unit::Sessions,
            },
        ],
    },
    UnitSpec {
        unit: Unit::LiveIntervals,
        name: "live_intervals",
        in_session: true,
        identity: &["session", "aspect", "start"],
        main_time: Some("start"),
        columns: &[
            ColumnSpec {
                name: "session",
                kind: Text,
            },
            ColumnSpec {
                name: "aspect",
                kind: Text,
            },
            ColumnSpec {
                name: "value",
                kind: Text,
            },
            ColumnSpec {
                name: "start",
                kind: Time,
            },
            ColumnSpec {
                name: "end",
                kind: Time,
            },
            ColumnSpec {
                name: "duration_ms",
                kind: Number,
            },
            ColumnSpec {
                name: "closed_by",
                kind: Text,
            },
        ],
        words: Words::None,
        relations: &[Relation {
            prefix: "session",
            target: Unit::Sessions,
        }],
    },
    UnitSpec {
        unit: Unit::Processes,
        name: "processes",
        in_session: true,
        identity: &["session", "boot_id", "pid", "start"],
        main_time: Some("started_at"),
        columns: &[
            ColumnSpec {
                name: "session",
                kind: Text,
            },
            ColumnSpec {
                name: "boot_id",
                kind: Text,
            },
            ColumnSpec {
                name: "pid",
                kind: Number,
            },
            ColumnSpec {
                name: "start",
                kind: Number,
            },
            ColumnSpec {
                name: "parent_pid",
                kind: Number,
            },
            ColumnSpec {
                name: "comm",
                kind: Text,
            },
            ColumnSpec {
                name: "argv",
                kind: Text,
            },
            ColumnSpec {
                name: "cwd",
                kind: Text,
            },
            ColumnSpec {
                name: "started_at",
                kind: Time,
            },
            ColumnSpec {
                name: "ended_at",
                kind: Time,
            },
            ColumnSpec {
                name: "role",
                kind: Text,
            },
            ColumnSpec {
                name: "tool_call",
                kind: Text,
            },
        ],
        words: Words::None,
        relations: &[Relation {
            prefix: "session",
            target: Unit::Sessions,
        }],
    },
    UnitSpec {
        unit: Unit::Changes,
        name: "changes",
        in_session: false,
        identity: &["kind", "ref"],
        main_time: Some("seq"),
        columns: &[
            ColumnSpec {
                name: "kind",
                kind: Text,
            },
            ColumnSpec {
                name: "ref",
                kind: Text,
            },
            ColumnSpec {
                name: "seq",
                kind: Number,
            },
            ColumnSpec {
                name: "live_changed",
                kind: Number,
            },
            ColumnSpec {
                name: "assertion_kind",
                kind: Text,
            },
        ],
        words: Words::None,
        relations: &[],
    },
];
