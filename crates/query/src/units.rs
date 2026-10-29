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
    Notes,
    Links,
    Turns,
    Deliveries,
    Work,
    Changes,
}

impl Unit {
    /// Every unit, in the registry's order.
    pub fn all() -> impl Iterator<Item = Unit> {
        crate::units::UNITS.iter().map(|s| s.unit)
    }

    pub fn name(self) -> &'static str {
        crate::units::spec(self).name
    }

    /// The singular form `exists` and `seq` use.
    pub fn singular(self) -> &'static str {
        crate::units::spec(self).singular
    }

    pub fn from_name(name: &str) -> Option<Unit> {
        let name = name.to_ascii_lowercase();
        Unit::all().find(|u| u.name() == name || u.singular() == name)
    }
}

use crate::ColKind;
use ColKind::*;
pub struct ColumnSpec {
    pub name: &'static str,
    pub kind: ColKind,
}
pub struct UnitSpec {
    pub unit: Unit,
    pub name: &'static str,
    pub singular: &'static str,
    pub in_session: bool,
    pub identity: &'static [&'static str],
    pub main_time: Option<&'static str>,
    pub columns: &'static [ColumnSpec],
    pub relations: &'static [Relation],
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
        singular: "session",
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
        relations: &[],
    },
    UnitSpec {
        unit: Unit::Documents,
        name: "documents",
        singular: "document",
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
        relations: &[],
    },
    UnitSpec {
        unit: Unit::Messages,
        name: "messages",
        singular: "message",
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
        relations: &[Relation {
            prefix: "session",
            target: Unit::Sessions,
        }],
    },
    UnitSpec {
        unit: Unit::Blocks,
        name: "blocks",
        singular: "block",
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
        singular: "tool_call",
        in_session: true,
        identity: &["message"],
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
        ],
    },
    UnitSpec {
        unit: Unit::Files,
        name: "files",
        singular: "file",
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
        relations: &[Relation {
            prefix: "session",
            target: Unit::Sessions,
        }],
    },
    UnitSpec {
        unit: Unit::Events,
        name: "events",
        singular: "event",
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
        relations: &[Relation {
            prefix: "session",
            target: Unit::Sessions,
        }],
    },
    UnitSpec {
        unit: Unit::Notes,
        name: "notes",
        singular: "note",
        in_session: false,
        identity: &["id"],
        main_time: Some("created_at"),
        columns: &[
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
                kind: Text,
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
                name: "first_note",
                kind: Text,
            },
            ColumnSpec {
                name: "target",
                kind: Text,
            },
        ],
        relations: &[],
    },
    UnitSpec {
        unit: Unit::Links,
        name: "links",
        singular: "link",
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
        singular: "turn",
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
                kind: Number,
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
                kind: Number,
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
                kind: Number,
            },
            ColumnSpec {
                name: "abandoned",
                kind: Number,
            },
        ],
        relations: &[Relation {
            prefix: "session",
            target: Unit::Sessions,
        }],
    },
    UnitSpec {
        unit: Unit::Deliveries,
        name: "deliveries",
        singular: "delivery",
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
        unit: Unit::Work,
        name: "work",
        singular: "work",
        in_session: true,
        identity: &[],
        main_time: Some("at"),
        columns: &[
            ColumnSpec {
                name: "kind",
                kind: Text,
            },
            ColumnSpec {
                name: "title",
                kind: Text,
            },
            ColumnSpec {
                name: "state",
                kind: Text,
            },
            ColumnSpec {
                name: "session",
                kind: Text,
            },
            ColumnSpec {
                name: "at",
                kind: Time,
            },
            ColumnSpec {
                name: "goal",
                kind: Text,
            },
        ],
        relations: &[Relation {
            prefix: "session",
            target: Unit::Sessions,
        }],
    },
    UnitSpec {
        unit: Unit::Changes,
        name: "changes",
        singular: "change",
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
                name: "note_kind",
                kind: Text,
            },
        ],
        relations: &[],
    },
];
