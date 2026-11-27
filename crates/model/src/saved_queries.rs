//! Built-in questions. An authored definition of the same name takes precedence.

pub struct ShippedQuery {
    pub name: &'static str,
    pub text: &'static str,
    pub help: &'static str,
}

pub const SHIPPED_QUERIES: &[ShippedQuery] = &[
    ShippedQuery {
        name: "running",
        text: "sessions where running:true | sort by live_rank, last_event_at desc",
        help: "Sessions running now",
    },
    ShippedQuery {
        name: "waiting",
        text: "sessions where waiting:true | sort by live_since",
        help: "Sessions waiting for a permission, answer or held turn",
    },
    ShippedQuery {
        name: "recent",
        text: "sessions | sort by last_event_at desc",
        help: "Sessions with the most recent activity first",
    },
    ShippedQuery {
        name: "unfinished",
        text: "sessions where repo:here ends_open:true | sort by last_event_at desc",
        help: "Unanswered prompts, unfinished calls, open todos, handoffs and partial outcomes here",
    },
    ShippedQuery {
        name: "notices",
        text: "deliveries where sender:signal | sort by at desc",
        help: "Signal notices, newest first",
    },
    ShippedQuery {
        name: "review",
        text: "annotations where NOT author:operator judged:unknown | sort by at desc",
        help: "Unjudged annotations written by someone else",
    },
    ShippedQuery {
        name: "questions",
        text: "annotations where kind:question answered:false",
        help: "Questions without a current answer",
    },
    ShippedQuery {
        name: "triage",
        text: "sessions where triage_score > 0 | sort by triage_score desc",
        help: "Sessions with unresolved failures, outliers, corrections or open items",
    },
    ShippedQuery {
        name: "saved",
        text: "queries where name:* | sort by saved_at desc",
        help: "Named questions, most recently saved first",
    },
    ShippedQuery {
        name: "open",
        text: "annotations where kind:(question|goal|problem) open:true",
        help: "Open questions, goals and problems",
    },
];

/// A named question's authored history and its immutable kept results.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SavedQueryRecord {
    pub name: String,
    pub revisions: Vec<SavedRevision>,
    pub saved_rows: Vec<KeptSummary>,
    pub watch: Option<String>,
    pub dependents: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SavedRevision {
    pub revision: u32,
    pub reference: String,
    pub text: String,
    pub hash: String,
    pub saved_at: Option<crate::Timestamp>,
    pub author: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct KeptSummary {
    pub result: String,
    pub rows: u64,
    pub at: crate::Timestamp,
}

/// The current authored definition behind an explicit dependent re-save call.
#[derive(Debug, Clone)]
pub struct DependentSavedQuery {
    pub name: String,
    pub revision: crate::RevisionReference,
    pub value: crate::annotation::values::SavedQuery,
}
