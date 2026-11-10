//! Routes: the ways one conversation's reading reaches the archive, and the
//! precedence that decides, field group by field group, which route's value
//! a message shows (docs/sessions.md, one conversation through several
//! routes).

use crate::{Message, Role};
use serde::{Deserialize, Serialize};

/// How an input's reading of a conversation reached the archive.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    /// The tool's own file or database, where the tool writes it.
    Live,
    /// The provider's own data for one conversation, fetched as it is shown:
    /// the browser extension's capture of the page's data, a Drive revision.
    Capture,
    /// Records Polylogue's own hooks, process watcher and collectors write.
    Recorded,
    /// A provider export brought in with `polylogue import`.
    Export,
    /// A copy another program or archive made of a reading: an imported
    /// tool file, a Polylogue bundle, an older archive.
    Copy,
    /// Text read from the page instead of the provider's data: partial
    /// fidelity, outranked by every other route for every field.
    PageText,
}

stored_names!(Route {
    Live => "live",
    Capture => "capture",
    Recorded => "recorded",
    Export => "export",
    Copy => "copy",
    PageText => "page_text",
});

/// Origins whose export lists every conversation of the account, so a
/// conversation a newer export lacks is gone upstream.
const EXPORTS_EVERY_CONVERSATION: &[&str] = &["chatgpt", "claude-ai", "grok"];

impl Route {
    /// Whether an input of this route and format is an export that lists
    /// every conversation of its account.
    pub fn lists_every_conversation(self, format: &str) -> bool {
        self == Self::Export && EXPORTS_EVERY_CONVERSATION.contains(&format)
    }

    /// The route of an origin's input brought in by `polylogue import`
    /// rather than found where the tool writes it.
    pub const fn imported(self) -> Self {
        match self {
            Self::Export => Self::Export,
            Self::Live | Self::Capture | Self::Recorded | Self::Copy | Self::PageText => Self::Copy,
        }
    }

    /// Whether the input was brought in by `polylogue import`, named by its
    /// bytes.
    pub const fn is_imported(self) -> bool {
        matches!(self, Self::Export | Self::Copy)
    }
}

/// The fields of a message that are taken together from one route.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum FieldGroup {
    /// Role, prompt kind, blocks and the provider fields indexed by block.
    /// The route that supplies them is the message's main route: its
    /// version is the stored reading.
    Content,
    /// The provider time and its evidence.
    Time,
    Model,
    /// Token usage, the reported cost and the reported duration.
    Usage,
    /// The stop reason.
    Stop,
    /// The message this one replies to.
    Parent,
    /// How the harness produced the message.
    Production,
}

stored_names!(FieldGroup {
    Content => "content",
    Time => "time",
    Model => "model",
    Usage => "usage",
    Stop => "stop",
    Parent => "parent",
    Production => "production",
});

impl FieldGroup {
    /// The route precedence for this group: lower ranks first. Every origin
    /// shares this table, because no origin's routes call for another order;
    /// an origin that does gets its own arm here. Page text is last for
    /// every group.
    pub const fn rank(self, route: Route) -> u8 {
        match self {
            Self::Content
            | Self::Time
            | Self::Model
            | Self::Usage
            | Self::Stop
            | Self::Parent
            | Self::Production => match route {
                Route::Live => 0,
                Route::Capture => 1,
                Route::Recorded => 2,
                Route::Export => 3,
                Route::Copy => 4,
                Route::PageText => u8::MAX,
            },
        }
    }

    /// Whether the message has a value for this group.
    pub fn present(self, message: &Message) -> bool {
        match self {
            Self::Content => !message.blocks.is_empty(),
            Self::Time => message.at.is_some(),
            Self::Model => message.model.is_some(),
            Self::Usage => {
                !message.usage.is_empty()
                    || message.reported_cost_usd.is_some()
                    || message.duration_ms.is_some()
            }
            Self::Stop => message.stop_reason.is_some(),
            Self::Parent => message.parent.is_some(),
            Self::Production => !message.production.is_empty(),
        }
    }

    /// Replaces this group's fields of `shown` with `from`'s.
    pub fn take(self, shown: &mut Message, from: &Message) {
        match self {
            Self::Content => {
                shown.role = from.role;
                shown.prompt_kind = from.prompt_kind;
                shown.blocks.clone_from(&from.blocks);
                shown.attrs.clone_from(&from.attrs);
            }
            Self::Time => {
                shown.at = from.at;
                shown.at_quality = from.at_quality;
                shown.at_precision = from.at_precision;
                shown.at_until = from.at_until;
            }
            Self::Model => shown.model.clone_from(&from.model),
            Self::Usage => {
                shown.usage.clone_from(&from.usage);
                shown.reported_cost_usd.clone_from(&from.reported_cost_usd);
                shown.duration_ms = from.duration_ms;
            }
            Self::Stop => shown.stop_reason.clone_from(&from.stop_reason),
            Self::Parent => shown.parent.clone_from(&from.parent),
            Self::Production => shown.production.clone_from(&from.production),
        }
    }
}

/// Whether a reading's message is finished. Between two readings of the
/// same route, a finished one wins over one of unknown completion, which
/// wins over one still being produced.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Completion {
    Finished,
    /// The source does not say: an assistant message without a stop reason
    /// from an origin that does not record one.
    Unknown,
    /// An assistant message without a stop reason from an origin that
    /// records one when a response ends.
    InProgress,
}

stored_names!(Completion {
    Finished => "finished",
    Unknown => "unknown",
    InProgress => "in_progress",
});

/// Origins whose finished assistant messages always carry a stop reason.
const RECORDS_STOP_REASON: &[&str] = &["claude-code", "claude-ai", "ai-studio", "hermes"];

impl Completion {
    /// A message's completion as its route read it. Page text records no
    /// stop reasons.
    pub fn of(message: &Message, route: Route) -> Self {
        if message.role != Role::Assistant || message.stop_reason.is_some() {
            Self::Finished
        } else if route != Route::PageText
            && RECORDS_STOP_REASON.contains(&message.id.origin().as_str())
        {
            Self::InProgress
        } else {
            Self::Unknown
        }
    }
}

#[cfg(test)]
mod tests;
