//! Types every Polylogue crate shares, their rules and
//! crash-safe file replacement.
// build-settings experiment: rebuild 2

#[macro_use]
mod stored_names;
pub use stored_names::StoredName;

mod helpers;
pub use helpers::{JsonExt, LockExt, hex, spawn_named, unpoisoned};
pub mod cause;
pub use cause::{
    Cause, Caused, ChainFailure, ChainFailureKind, Failure, Panicked, caught, install_panic_hook,
};
pub mod failure;
pub use failure::{Code, Names, Reason};

pub mod annotation;
pub mod atomic_write;
pub mod attachment_binding;
pub mod authored;
pub mod authorship;
pub mod batch;
mod block;
pub mod bundle;
pub mod capture;
pub mod capture_log;
mod change_position;
pub mod clock;
pub mod freshness;
pub use clock::WallClock;
pub mod command;
pub mod context;
mod delivery;
mod document;
mod event;
mod id;
pub mod injection;
pub mod kept;
pub mod live;
pub mod markdown;
pub mod sinex_event;
pub use injection::{HookContextEmission, ParsedHookField};
mod message;
pub mod money;
pub mod price;
pub mod query_objects;
mod quota;
mod reference;
pub mod saved_queries;
pub use reference::{
    Reference, ReferenceError, RevisionReference, SINEX_LOCAL_KINDS, SpanBounds, SpanError,
    SpanUnit, TEXT_BODY_SEPARATOR, TextBodyError, TextBodyOffsets, TextSpanPiece,
    annotation_revisions, span_bodies, text_body,
};
pub mod read;
mod route;
pub use route::{Completion, FieldGroup, Route};
mod alternate;
mod session;
pub mod settings;
mod time;
mod usage;
pub mod user;
pub use alternate::{
    Alternate, AlternateEffect, AlternateKind, AlternatePart, AlternateRelation, NewAlternate,
    NewAlternatePart,
};

pub use batch::{
    Alias, BackgroundUpdate, CountMismatch, DocumentRow, ExtractedBlob, IdKind, MessageRow,
    OrderKey, Output, RecordCounts, RecordOutcome, RepoLookup, StoredField, WireShape,
};
pub use block::{
    Attachment, AttachmentDirection, Block, BlockKind, CallOutcome, ErrorReason, ToolAction,
    ToolOutcome, UnknownReason,
};
pub use change_position::ChangePosition;
pub use delivery::{DELIVERY_STATES, DeliveryEvent, DeliveryOutcome, DeliverySender};
pub use document::{Document, DocumentAuthorship, DocumentPurpose};
pub use event::{
    CompactionBoundary, Event, KeptRange, LINK_HOW_NAMES, LINK_KIND_NAMES, Link, LinkKind,
    MicrocompactionBoundary, PrefixCopy, PrefixMember, PrefixSource, ReferenceReason,
};
pub use id::{BlobHash, DocumentId, IdError, MessageId, NativeId, Origin, RunId, SessionId};
pub mod reports;
pub use authorship::Author;
pub use message::{
    ApiError, Attribution, Message, Production, PromptKind, Role, ServerToolUse, WrittenBy,
    is_harness_message, is_prompt,
};
pub use reports::{QuotaReport, UsageReport};
pub use session::{Session, SessionKind, Title, TitleSource};
pub use time::{
    Precision, TimeAccuracy, TimeAccuracyError, TimeQuality, Timestamp, local_clock, local_day,
    local_minute, rfc3339_in,
};
pub mod text;
pub use price::{Catalog, LongContext, Multiplier, Price, Rate};
pub use quota::{FIVE_HOUR_MINUTES, QuotaLimit, QuotaReading, QuotaWindow, WEEK_MINUTES};
pub use usage::Usage;

/// Opaque JSON kept exactly as the source wrote it. Numbers are lossless
/// (`serde_json` with `arbitrary_precision`).
pub type Json = serde_json::Value;

pub mod derived;

macro_rules! string_schema {
    ($($type:ty),* $(,)?) => { $(
        impl schemars::JsonSchema for $type {
            fn schema_name() -> std::borrow::Cow<'static, str> {
                stringify!($type).into()
            }
            fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
                <String as schemars::JsonSchema>::json_schema(generator)
            }
        }
    )* };
}
string_schema!(
    Origin, SessionId, MessageId, DocumentId, BlobHash, RunId, Reference, Author
);

pub mod signal;
