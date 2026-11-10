//! Types every Polylogue crate shares, their rules, the export codec and
//! durable file replacement.

#[macro_use]
mod stored_names;
pub use stored_names::StoredName;

pub mod assertion;
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
pub mod currency;
pub use clock::WallClock;
pub mod command;
pub mod context;
mod delivery;
mod document;
mod event;
pub mod export;
mod id;
pub mod injection;
pub mod kept;
pub mod live;
pub mod native;
pub mod parser_context;
pub use injection::{HookContextEmission, HookFieldClaim};
mod message;
pub mod money;
pub mod price;
pub mod query_objects;
mod quota;
mod reference;
pub use reference::{
    Reference, ReferenceError, RevisionReference, SpanBounds, SpanError, SpanUnit,
    TEXT_BODY_SEPARATOR, TextBodyError, TextBodyOffsets, TextSpanPiece, assertion_revisions,
    span_bodies, text_body,
};
pub mod read;
mod session;
pub mod settings;
mod time;
mod usage;
pub mod user;
mod variant;
pub use variant::{
    NewVariant, NewVariantPart, Variant, VariantEffect, VariantKind, VariantPart, VariantRelation,
};

pub use batch::{
    BackgroundUpdate, CountMismatch, DocumentRow, ExtractedBlob, IdKind, MessageRow, OrderKey,
    Output, RecordCounts, RecordOutcome, RepoLookup, WireShape,
};
pub use block::{
    Attachment, AttachmentDirection, Block, BlockKind, CallOutcome, ErrorReason, ToolAction,
    ToolOutcome, UnknownReason,
};
pub use change_position::ChangePosition;
pub use delivery::{DELIVERY_STATES, DeliveryEvent, DeliveryOutcome, DeliverySender};
pub use document::{Document, DocumentAuthorship, DocumentPurpose};
pub use event::{Event, LINK_HOW_NAMES, LINK_KIND_NAMES, Link, LinkKind};
pub use id::{BlobHash, DocumentId, IdError, MessageId, NativeId, Origin, RunId, SessionId};
pub mod reports;
pub use authorship::Author;
pub use message::{
    ApiError, Attribution, Message, Production, PromptKind, Role, ServerToolUse, WrittenBy,
    is_prompt,
};
pub use reports::{QuotaReport, UsageReport};
pub use session::{Session, SessionKind, Title, TitleSource};
pub use time::{
    Precision, TimeEvidence, TimeEvidenceError, TimeQuality, Timestamp, local_minute, rfc3339_in,
};
pub mod text;
pub use price::{Catalog, LongContext, Multiplier, Price, Rate};
pub use quota::{QuotaLimit, QuotaReading, QuotaWindow};
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
