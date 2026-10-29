//! Types every Polylogue crate shares, their rules, the export codec and
//! durable file replacement.

#[macro_use]
mod stored_names;

pub mod atomic_write;
pub mod authored;
pub mod authorship;
pub mod batch;
mod block;
pub mod bundle;
pub mod currency;
pub mod capture;
pub mod capture_log;
mod change_position;
mod clock;
pub use clock::WallClock;
pub mod command;
mod delivery;
mod document;
mod event;
pub mod export;
mod id;
mod message;
mod note;
pub mod price;
mod quota;
mod reference;
pub use reference::{Reference, ReferenceError, RevisionReference, SpanUnit};
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
mod work;

pub use batch::{
    BackgroundUpdate, CountMismatch, DocumentRow, ExtractedBlob, MessageRow, OrderKey, Output,
    RecordCounts, RecordOutcome, RepoLookup, WireShape,
};
pub use block::{
    Attachment, AttachmentDirection, Block, BlockKind, CallOutcome, ErrorReason, ToolAction,
    ToolOutcome, UnknownReason,
};
pub use change_position::ChangePosition;
pub use delivery::{DELIVERY_STATES, DeliveryEvent, DeliveryOutcome, DeliverySender};
pub use document::{Document, DocumentAuthorship, DocumentPurpose};
pub use event::{Event, LINK_HOW_NAMES, LINK_KIND_NAMES, Link, LinkKind};
pub use id::{BlobHash, DocumentId, IdError, MessageId, NativeId, Origin, SessionId};
pub mod reports;
pub use message::{
    ApiError, Attribution, Message, Production, PromptKind, Role, ServerToolUse, WrittenBy,
    is_prompt,
};
pub use note::{Author, NewNote, Note, NoteKind, Relation};
pub use reports::{QuotaReport, UsageReport};
pub use session::{Session, SessionKind, Title, TitleSource};
pub use time::{LogicalDay, Precision, TimeEvidence, TimeEvidenceError, TimeQuality, Timestamp};
pub mod text;
pub use price::{Catalog, LongContext, Price};
pub use quota::{QuotaLimit, QuotaReading, QuotaWindow};
pub use usage::Usage;
pub use work::{WorkEdgeKind, WorkNodeKind};

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
    Origin, SessionId, MessageId, DocumentId, BlobHash, Reference, Author
);

pub mod work_products;

pub mod currency;
