//! Types every Polylogue crate shares. No IO and no storage: values, their
//! rules, and the export format's codec.

#[macro_use]
mod stored_names;

pub mod batch;
mod block;
pub mod command;
mod event;
pub mod export;
mod id;
mod message;
mod note;
pub mod price;
mod session;
mod time;
mod usage;
mod work;

pub use batch::{
    Accounting, BackgroundUpdate, CountMismatch, Disposition, InlineBlob, MessageRow, OrderKey,
    Output, RepoLookup,
};
pub use block::{
    Attachment, AttachmentDirection, Block, BlockKind, ErrorReason, ToolAction, ToolOutcome,
    UnknownReason,
};
pub use event::{Event, LINK_HOW_NAMES, LINK_KIND_NAMES, Link, LinkKind};
pub use id::{BlobHash, IdError, MessageId, NativeId, Origin, SessionId};
pub use message::{ApiError, Attribution, Message, Production, PromptKind, Role};
pub use note::{Author, NewNote, Note, NoteKind, Relation, Target, TargetError};
pub use session::{Session, SessionKind, Title, TitleSource};
pub use time::{LogicalDay, Timestamp};
pub mod text;
pub use price::{Catalog, LongContext, Price};
pub use usage::Usage;
pub use work::{WorkEdgeKind, WorkNodeKind};

/// Opaque JSON kept exactly as the source wrote it. Numbers are lossless
/// (`serde_json` with `arbitrary_precision`).
pub type Json = serde_json::Value;
