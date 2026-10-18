//! Types every Polylogue crate shares. No IO and no storage: values, their
//! rules, and the export format's codec.

pub mod batch;
mod block;
mod event;
pub mod export;
mod id;
mod message;
mod note;
mod session;
mod time;
mod usage;
mod work;

pub use batch::{
    Accounting, BackgroundUpdate, CountMismatch, Disposition, InlineBlob, MessageRow, OrderKey,
    Output,
};
pub use block::{Attachment, AttachmentDirection, Block, ErrorReason, ToolOutcome, UnknownReason};
pub use event::{Event, Link, LinkKind};
pub use id::{BlobHash, IdError, MessageId, NativeId, Origin, SessionId};
pub use message::{ApiError, Attribution, Message, Production, PromptKind, Role};
pub use note::{Author, Note, NoteKind, Target};
pub use session::{Session, SessionKind, Title, TitleSource};
pub use time::Timestamp;
pub use usage::Usage;
pub use work::{WorkEdge, WorkEdgeKind, WorkNode, WorkNodeKind};

/// Opaque JSON kept exactly as the source wrote it. Numbers are lossless
/// (`serde_json` with `arbitrary_precision`).
pub type Json = serde_json::Value;
