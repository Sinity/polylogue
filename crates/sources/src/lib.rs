//! Parsing for every origin. Each origin is one declaration ([`OriginSpec`])
//! and one parser, a pure function of bytes plus a small carried state. The
//! framework around it frames records, accounts for every one of them,
//! reads timestamps and checks that every wire field is read or excluded
//! with a reason.

pub mod account;
pub mod chatgpt;
pub mod claude_ai;
pub mod claude_code;
pub mod codex;
pub mod document;
pub mod fields;
pub mod frame;
pub mod kept_rows;
pub mod origin;
pub mod parse;
pub mod time;

pub use account::{Accounting, Disposition};
pub use frame::Record;
pub use origin::{Capture, Framing, ORIGINS, OriginSpec, Root, Route, Support};
pub use parse::{
    BackgroundUpdate, DocumentParser, InlineBlob, MessageRow, Output, ParseError, Parser, Step,
    StepParser, Whole, run, run_document,
};
