//! Parsing for every origin. Each origin is one declaration ([`OriginSpec`])
//! and one parser, a pure function of bytes plus a small carried state. The
//! framework around it frames records, accounts for every one of them,
//! reads timestamps and checks that every wire field is read or excluded
//! with a reason.

pub mod account;
pub mod claude_code;
pub mod fields;
pub mod frame;
pub mod origin;
pub mod parse;
pub mod time;

pub use account::{Accounting, Disposition};
pub use frame::Record;
pub use origin::{ORIGINS, OriginSpec, Route, Support};
pub use parse::{
    BackgroundUpdate, InlineBlob, MessageRow, Output, ParseError, Parser, Step, StepParser, run,
};
