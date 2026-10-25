//! Parsing for every origin. Each origin is one declaration ([`OriginSpec`])
//! and one parser over bytes and carried state. Parsers do not open files or
//! databases; export deduplication calls the supplied first-message-owner lookup,
//! backed by operation-scoped storage in the engine. The framework frames
//! records, accounts for every one of them,
//! reads timestamps and checks that every wire field is read or excluded
//! with a reason.

#[cfg(test)]
extern crate self as polylogue_sources;

pub mod ai_studio;
pub mod browser_capture;
pub mod chatgpt;
pub mod claude_ai;
pub mod claude_code;
pub mod codex;
pub mod document;
pub mod export;
pub mod fields;
pub mod lines;
pub mod gemini_app;
pub mod gemini_cli;
pub mod grok;
pub mod grok_cli;
pub mod hermes;
pub mod hook_spool;
pub mod openai_chat;
pub mod openrouter;
pub mod origin;
pub mod otel_genai;
pub mod parse;
pub mod polylogue_export;
mod records;
pub mod repo_lookup;
pub mod time;

pub use lines::Record;
pub use origin::{Capture, ORIGINS, OriginSpec, Root, Arrival, Support};
pub use parse::{
    BackgroundUpdate, DocumentParser, ExtractedBlob, MessageRow, Output, Parse, ParseError, Parser,
    Step, StepParser, run, run_document,
};
pub use polylogue_model::{RecordCounts, RecordOutcome};
pub use records::Records;
