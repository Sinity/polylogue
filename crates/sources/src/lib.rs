//! Parsing for every origin. Each origin is one declaration ([`OriginSpec`])
//! and one parser over bytes and carried state. Parsers do not open files or
//! databases; export deduplication calls the supplied first-session lookup,
//! backed by operation-scoped storage in the engine. The framework frames
//! records, accounts for every one of them,
//! reads timestamps and checks that every wire field is read or excluded
//! with a reason.

#[cfg(test)]
extern crate self as polylogue_sources;

// A parser's child declarations also collect their row versions.
macro_rules! parser_modules {
    (@join $($part:expr),* $(,)?) => { &{
        const PARTS: &[&[crate::origin::ParserModule]] = &[$($part),*];
        const LEN: usize = {
            let mut len = 0;
            let mut part = 0;
            while part < PARTS.len() {
                len += PARTS[part].len();
                part += 1;
            }
            len
        };
        let mut modules = [crate::origin::ParserModule { name: "", version: 0 }; LEN];
        let mut next = 0;
        let mut part = 0;
        while part < PARTS.len() {
            let mut index = 0;
            while index < PARTS[part].len() {
                modules[next] = PARTS[part][index];
                next += 1;
                index += 1;
            }
            part += 1;
        }
        modules
    } };
    ($($visibility:vis mod $module:ident;)*) => {
        $($visibility mod $module;)*
        pub(crate) const MODULES: &[crate::origin::ParserModule] = parser_modules!(@join
            &[$(crate::origin::ParserModule {
                name: concat!(module_path!(), "::", stringify!($module), "::VERSION"),
                version: $module::VERSION,
            }),*],
            $($module::MODULES),*
        );
    };
}

pub mod ai_studio;
pub mod aider;
pub mod antigravity;
pub mod browser_capture;
pub mod chain_names;
pub mod chatgpt;
pub mod claude_ai;
pub mod claude_code;
pub mod codex;
pub mod codex_cloud;
pub mod content_ids;
pub mod document;
pub mod export;
pub mod fields;
pub mod gemini_app;
pub mod gemini_cli;
pub mod grok;
pub mod grok_cli;
pub mod hermes;
pub mod hook_spool;
mod inline_bytes;
pub mod lines;
pub mod locations;
pub mod openai_chat;
pub mod openrouter;
pub mod origin;
pub mod otel_genai;
pub mod parse;
pub mod polylogue_bundle;
mod records;
pub mod repo_lookup;
pub mod row_dump;
pub mod schema;
pub mod time;

pub use lines::Record;
pub use origin::{Arrival, Capture, InputIdentity, ORIGINS, OriginSpec, Root, Support};
pub use parse::{
    BackgroundUpdate, DocumentParser, ExtractedBlob, MessageRow, Output, Parse, ParseError, Parser,
    Step, StepParser, run, run_document,
};
pub use polylogue_model::{RecordCounts, RecordOutcome};
pub use records::Records;
