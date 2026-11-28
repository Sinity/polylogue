//! Product failure codes classified once into the shared transport contract.
pub mod acts;
pub mod browser;
pub mod complete {
    //! Completion of a query, or of a setting's key or value, uses the shared
    //! replacement span and candidate type.
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct Arguments {
        /// The full query text
        #[serde(default)]
        #[schemars(extend("argument_form" = "positional"))]
        pub text: String,
        /// UTF-8 byte offset of the cursor (default: end of text)
        pub position: Option<u64>,
        /// Complete setting keys instead of a query: the text is the typed
        /// start of the key
        #[serde(default)]
        pub setting: bool,
        /// With `setting`, complete this key's values instead
        pub key: Option<String>,
    }
    pub type Answer = Vec<query_language::complete::Completion>;
}

pub mod embed {
    //! Embedding builds and their previews.
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct Arguments {
        /// A sessions query selecting the work; use | limit N to bound its rows
        pub query: Option<String>,
        /// Confirm a build; without this, only preview
        #[serde(default)]
        pub yes: bool,
    }
}

pub mod help {
    //! Pages of the language and the full command reference.
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct Arguments {
        /// The language topic or verb to describe
        #[schemars(extend("argument_form" = "positional"))]
        pub topic: Option<String>,
        /// Print the full reference
        #[serde(default)]
        pub all: bool,
        /// List fields for this unit
        #[schemars(extend("unlisted" = true))]
        pub unit: Option<String>,
        /// Declarations per page (default 20)
        #[schemars(extend("unlisted" = true))]
        pub page_size: Option<u64>,
    }
}

pub mod follow;
pub mod import {
    //! Explicit file ingestion through the accepting daemon.
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct Arguments {
        /// Files or directories on the daemon host (relative paths need the caller's working directory), or one web chat's name (chatgpt, claude-ai, grok) to import its history through the browser extension; a file named like one is ./chatgpt
        #[serde(default)]
        #[schemars(extend("argument_form" = "words"))]
        pub paths: Vec<std::path::PathBuf>,
        /// The web chat whose history the browser extension imports: chatgpt, claude-ai or grok (on the command line, its name alone in place of the paths)
        pub provider: Option<String>,
        /// Read the paths as this declared structure (an origin name), or python-vectors for one of Python Polylogue's vector stores
        #[serde(default)]
        pub origin: Option<String>,
        /// Fetch provider conversations changed since this time
        pub since: Option<String>,
        /// Cancel a provider import
        #[serde(default)]
        pub cancel: bool,
        /// Wait this long for a provider import
        pub wait: Option<String>,
        /// Recognize and classify every member, storing nothing
        #[serde(default)]
        pub dry_run: bool,
    }
    impl Arguments {
        /// The provider by the extension's first name for it: `provider`, or
        /// the one path when it is a web chat's name (a file named like one
        /// is written `./chatgpt`).
        pub fn provider(&self) -> Option<&'static str> {
            let name = match (self.provider.as_deref(), self.paths.as_slice()) {
                (Some(name), _) => name,
                (None, [path]) => path.to_str()?,
                _ => return None,
            };
            polylogue_sources::origin::web_chat_name(name)
        }
    }

    /// What files and bundles brought in, or the provider's import job.
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    #[serde(untagged)]
    pub enum Answer {
        Files(Box<polylogue_engine::ImportReport>),
        Provider(crate::protocol::browser::ImportJobAnswer),
    }
}

pub mod line;
pub mod query;
pub mod resume {
    //! Resuming a session through its tool's own command.
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    /// The sessions in a directory worth resuming, or one session's plan.
    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    #[serde(untagged)]
    pub enum Answer {
        Sessions(Vec<crate::resume::ResumableSession>),
        Plan(crate::resume::Plan),
    }
}

pub mod schema {
    //! Schema operations validate the declaration's accepted selection arguments.
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct Infer {
        /// One origin (default: all origins)
        #[schemars(extend("argument_form" = "positional"))]
        pub origin: Option<String>,
        /// Select every origin
        #[serde(default)]
        #[schemars(extend("unlisted" = true))]
        pub all: bool,
        /// Explicit stored input IDs (repeatable)
        #[serde(default)]
        #[schemars(extend("unlisted" = true))]
        pub input: Vec<String>,
        /// Observation time lower bound, RFC 3339 with an offset
        #[schemars(extend("unlisted" = true))]
        pub since: Option<String>,
        /// Exact revision (default: latest for each input)
        #[schemars(extend("unlisted" = true))]
        pub revision: Option<u64>,
        /// Keep these field names instead of grouping them as dynamic keys
        #[serde(default)]
        #[schemars(extend("unlisted" = true))]
        pub field: Vec<String>,
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct Unread {
        /// One origin (default: all origins)
        #[schemars(extend("argument_form" = "positional"))]
        pub origin: Option<String>,
        /// Select every origin
        #[serde(default)]
        #[schemars(extend("unlisted" = true))]
        pub all: bool,
    }
}

pub mod setting {
    //! Runtime setting catalog, values and write results share one declared operation.
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct Arguments {
        /// The setting; without it, list catalog, defaults, scopes and values
        #[schemars(extend("argument_form" = "positional"))]
        pub key: Option<String>,
        /// The new value; without it, read the value, allowed choices and history
        #[schemars(extend("argument_form" = "positional"))]
        pub value: Option<String>,
        /// A named host for host-scoped settings; default: the accepting machine
        pub machine: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    #[serde(untagged)]
    pub enum Answer {
        Catalog(Vec<polylogue_model::settings::Current>),
        Current(Box<polylogue_model::settings::Details>),
        Outcome(polylogue_model::user::UserOutcome),
    }

    #[cfg(test)]
    mod tests;
}

pub mod status;
pub mod steps;
pub mod stream {
    //! Server-sent-event framing for every reply the daemon sends.
    use serde::Serialize;
    use std::io::{self, Write};

    pub fn event(output: &mut dyn Write, name: &str, value: &impl Serialize) -> io::Result<()> {
        let event = command_surface::SseEvent {
            id: None,
            event: name.into(),
            data: serde_json::to_string(value)?,
            retry: None,
        };
        let frame = event.encode().map_err(io::Error::other)?;
        output.write_all(frame.as_bytes())?;
        output.flush()
    }
}

pub mod wire;
use command_surface::{Answer, Failure, FailureKind, InputSpan};

pub fn kind(code: polylogue_model::Code) -> FailureKind {
    use polylogue_model::Code;
    match code {
        Code::Query | Code::Sql | Code::InvalidQuery => FailureKind::InvalidQuery,
        Code::UnknownField => FailureKind::UnknownField,
        Code::BadValue => FailureKind::BadValue,
        Code::NotAQuery => FailureKind::NotAQuery,
        Code::BadRequest
        | Code::Invalid
        | Code::IncompleteParents
        | Code::UnsupportedProvider
        | Code::EmptyText
        | Code::ModelRequired
        | Code::AttachmentNameNotABasename
        | Code::InvalidAction
        | Code::SubmissionReceiptRequired
        | Code::SubmissionConversationMismatch
        | Code::SinceIsADateOrTime
        | Code::UnknownOutcome
        | Code::UnknownAttachment
        | Code::UnknownReport
        | Code::NotAVerb
        | Code::NeedsAll
        | Code::RowNumbersOffTerminal => FailureKind::BadRequest,
        Code::Ambiguous => FailureKind::Ambiguous,
        Code::StaleCursor => FailureKind::StaleCursor,
        Code::NotBuilt => FailureKind::NotBuilt,
        Code::NotYetAvailable
        | Code::IndexBuilding
        | Code::PrimaryContextMissing
        | Code::PrimaryContextPending
        | Code::PrimaryContextStale => FailureKind::NotYetAvailable,
        Code::ModelCallsOff
        | Code::Refused
        | Code::OperatorRequired
        | Code::EmbeddingsOff
        | Code::Confirm
        | Code::Rejected
        | Code::PrimaryContextRefused => FailureKind::Refused,
        Code::NotFound
        | Code::UnknownChatAction
        | Code::UnknownChatActionAttachment
        | Code::AttachmentMissing
        | Code::AttachmentUnavailable
        | Code::UnknownImportJob => FailureKind::NotFound,
        Code::Conflict
        | Code::Stale
        | Code::ScopeChanged
        | Code::ResyncRequired
        | Code::IdempotencyKeyReused
        | Code::ActionFinished
        | Code::NotTheClaimant
        | Code::StageOnlyActionCannotSubmit
        | Code::NotAwaitingApproval
        | Code::JobEnded
        | Code::BrowserActionsOff => FailureKind::Conflict,
        Code::StoppedEarly => FailureKind::StoppedEarly,
        Code::NotRunning
        | Code::DaemonRequired
        | Code::ShuttingDown
        | Code::DaemonProtocolMismatch => FailureKind::NotRunning,
        Code::DeadlineExceeded | Code::Busy => FailureKind::Timeout,
        Code::Internal => FailureKind::Internal,
        Code::Cancelled => FailureKind::Cancelled,
        Code::Demo
        | Code::AttachmentOutput
        | Code::ReaderUnavailable
        | Code::Schema
        | Code::ReadOutput
        | Code::Io
        | Code::BlobCorrupt
        | Code::Store
        | Code::Import
        | Code::Embed
        | Code::SinexUnavailable
        | Code::PrimaryContextUnavailable
        | Code::PrimaryContextInvalid => FailureKind::Io,
    }
}

/// An answer whose value does not decode as its operation's declared type:
/// the daemon that answered was built from other sources.
pub fn protocol_mismatch(operation: &str, error: impl std::fmt::Display) -> Failure {
    failure(
        polylogue_model::Code::DaemonProtocolMismatch,
        format!(
            "the daemon answered {operation} in a shape this build does not read ({error}); \
             it speaks another protocol version, so restart it with `polylogue daemon`"
        ),
    )
}

/// A product failure code, classified once into the shared kind.
pub fn failure(code: polylogue_model::Code, message: impl Into<String>) -> Failure {
    let mut failure = Failure::new(kind(code), message);
    failure.code = Some(code.as_str().to_owned());
    if code == polylogue_model::Code::Cancelled {
        failure.retryable = true;
    }
    failure
}

/// A refusal under its reason's code, with the reason's sentence as the
/// message and the reason itself, its name and fields, on the wire.
pub fn refusal(reason: &polylogue_model::Reason) -> Failure {
    let mut wire = command_surface::FailureReason::new(reason.name());
    if let Ok(serde_json::Value::Object(mut serialised)) = serde_json::to_value(reason)
        && let Some(serde_json::Value::Object(fields)) = serialised.remove("fields")
    {
        wire.fields = fields;
    }
    failure(reason.code(), reason.to_string()).with_reason(wire)
}

/// A query error keeps its position, correction and hint as typed fields.
pub fn query_failure(error: &polylogue_store::query::QueryError) -> Failure {
    use polylogue_store::query::QueryError;
    let mut failure = failure(error.kind(), error.to_string());
    let position = match error {
        QueryError::Language { error, .. } => {
            failure.hint = error.hint.clone();
            failure.retryable = error.retryable;
            Some(error.position)
        }
        QueryError::Syntax { position, .. }
        | QueryError::UnknownField { position, .. }
        | QueryError::BadValue { position, .. }
        | QueryError::Invalid { position, .. } => Some(*position),
        _ => None,
    };
    let mut failure = failure.with_candidates(match error {
        QueryError::Ambiguous { candidates, .. } => candidates.clone(),
        QueryError::UnknownField { similar, .. } => similar.clone(),
        QueryError::Language { error, .. } => error.candidates.clone(),
        _ => Vec::new(),
    });
    failure.position = position.map(|offset| InputSpan { offset, length: 1 });
    if let Some(text) = error.suggestion() {
        failure = failure.with_suggestion(command_surface::Suggestion::Query(text.to_owned()));
    }
    failure
}

/// Every store error reaches surfaces through this one classification.
pub fn store_failure(error: &polylogue_store::StoreError) -> Failure {
    match error {
        polylogue_store::StoreError::Query(error) => query_failure(error),
        polylogue_store::StoreError::Refused(reason) => refusal(reason),
        error => {
            let mut failure = failure(error.code(), error.to_string());
            failure.retryable = polylogue_model::Caused::cause(error).retryable();
            match error {
                polylogue_store::StoreError::Ambiguous { candidates, .. } => {
                    failure.with_candidates(candidates.clone())
                }
                _ => failure,
            }
        }
    }
}

/// Human-facing diagnostics include the missing parts beside the other notes.
pub fn notes<T>(answer: &Answer<T>) -> impl Iterator<Item = &String> {
    answer.envelope.notes.iter().chain(
        match &answer.outcome {
            command_surface::Outcome::Degraded { missing, .. } => missing.as_slice(),
            _ => &[],
        }
        .iter(),
    )
}

pub fn suggestion_text(failure: &Failure) -> Option<std::borrow::Cow<'_, str>> {
    match failure.suggestion()? {
        command_surface::Suggestion::Query(text) => Some(text.as_str().into()),
        command_surface::Suggestion::Request(request) => Some(
            format!(
                "polylogue --request {}",
                polylogue_render::shell_quote(
                    &serde_json::to_string(request).expect("request serializes")
                )
            )
            .into(),
        ),
    }
}

/// The value of an answer that carries one, or its failure.
pub fn into_value<T>(answer: Answer<T>) -> Result<T, Failure> {
    match answer {
        command_surface::Answer {
            outcome: command_surface::Outcome::Ok(value),
            ..
        }
        | command_surface::Answer {
            outcome: command_surface::Outcome::Empty(value),
            ..
        }
        | command_surface::Answer {
            outcome: command_surface::Outcome::Degraded { value, .. },
            ..
        } => Ok(value),
        command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            ..
        } => Err(error),
    }
}

/// A write accepted into the local primary spool is degraded until Sinex confirms it.
pub fn written<T>(value: T, pending: bool) -> Answer<T> {
    if pending {
        command_surface::Answer {
            outcome: command_surface::Outcome::Degraded {
                value,
                missing: vec![AWAITING_CONFIRMATION.into()],
            },
            envelope: command_surface::Envelope {
                next: vec![],
                ..Default::default()
            },
        }
    } else {
        Answer::ok(value)
    }
}

pub const AWAITING_CONFIRMATION: &str = "accepted locally; awaiting Sinex confirmation";

/// A read over an index that is not current keeps its value and says why.
pub fn read<T>(value: T, freshness: &polylogue_model::freshness::ReadFreshness) -> Answer<T> {
    if freshness.up_to_date() {
        Answer::ok(value)
    } else {
        command_surface::Answer {
            outcome: command_surface::Outcome::Degraded {
                value,
                missing: freshness.notes(),
            },
            envelope: command_surface::Envelope {
                next: vec![],
                ..Default::default()
            },
        }
    }
}

#[cfg(test)]
mod tests;
