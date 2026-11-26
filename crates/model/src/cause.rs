//! What ends a failure, declared once. Every error type that crosses a crate
//! boundary answers [`Caused::cause`]; whether work is tried again, recorded
//! against its input, reported to the caller, or stops the unit doing it is
//! decided from that answer and never from an error's text.

use crate::Code;
use std::cell::RefCell;
use std::fmt;

stored_names! {
    /// What would end a failure.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum Cause {
        /// Passes by itself: a busy database, an index replaced under a read,
        /// an interrupted call. Doing the same thing again is the remedy.
        Transient => "transient",
        /// These bytes under this reading version: stands until either
        /// changes.
        Content => "content",
        /// The caller's request: the caller changes it.
        Request => "request",
        /// The machine: permissions, a full disk, an unreadable location.
        /// Stands until someone changes the machine.
        Environment => "environment",
        /// Stored data this build cannot read back, or a blob failing its
        /// hash.
        StoredData => "stored_data",
        /// A broken invariant in Polylogue.
        Bug => "bug",
        /// The process is stopping or the caller withdrew.
        Stopping => "stopping",
    }
}

impl Cause {
    /// Whether a client may send the same request again unchanged: the
    /// condition passes by itself, or the daemon that refused it was
    /// stopping.
    pub fn retryable(self) -> bool {
        matches!(self, Self::Transient | Self::Stopping)
    }

    /// Whether the same work on the same input fails the same way again:
    /// the input's content, the request or a bug caused it. Any other
    /// failure is one a later attempt can get past.
    pub fn recurs(self) -> bool {
        matches!(self, Self::Content | Self::Request | Self::Bug)
    }

    /// The cause of an operating-system I/O failure, from its kind. A
    /// failure the system does not classify (`EIO` among them) is taken to
    /// pass: the same bytes are written or read again.
    pub fn of_io(error: &std::io::Error) -> Self {
        use std::io::ErrorKind;
        match error.kind() {
            ErrorKind::PermissionDenied
            | ErrorKind::ReadOnlyFilesystem
            | ErrorKind::NotFound
            | ErrorKind::IsADirectory
            | ErrorKind::NotADirectory
            | ErrorKind::AlreadyExists
            | ErrorKind::DirectoryNotEmpty
            | ErrorKind::StorageFull
            | ErrorKind::QuotaExceeded
            | ErrorKind::FileTooLarge
            | ErrorKind::NotSeekable
            | ErrorKind::CrossesDevices
            | ErrorKind::TooManyLinks
            | ErrorKind::InvalidFilename
            | ErrorKind::ExecutableFileBusy => Self::Environment,
            ErrorKind::InvalidInput | ErrorKind::Unsupported => Self::Bug,
            ErrorKind::InvalidData | ErrorKind::UnexpectedEof => Self::Content,
            _ => Self::Transient,
        }
    }
}

impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An error that knows what ends it and which product code names it.
pub trait Caused: fmt::Display {
    fn cause(&self) -> Cause;
    fn code(&self) -> Code;
}

impl Caused for crate::Reason {
    fn cause(&self) -> Cause {
        Cause::Request
    }
    fn code(&self) -> Code {
        crate::Reason::code(self)
    }
}

/// One failure a unit of work ended with, kept as a value: what it happened
/// to, what ends it, its code and its sentence. A pass reports these; the
/// log and `status` show them.
#[derive(
    Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct Failure {
    /// What failed: a file's path, a location, an input or a step; empty
    /// for a whole turn.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub subject: String,
    pub cause: Cause,
    pub code: Code,
    pub message: String,
}

impl Failure {
    /// `error` as it happened to `subject`.
    pub fn new(subject: impl fmt::Display, error: &(impl Caused + ?Sized)) -> Self {
        Self {
            subject: subject.to_string(),
            cause: error.cause(),
            code: error.code(),
            message: error.to_string(),
        }
    }

    /// The failure as met while doing `what` to its subject.
    pub fn doing(mut self, what: &str) -> Self {
        self.message = format!("{what}: {}", self.message);
        self
    }

    /// A failure of a whole turn, with no one subject.
    pub fn of(error: &(impl Caused + ?Sized)) -> Self {
        Self::new("", error)
    }

    /// A failure stated in words where no error value carries it.
    pub fn stated(
        subject: impl fmt::Display,
        cause: Cause,
        code: Code,
        message: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.to_string(),
            cause,
            code,
            message: message.into(),
        }
    }

    /// A broken invariant found while working on `subject`.
    pub fn bug(subject: impl fmt::Display, message: impl Into<String>) -> Self {
        Self {
            subject: subject.to_string(),
            cause: Cause::Bug,
            code: Code::Internal,
            message: message.into(),
        }
    }

    /// Several failures as one: the first one's cause and code, every
    /// sentence. `None` when there are none.
    pub fn joined(failures: &[Failure]) -> Option<Self> {
        let first = failures.first()?;
        Some(Self {
            subject: String::new(),
            cause: first.cause,
            code: first.code,
            message: failures
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; "),
        })
    }

    /// The failure's identity for reporting it once: the same subject
    /// failing for the same cause and code is the same condition, whatever
    /// counts or offsets its sentence carries.
    pub fn key(&self) -> (&str, Cause, Code) {
        (&self.subject, self.cause, self.code)
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.subject.is_empty() {
            f.write_str(&self.message)
        } else {
            write!(f, "{}: {}", self.subject, self.message)
        }
    }
}

impl Caused for Failure {
    fn cause(&self) -> Cause {
        self.cause
    }
    fn code(&self) -> Code {
        self.code
    }
}

stored_names! {
    /// Why a chain's bytes have no reading under one reading version.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum ChainFailureKind {
        /// The parser could not read a record.
        Malformed => "malformed",
        /// The records a step counted differ from the records it read.
        CountsDiffer => "counts_differ",
        /// Decoded bytes differ from their declared hash or length.
        BytesDiffer => "bytes_differ",
        /// The parser panicked; the message and location are in the detail.
        ParserPanicked => "parser_panicked",
        /// The input failed its origin's shape check: no parser reads its
        /// recorded path, or it recorded none.
        Refused => "refused",
    }
}

/// A failure caused by a chain's bytes under its reading version: it
/// stands until either changes. Built only from a parser's own error, a
/// count mismatch, a hash mismatch or a caught parser panic, so a store or
/// I/O error can never be recorded against a chain.
#[derive(
    Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct ChainFailure {
    pub kind: ChainFailureKind,
    /// The parser's own sentence, for people.
    pub detail: String,
}

impl ChainFailure {
    pub fn malformed(detail: impl Into<String>) -> Self {
        Self {
            kind: ChainFailureKind::Malformed,
            detail: detail.into(),
        }
    }

    pub fn refused(detail: impl Into<String>) -> Self {
        Self {
            kind: ChainFailureKind::Refused,
            detail: detail.into(),
        }
    }

    pub fn bytes_differ() -> Self {
        Self {
            kind: ChainFailureKind::BytesDiffer,
            detail: "decoded bytes differ from their declared hash or length".into(),
        }
    }
}

impl From<crate::CountMismatch> for ChainFailure {
    fn from(mismatch: crate::CountMismatch) -> Self {
        Self {
            kind: ChainFailureKind::CountsDiffer,
            detail: mismatch.to_string(),
        }
    }
}

impl From<Panicked> for ChainFailure {
    fn from(panic: Panicked) -> Self {
        Self {
            kind: ChainFailureKind::ParserPanicked,
            detail: panic.to_string(),
        }
    }
}

impl fmt::Display for ChainFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for ChainFailure {}

impl Caused for ChainFailure {
    fn cause(&self) -> Cause {
        match self.kind {
            ChainFailureKind::Malformed => Cause::Content,
            ChainFailureKind::BytesDiffer => Cause::StoredData,
            ChainFailureKind::CountsDiffer | ChainFailureKind::ParserPanicked => Cause::Bug,
        }
    }
    fn code(&self) -> Code {
        match self.kind {
            ChainFailureKind::Malformed => Code::Import,
            ChainFailureKind::BytesDiffer
            | ChainFailureKind::CountsDiffer
            | ChainFailureKind::ParserPanicked => Code::Internal,
        }
    }
}

/// A panic caught by [`caught`]: its message and, when the process panic
/// hook recorded it, where it was raised.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Panicked {
    pub message: String,
    pub location: Option<String>,
}

impl fmt::Display for Panicked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.location {
            Some(location) => write!(f, "panicked at {location}: {}", self.message),
            None => write!(f, "panicked: {}", self.message),
        }
    }
}

impl std::error::Error for Panicked {}

impl Caused for Panicked {
    fn cause(&self) -> Cause {
        Cause::Bug
    }
    fn code(&self) -> Code {
        Code::Internal
    }
}

thread_local! {
    /// Set while this thread runs inside [`caught`]: how deep, and the
    /// location the panic hook recorded for the panic being caught.
    static CATCHING: RefCell<(u32, Option<String>)> = const { RefCell::new((0, None)) };
}

/// Runs `work`, turning a panic into a value with its message and location.
/// The one place a panic becomes a value.
pub fn caught<T>(work: impl FnOnce() -> T) -> Result<T, Panicked> {
    CATCHING.with_borrow_mut(|(depth, location)| {
        *depth += 1;
        *location = None;
    });
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
    let location = CATCHING.with_borrow_mut(|(depth, location)| {
        *depth -= 1;
        location.take()
    });
    result.map_err(|payload| Panicked {
        message: panic_message(payload.as_ref()),
        location,
    })
}

/// The text a panic was raised with.
pub fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|text| (*text).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic without a message".to_owned())
}

/// Installs the process panic hook: a panic inside [`caught`] leaves its
/// location for `caught` to return and prints nothing; any other panic is
/// handed to `uncaught` with the thread's name, the panic and a backtrace.
/// Called once by each binary's `main`.
pub fn install_panic_hook(uncaught: impl Fn(&str, &Panicked, &str) + Send + Sync + 'static) {
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|location| format!("{}:{}", location.file(), location.line()));
        let inside = CATCHING.try_with(|catching| {
            let mut catching = catching.borrow_mut();
            if catching.0 > 0 {
                catching.1.clone_from(&location);
                true
            } else {
                false
            }
        });
        if inside == Ok(true) {
            return;
        }
        let thread = std::thread::current();
        let panic = Panicked {
            message: panic_message(info.payload()),
            location,
        };
        let backtrace = std::backtrace::Backtrace::force_capture().to_string();
        uncaught(thread.name().unwrap_or("unnamed"), &panic, &backtrace);
    }));
}

#[cfg(test)]
mod tests;
