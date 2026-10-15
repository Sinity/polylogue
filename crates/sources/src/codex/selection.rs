//! Immutable extraction instructions for one mapper-proved field. These are
//! record selectors, never payload byte extents or content-version evidence.
use super::origins::RecordRange;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};

/// Descriptor wire layout, independent of parser checkpoint/cell storage.
pub const CODEC_VERSION: u32 = 1;
/// Source interpretation recipe. Unrelated parser-cell layout changes must not
/// make immutable historical field descriptors unreadable.
pub const ADAPTER_RECIPE_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathPart {
    Key(String),
    Index(u64),
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedRaw {
    pub record: RecordRange,
    pub path: Vec<PathPart>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Transform {
    /// Decode as the adapter's serde_json::Value; retain an opaque typed value,
    /// not raw numeric spelling or raw content-version canonical bytes.
    JsonValueV1,
    /// Decode the one selected JSON string.
    JsonStringV1,
    /// Decode selected strings in this exact order and join with LF.
    TextJoinLfV1,
    /// Current adapter: RFC3339 via jiff, milliseconds, Intrinsic/Instant.
    ProviderTimeV1,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactSelection {
    pub codec_version: u32,
    pub adapter_version: u32,
    pub transform: Transform,
    pub inputs: Vec<SelectedRaw>,
}
#[derive(Clone, Copy, Serialize)]
pub struct Binding<'a> {
    pub input: &'a str,
    /// Owning archive revision number (1-based), never a source byte offset.
    pub chain: i64,
}
#[derive(Debug)]
pub enum Error {
    Invalid(&'static str),
    Encoding(String),
    Io(io::Error),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "invalid Codex field selection: {reason}"),
            Self::Encoding(reason) => write!(f, "Codex field selection encoding failed: {reason}"),
            Self::Io(error) => write!(f, "Codex field selection sink failed: {error}"),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}
/// Evaluation keeps source storage failures distinct from unsupported paths and
/// sink failures. No resolver error is converted into an empty/unknown value.
#[derive(Debug)]
pub enum ValueError<E> {
    Selection(Error),
    Admission { required: u64, allowance: usize },
    Resolve(E),
    ByteCount { expected: u64, actual: usize },
    Json(serde_json::Error),
    Path { input: usize, component: usize },
    Type { input: usize },
    Time,
    Leaf(polylogue_model::source_leaf::Error),
    Io(io::Error),
}
impl<E: std::fmt::Display> std::fmt::Display for ValueError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Selection(error) => write!(f, "{error}"),
            Self::Admission {
                required,
                allowance,
            } => write!(
                f,
                "origin record requires {required} bytes; allowance is {allowance}"
            ),
            Self::Resolve(error) => write!(f, "origin record resolver failed: {error}"),
            Self::ByteCount { expected, actual } => write!(
                f,
                "origin record expected {expected} bytes, received {actual}"
            ),
            Self::Json(error) => write!(f, "origin record JSON failed: {error}"),
            Self::Path { input, component } => write!(
                f,
                "origin input {input} has no matching path component {component}"
            ),
            Self::Type { input } => write!(f, "origin input {input} has the wrong value type"),
            Self::Time => write!(f, "origin timestamp is invalid"),
            Self::Leaf(error) => write!(f, "normalized origin value failed: {error}"),
            Self::Io(error) => write!(f, "origin value sink failed: {error}"),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for ValueError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Selection(error) => Some(error),
            Self::Resolve(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Leaf(error) => Some(error),
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Cursor {
    next: usize,
}
impl FactSelection {
    pub(super) fn new(transform: Transform, inputs: Vec<SelectedRaw>) -> Self {
        Self {
            codec_version: CODEC_VERSION,
            adapter_version: ADAPTER_RECIPE_VERSION,
            transform,
            inputs,
        }
    }
    pub fn validate(&self) -> Result<(), Error> {
        if self.codec_version != CODEC_VERSION || self.adapter_version != ADAPTER_RECIPE_VERSION {
            return Err(Error::Invalid("unsupported version"));
        }
        if self.inputs.is_empty()
            || (self.transform != Transform::TextJoinLfV1 && self.inputs.len() != 1)
        {
            return Err(Error::Invalid("invalid input count"));
        }
        for input in &self.inputs {
            if input.record.start >= input.record.end {
                return Err(Error::Invalid("invalid record range"));
            }
        }
        Ok(())
    }
    pub fn cursor(&self) -> Cursor {
        Cursor::default()
    }
    /// One range per call. The immutable candidate's SET performs deduplication.
    /// Pin this descriptor, not a mutable parser cell, before retaining a cursor.
    pub fn advance(&self, cursor: &mut Cursor) -> Result<Option<RecordRange>, Error> {
        if self.codec_version != CODEC_VERSION || self.adapter_version != ADAPTER_RECIPE_VERSION {
            return Err(Error::Invalid("unsupported version"));
        }
        if self.inputs.is_empty()
            || (self.transform != Transform::TextJoinLfV1 && self.inputs.len() != 1)
        {
            return Err(Error::Invalid("invalid input count"));
        }
        if cursor.next > self.inputs.len() {
            return Err(Error::Invalid("cursor past end"));
        }
        let Some(input) = self.inputs.get(cursor.next) else {
            return Ok(None);
        };
        if input.record.start >= input.record.end {
            return Err(Error::Invalid("invalid record range"));
        }
        cursor.next += 1;
        Ok(Some(input.record))
    }
    /// Evaluate one immutable source-owned recipe into the SAME normalized leaf
    /// JSON spelling. The caller binds/authorizes every range against its pinned
    /// candidate before resolving it. This is synchronous admitted-field work:
    /// `record_allowance` admits each raw record, not cumulative read work. One
    /// full raw record is parsed at a time, with no universal CPU/RSS bound.
    /// Adjacent inputs in the same record share one decoded record. Failed output
    /// is partial and not resumable; callers must discard its private producer.
    pub fn write_value<E, W: Write>(
        &self,
        mut resolver: impl FnMut(RecordRange) -> Result<Vec<u8>, E>,
        record_allowance: usize,
        writer: &mut W,
    ) -> Result<(), ValueError<E>> {
        self.validate().map_err(ValueError::Selection)?;
        let mut sink = Sink {
            inner: writer,
            failure: None,
        };
        let result = (|| {
            let mut cached: Option<(RecordRange, serde_json::Value)> = None;
            let text = self.transform == Transform::TextJoinLfV1;
            if text {
                sink.write_all(b"\"").map_err(ValueError::Io)?;
            }
            for (ordinal, input) in self.inputs.iter().enumerate() {
                let required = input.record.end.checked_sub(input.record.start).ok_or(
                    ValueError::Selection(Error::Invalid("invalid record range")),
                )?;
                if required > u64::try_from(record_allowance).unwrap_or(u64::MAX) {
                    return Err(ValueError::Admission {
                        required,
                        allowance: record_allowance,
                    });
                }
                if cached
                    .as_ref()
                    .is_none_or(|(range, _)| *range != input.record)
                {
                    // Release the preceding decoded record BEFORE resolving the
                    // next. The raw Vec is dropped as soon as decoding completes.
                    drop(cached.take());
                    let bytes = resolver(input.record).map_err(ValueError::Resolve)?;
                    if u64::try_from(bytes.len()).ok() != Some(required) {
                        return Err(ValueError::ByteCount {
                            expected: required,
                            actual: bytes.len(),
                        });
                    }
                    let value = serde_json::from_slice(&bytes).map_err(ValueError::Json)?;
                    cached = Some((input.record, value));
                }
                let mut value = &cached
                    .as_ref()
                    .ok_or(ValueError::Selection(Error::Invalid(
                        "decoded record cache missing",
                    )))?
                    .1;
                for (component, part) in input.path.iter().enumerate() {
                    value = match part {
                        PathPart::Key(key) => value.as_object().and_then(|object| object.get(key)),
                        PathPart::Index(index) => usize::try_from(*index)
                            .ok()
                            .and_then(|index| value.as_array().and_then(|array| array.get(index))),
                    }
                    .ok_or(ValueError::Path {
                        input: ordinal,
                        component,
                    })?;
                }
                match self.transform {
                    Transform::JsonValueV1 => {
                        serde_json::to_writer(&mut sink, value).map_err(ValueError::Json)?
                    }
                    Transform::JsonStringV1 => {
                        let text = value.as_str().ok_or(ValueError::Type { input: ordinal })?;
                        serde_json::to_writer(&mut sink, text).map_err(ValueError::Json)?;
                    }
                    Transform::TextJoinLfV1 => {
                        use serde::Serializer;
                        let text = value.as_str().ok_or(ValueError::Type { input: ordinal })?;
                        let mut serializer =
                            serde_json::Serializer::with_formatter(&mut sink, StringFragment);
                        if ordinal > 0 {
                            serializer.serialize_str("\n").map_err(ValueError::Json)?;
                        }
                        serializer.serialize_str(text).map_err(ValueError::Json)?;
                    }
                    Transform::ProviderTimeV1 => {
                        let text = value.as_str().ok_or(ValueError::Type { input: ordinal })?;
                        let at = super::parse_timestamp(text).ok_or(ValueError::Time)?;
                        let time = super::provider_time(at);
                        let status = polylogue_model::source_leaf::write_time(&time, &mut sink)
                            .map_err(ValueError::Leaf)?;
                        if status != polylogue_model::source_leaf::PresenceStatus::Written {
                            return Err(ValueError::Selection(Error::Invalid(
                                "timestamp normalization unavailable",
                            )));
                        }
                    }
                }
            }
            if text {
                sink.write_all(b"\"").map_err(ValueError::Io)?;
            }
            Ok(())
        })();
        if let Some(error) = sink.failure {
            return Err(ValueError::Io(error));
        }
        result
    }
    /// Stream an already admitted descriptor into the caller's chunk producer.
    /// Binding must come from the pinned source snapshot. The owning writer must
    /// separately validate candidate membership and supported interpretation.
    /// A failed write is not resumable; discard the partial producer or replay.
    pub fn write_bound<W: Write>(&self, binding: Binding<'_>, writer: &mut W) -> Result<(), Error> {
        self.validate()?;
        if binding.input.is_empty() || binding.chain <= 0 {
            return Err(Error::Invalid("invalid binding"));
        }
        #[derive(Serialize)]
        struct Bound<'a> {
            format: &'static str,
            binding: Binding<'a>,
            selection: &'a FactSelection,
        }
        let mut sink = Sink {
            inner: writer,
            failure: None,
        };
        let result = serde_json::to_writer(
            &mut sink,
            &Bound {
                format: "codex-field-origin",
                binding,
                selection: self,
            },
        );
        if let Some(error) = sink.failure {
            return Err(Error::Io(error));
        }
        result.map_err(|error| Error::Encoding(error.to_string()))
    }
}
struct Sink<'a, W> {
    inner: &'a mut W,
    failure: Option<io::Error>,
}
impl<W: Write> Write for Sink<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.failure.is_some() {
            return Err(io::Error::other("origin sink already failed"));
        }
        match self.inner.write(bytes) {
            Ok(0) if !bytes.is_empty() => {
                self.failure = Some(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "origin sink accepted zero bytes",
                ));
                Err(io::Error::other("origin sink failed"))
            }
            Ok(n) => Ok(n),
            Err(error) => {
                self.failure = Some(error);
                Err(io::Error::other("origin sink failed"))
            }
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.failure.is_some() {
            return Err(io::Error::other("origin sink already failed"));
        }
        match self.inner.flush() {
            Ok(()) => Ok(()),
            Err(error) => {
                self.failure = Some(error);
                Err(io::Error::other("origin sink failed"))
            }
        }
    }
}

/// Reuse serde_json's string escaping across ordered fragments. Only the outer
/// field owns quotes; all fragment/control/Unicode handling stays library-owned.
struct StringFragment;
impl serde_json::ser::Formatter for StringFragment {
    fn begin_string<W: ?Sized + Write>(&mut self, _writer: &mut W) -> io::Result<()> {
        Ok(())
    }
    fn end_string<W: ?Sized + Write>(&mut self, _writer: &mut W) -> io::Result<()> {
        Ok(())
    }
}
