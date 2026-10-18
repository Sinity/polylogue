//! Provider-neutral original-record selection, bounded record decoding and leaf
//! streaming. Provider modules own attribution, descriptor versions and recipes.
use polylogue_model::RawJson as Value;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
/// Exact half-open original-record coordinates; the provider-owned envelope
/// pins input/chain. These coordinates are never inferred payload extents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RecordRange {
    pub start: u64,
    pub end: u64,
}
impl RecordRange {
    pub fn new(start: u64, end: u64) -> Result<Self, String> {
        if start >= end {
            return Err("invalid original-record range".into());
        }
        Ok(Self { start, end })
    }
}
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
/// These operations interpret a provider-proved selected value. They neither
/// discover paths nor establish that a field was supplied by any provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    JsonValue,
    JsonString,
    NativeId,
    TextJoinLf,
    ProviderTime,
    ProviderScalar,
    /// Two supplied unsigned counts, in minuend/subtrahend order.
    UnsignedSaturatingDifference,
}
#[derive(Debug)]
pub enum ScalarError<S> {
    Invalid(S),
    Type,
    Time,
    Json(serde_json::Error),
    Leaf(polylogue_model::source_leaf::Error),
    Io(io::Error),
}
/// One immutable provider-owned interpretation recipe. Implementations validate
/// their own descriptor/recipe versions and suppliedness; this core never maps
/// a target Slot back into a source path. `inputs` and `operation` stay fixed
/// for the call. Optional scalar/time hooks must reuse that provider's mapper.
pub trait Recipe {
    type Error;
    fn provider(&self) -> &'static str;
    fn validate(&self) -> Result<(), Self::Error>;
    fn inputs(&self) -> &[SelectedRaw];
    fn operation(&self) -> Operation;
    fn invalid(&self, reason: &'static str) -> Self::Error;
    fn write_scalar<W: Write>(
        &self,
        _value: &Value,
        _sink: &mut W,
    ) -> Result<(), ScalarError<Self::Error>> {
        Err(ScalarError::Invalid(
            self.invalid("unsupported provider scalar transform"),
        ))
    }
}
pub fn validate_count(inputs: &[SelectedRaw], operation: Operation) -> Result<(), &'static str> {
    let valid = match operation {
        Operation::TextJoinLf => !inputs.is_empty(),
        Operation::UnsignedSaturatingDifference => inputs.len() == 2,
        _ => inputs.len() == 1,
    };
    if !valid {
        return Err("invalid input count");
    }
    Ok(())
}
pub fn validate_inputs(inputs: &[SelectedRaw], operation: Operation) -> Result<(), &'static str> {
    validate_count(inputs, operation)?;
    if inputs
        .iter()
        .any(|input| input.record.start >= input.record.end)
    {
        return Err("invalid record range");
    }
    Ok(())
}
/// Evaluation keeps source storage failures distinct from unsupported paths and
/// sink failures. No resolver error is converted into an empty/unknown value.
#[derive(Debug)]
pub enum ValueError<E, S> {
    Selection(S),
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
impl<E: std::fmt::Display, S: std::fmt::Display> std::fmt::Display for ValueError<E, S> {
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
impl<E: std::error::Error + 'static, S: std::error::Error + 'static> std::error::Error
    for ValueError<E, S>
{
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
impl Cursor {
    /// The caller pins one immutable descriptor before retaining this cursor.
    pub fn advance(&mut self, inputs: &[SelectedRaw]) -> Result<Option<RecordRange>, &'static str> {
        if self.next > inputs.len() {
            return Err("cursor past end");
        }
        let Some(input) = inputs.get(self.next) else {
            return Ok(None);
        };
        if input.record.start >= input.record.end {
            return Err("invalid record range");
        }
        self.next += 1;
        Ok(Some(input.record))
    }
}
/// Evaluate one immutable source-owned recipe into the SAME normalized leaf
/// JSON spelling. The caller binds/authorizes every range against its pinned
/// candidate before resolving it. This is synchronous admitted-field work:
/// `record_allowance` admits each raw record, not cumulative read work. One
/// full raw record is parsed at a time, with no universal CPU/RSS bound.
/// Adjacent inputs in the same record share one decoded record. Failed output
/// is partial and not resumable; callers must discard its private producer.
pub fn write_value<E, W: Write, R: Recipe>(
    recipe: &R,
    mut resolver: impl FnMut(RecordRange) -> Result<Vec<u8>, E>,
    record_allowance: usize,
    writer: &mut W,
) -> Result<(), ValueError<E, R::Error>> {
    recipe.validate().map_err(ValueError::Selection)?;
    validate_count(recipe.inputs(), recipe.operation())
        .map_err(|reason| ValueError::Selection(recipe.invalid(reason)))?;
    let mut sink = Sink {
        inner: writer,
        failure: None,
    };
    let result = (|| {
        let mut cached: Option<(RecordRange, polylogue_model::RawJson)> = None;
        let text = recipe.operation() == Operation::TextJoinLf;
        let mut minuend = None;
        if text {
            sink.write_all(b"\"").map_err(ValueError::Io)?;
        }
        for (ordinal, input) in recipe.inputs().iter().enumerate() {
            let required = input
                .record
                .end
                .checked_sub(input.record.start)
                .filter(|required| *required > 0)
                .ok_or(ValueError::Selection(
                    recipe.invalid("invalid record range"),
                ))?;
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
                .ok_or(ValueError::Selection(
                    recipe.invalid("decoded record cache missing"),
                ))?
                .1;
            for (component, part) in input.path.iter().enumerate() {
                value = match part {
                    PathPart::Key(key) => value.get(key),
                    PathPart::Index(index) => usize::try_from(*index)
                        .ok()
                        .and_then(|index| value.as_array().and_then(|array| array.get(index))),
                }
                .ok_or(ValueError::Path {
                    input: ordinal,
                    component,
                })?;
            }
            match recipe.operation() {
                Operation::JsonValue => {
                    serde_json::to_writer(&mut sink, value).map_err(ValueError::Json)?
                }
                Operation::NativeId => {
                    let native = value.native().ok_or(ValueError::Type { input: ordinal })?;
                    serde_json::to_writer(&mut sink, native).map_err(ValueError::Json)?;
                }
                Operation::JsonString => {
                    let text = value.as_str().ok_or(ValueError::Type { input: ordinal })?;
                    serde_json::to_writer(&mut sink, text).map_err(ValueError::Json)?;
                }
                Operation::TextJoinLf => {
                    use serde::Serializer;
                    let text = value.as_str().ok_or(ValueError::Type { input: ordinal })?;
                    let mut serializer =
                        serde_json::Serializer::with_formatter(&mut sink, StringFragment);
                    if ordinal > 0 {
                        serializer.serialize_str("\n").map_err(ValueError::Json)?;
                    }
                    serializer.serialize_str(text).map_err(ValueError::Json)?;
                }
                Operation::ProviderTime => {
                    let text = value.as_str().ok_or(ValueError::Type { input: ordinal })?;
                    let time = crate::time::parse_intrinsic(text).ok_or(ValueError::Time)?;
                    let status = polylogue_model::source_leaf::write_time(&time, &mut sink)
                        .map_err(ValueError::Leaf)?;
                    if status != polylogue_model::source_leaf::PresenceStatus::Written {
                        return Err(ValueError::Selection(
                            recipe.invalid("timestamp normalization unavailable"),
                        ));
                    }
                }
                Operation::ProviderScalar => {
                    recipe
                        .write_scalar(value, &mut sink)
                        .map_err(|error| match error {
                            ScalarError::Invalid(error) => ValueError::Selection(error),
                            ScalarError::Type => ValueError::Type { input: ordinal },
                            ScalarError::Time => ValueError::Time,
                            ScalarError::Json(error) => ValueError::Json(error),
                            ScalarError::Leaf(error) => ValueError::Leaf(error),
                            ScalarError::Io(error) => ValueError::Io(error),
                        })?;
                }
                Operation::UnsignedSaturatingDifference => {
                    let count = value.as_u64().ok_or(ValueError::Type { input: ordinal })?;
                    if ordinal == 0 {
                        minuend = Some(count);
                    } else {
                        let first = minuend.ok_or(ValueError::Selection(
                            recipe.invalid("missing unsigned minuend"),
                        ))?;
                        serde_json::to_writer(&mut sink, &first.saturating_sub(count))
                            .map_err(ValueError::Json)?;
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
/// The archive input whose records a descriptor selects.
#[derive(Clone, Copy, Serialize)]
pub struct Binding<'a> {
    pub input: &'a str,
    /// Archive revision number (1-based), never a source byte offset.
    pub chain: i64,
}
impl Binding<'_> {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.input.is_empty() || self.chain <= 0 {
            return Err("invalid binding");
        }
        Ok(())
    }
}
/// Stream `{"format", "binding", "selection"}` for one validated descriptor.
/// A failed write is not resumable; the caller discards the partial output.
pub fn write_bound<T: Serialize, W: Write>(
    format: &'static str,
    selection: &T,
    binding: Binding<'_>,
    writer: &mut W,
) -> Result<(), JsonWriteError> {
    #[derive(Serialize)]
    struct Bound<'a, T> {
        format: &'static str,
        binding: Binding<'a>,
        selection: &'a T,
    }
    write_json(
        &Bound {
            format,
            binding,
            selection,
        },
        writer,
    )
}
/// Streaming JSON encoding for provider-owned envelopes. The caller owns the
/// producer/admission and discards partial output on error; this is not resumable
/// serialization. The original first writer error is preserved.
#[derive(Debug)]
pub enum JsonWriteError {
    Encoding(serde_json::Error),
    Io(io::Error),
}
impl std::fmt::Display for JsonWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encoding(error) => write!(f, "origin JSON encoding failed: {error}"),
            Self::Io(error) => write!(f, "origin JSON sink failed: {error}"),
        }
    }
}
impl std::error::Error for JsonWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encoding(error) => Some(error),
            Self::Io(error) => Some(error),
        }
    }
}
pub fn write_json<T: Serialize, W: Write>(value: &T, writer: &mut W) -> Result<(), JsonWriteError> {
    let mut sink = Sink {
        inner: writer,
        failure: None,
    };
    let result = serde_json::to_writer(&mut sink, value);
    if let Some(error) = sink.failure {
        return Err(JsonWriteError::Io(error));
    }
    result.map_err(JsonWriteError::Encoding)
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
