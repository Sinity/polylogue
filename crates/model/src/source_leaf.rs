//! One normalized donor leaf, selected by a mapper-proved slot. This module
//! neither discovers suppliedness nor scans a Message's blocks or ranks it.
use crate::merge::{BlockField, MessageField, Slot, UsageField};
use crate::{BlockContent, BlockSlot, ErrorReason, Message, ProviderTime, ToolOutcome};
use serde::{Serialize, Serializer, ser::SerializeMap};
use serde_json::Value;
use std::{
    fmt,
    io::{self, Write},
};

#[derive(Clone, Copy)]
pub enum Subject<'a> {
    Message(&'a Message),
    /// The caller establishes semantic correspondence; no ordinal search.
    Block {
        slot: &'a BlockSlot,
        content: &'a BlockContent,
    },
    /// Immediate provider extras only; values stay opaque, including null.
    Extras {
        block: Option<&'a BlockSlot>,
        values: &'a Value,
    },
}
#[derive(Debug, PartialEq, Eq)]
pub enum Projection {
    Encoded(Vec<u8>),
    Missing,
    Unsupported,
}
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("normalized leaf exceeds {maximum} encoded bytes")]
    Admission { maximum: usize },
    #[error("normalized leaf encoding failed: {0}")]
    Encoding(String),
    #[error("normalized leaf sink failed: {0}")]
    Io(#[source] io::Error),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresenceStatus {
    Written,
    Missing,
    Unsupported,
}

struct Capped {
    bytes: Vec<u8>,
    maximum: usize,
    denied: bool,
}
impl Write for Capped {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.maximum.saturating_sub(self.bytes.len()) {
            self.denied = true;
            return Err(io::Error::other("leaf admission"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
// serde_json does not need to expose or reconstruct an original writer error.
// Keep that exact object here and fuse all subsequent access to the sink.
struct Sink<'a, W: Write> {
    inner: &'a mut W,
    failure: Option<io::Error>,
}
impl<W: Write> Write for Sink<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.failure.is_some() {
            return Err(io::Error::other("leaf sink already failed"));
        }
        match self.inner.write(bytes) {
            Ok(0) if !bytes.is_empty() => {
                self.failure = Some(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "leaf sink accepted zero bytes",
                ));
                Err(io::Error::other("leaf sink failed"))
            }
            Ok(n) => Ok(n),
            Err(error) => {
                self.failure = Some(error);
                Err(io::Error::other("leaf sink failed"))
            }
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.failure.is_some() {
            return Err(io::Error::other("leaf sink already failed"));
        }
        match self.inner.flush() {
            Ok(()) => Ok(()),
            Err(error) => {
                self.failure = Some(error);
                Err(io::Error::other("leaf sink failed"))
            }
        }
    }
}
fn encoded(value: &impl Serialize, sink: &mut impl Write) -> Result<PresenceStatus, Error> {
    let mut writer = Sink {
        inner: sink,
        failure: None,
    };
    let result = serde_json::to_writer(&mut writer, value);
    if let Some(error) = writer.failure {
        return Err(Error::Io(error));
    }
    result.map_err(|e| Error::Encoding(e.to_string()))?;
    Ok(PresenceStatus::Written)
}
fn optional<T: Serialize>(
    value: Option<&T>,
    sink: &mut impl Write,
) -> Result<PresenceStatus, Error> {
    match value {
        Some(v) => encoded(v, sink),
        None => Ok(PresenceStatus::Missing),
    }
}
/// Convenience collector over the same sink encoder. Logical encoded length is
/// admitted before Vec growth; allocator capacity and source decoding are not
/// RSS guarantees. Sink failure never returns a partial successful leaf.
pub fn encode(subject: Subject<'_>, proved: &Slot, maximum: usize) -> Result<Projection, Error> {
    let mut writer = Capped {
        bytes: Vec::new(),
        maximum,
        denied: false,
    };
    let result = write(subject, proved, &mut writer);
    if writer.denied {
        return Err(Error::Admission { maximum });
    }
    Ok(match result? {
        PresenceStatus::Written => Projection::Encoded(writer.bytes),
        PresenceStatus::Missing => Projection::Missing,
        PresenceStatus::Unsupported => Projection::Unsupported,
    })
}
// Fixed typed keys are emitted lexically. JSON Value maps use serde_json's
// default sorted map representation (this workspace does not enable preserve_order).
#[derive(Serialize)]
struct MessageRef<'a> {
    id: &'a str,
    session: &'a str,
}
impl<'a> From<&'a crate::MessageId> for MessageRef<'a> {
    fn from(id: &'a crate::MessageId) -> Self {
        Self {
            id: id.as_str(),
            session: id.session().as_str(),
        }
    }
}
#[derive(Serialize)]
struct BlockRef<'a> {
    id: &'a str,
    message: MessageRef<'a>,
    slot: &'a BlockSlot,
}
impl<'a> From<&'a crate::BlockId> for BlockRef<'a> {
    fn from(id: &'a crate::BlockId) -> Self {
        Self {
            id: id.as_str(),
            message: id.message().into(),
            slot: id.slot(),
        }
    }
}
struct Time<'a>(&'a ProviderTime);
impl Serialize for Time<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let (at, precision, quality, until) = match self.0 {
            ProviderTime::Intrinsic {
                at,
                precision,
                until,
            } => (at, precision, "intrinsic", until),
            ProviderTime::FileMtime {
                at,
                precision,
                until,
            } => (at, precision, "file_mtime", until),
            ProviderTime::Unknown => return serializer.serialize_none(),
        };
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry("at", at)?;
        map.serialize_entry("precision", precision)?;
        map.serialize_entry("time_quality", quality)?;
        map.serialize_entry("until", until)?;
        map.end()
    }
}
/// The same borrowed identity-relation wire used by normalized Parent leaves.
/// Source recipes must independently prove the relation before calling this.
pub fn write_message_reference(
    value: &crate::MessageId,
    sink: &mut impl Write,
) -> Result<PresenceStatus, Error> {
    encoded(&MessageRef::from(value), sink)
}
/// Encode a provider time through the same normalized leaf representation used
/// by Message fields. A source selector must still prove where it came from.
pub fn write_time(value: &ProviderTime, sink: &mut impl Write) -> Result<PresenceStatus, Error> {
    time(Some(value), sink)
}

fn time(value: Option<&ProviderTime>, sink: &mut impl Write) -> Result<PresenceStatus, Error> {
    match value {
        Some(ProviderTime::Unknown) | None => Ok(PresenceStatus::Missing),
        Some(v) => encoded(&Time(v), sink),
    }
}
struct Reason<'a>(&'a ErrorReason);
impl fmt::Display for Reason<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            ErrorReason::Denied(kind) => write!(f, "denied:{kind}"),
            ErrorReason::Interrupted => f.write_str("interrupted"),
            ErrorReason::TimedOut => f.write_str("timed_out"),
            ErrorReason::BlockedByHook => f.write_str("blocked_by_hook"),
            ErrorReason::Other(value) => f.write_str(value),
        }
    }
}
impl Serialize for Reason<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}
/// Write exactly one requested, caller-proved normalized donor leaf. This does
/// not discover suppliedness, rank or raw version evidence. Missing/Unsupported
/// write no bytes. Written means all JSON bytes were accepted, never that the
/// sink flushed, sealed or committed. A failed call is not resumable; callers
/// own rollback/replay and any counting or encoded-prefix work.
pub fn write(
    subject: Subject<'_>,
    proved: &Slot,
    sink: &mut impl Write,
) -> Result<PresenceStatus, Error> {
    match (subject, proved) {
        (Subject::Message(m), Slot::Message(field)) => match field {
            MessageField::PromptKind => optional(m.prompt_kind.as_ref(), sink),
            MessageField::Model => optional(m.model.as_ref(), sink),
            MessageField::At => time(m.at.as_ref(), sink),
            MessageField::CompletedAt => time(m.completed_at.as_ref(), sink),
            MessageField::Parent => match m.parent.as_ref() {
                Some(parent) => write_message_reference(parent, sink),
                None => Ok(PresenceStatus::Missing),
            },
            MessageField::FromSession => {
                optional(m.from_session.as_ref().map(|id| id.as_str()).as_ref(), sink)
            }
            MessageField::FromName => optional(m.from_name.as_ref(), sink),
            MessageField::StopReason => optional(m.stop_reason.as_ref(), sink),
            MessageField::DurationMs => optional(m.duration_ms.as_ref(), sink),
        },
        (Subject::Message(m), Slot::Usage(field)) => {
            let Some(u) = m.usage.as_ref() else {
                return Ok(PresenceStatus::Missing);
            };
            match field {
                UsageField::Input => optional(u.input.as_ref(), sink),
                UsageField::Output => optional(u.output.as_ref(), sink),
                UsageField::CacheRead => optional(u.cache_read.as_ref(), sink),
                UsageField::CacheWrite => optional(u.cache_write.as_ref(), sink),
                UsageField::CacheWrite5m => optional(u.cache_write_5m.as_ref(), sink),
                UsageField::CacheWrite1h => optional(u.cache_write_1h.as_ref(), sink),
                UsageField::Reasoning => optional(u.reasoning.as_ref(), sink),
                UsageField::ServiceTier => optional(u.service_tier.as_ref(), sink),
                UsageField::ProviderCostNusd => optional(u.provider_cost_nusd.as_ref(), sink),
            }
        }
        (Subject::Block { slot, content }, Slot::Block { block, field }) if slot == block => {
            block_leaf(content, *field, sink)
        }
        (Subject::Message(m), Slot::Extra { block: None, key }) => {
            extra(m.attrs.as_ref(), key, sink)
        }
        (
            Subject::Extras { block, values },
            Slot::Extra {
                block: requested,
                key,
            },
        ) if block == requested.as_ref() => extra(Some(values), key, sink),
        _ => Ok(PresenceStatus::Unsupported),
    }
}
fn extra(
    values: Option<&Value>,
    key: &str,
    sink: &mut impl Write,
) -> Result<PresenceStatus, Error> {
    match values {
        None => Ok(PresenceStatus::Missing),
        Some(Value::Object(map)) => optional(map.get(key), sink),
        Some(_) => Ok(PresenceStatus::Unsupported),
    }
}
fn block_leaf(
    content: &BlockContent,
    field: BlockField,
    sink: &mut impl Write,
) -> Result<PresenceStatus, Error> {
    use BlockField::*;
    match (content, field) {
        (BlockContent::Text { text }, Text) => encoded(text, sink),
        (BlockContent::Thinking(t), ThinkingText) => optional(t.text.as_ref(), sink),
        (BlockContent::Thinking(t), Encrypted) => optional(t.encrypted.as_ref(), sink),
        (BlockContent::Thinking(t), ThinkingData) => optional(t.data.as_ref(), sink),
        (BlockContent::Thinking(t), ThinkingSignature) => optional(t.signature.as_ref(), sink),
        (BlockContent::ToolCall(c), Tool) => optional(c.tool.as_ref(), sink),
        (BlockContent::ToolCall(c), CallId) => optional(c.call_id.as_ref(), sink),
        (BlockContent::ToolCall(c), Input) => optional(c.input.as_ref(), sink),
        (BlockContent::ToolCall(c), Command) => optional(c.command.as_ref(), sink),
        (BlockContent::ToolResult(r), CallId) => optional(r.call_id.as_ref(), sink),
        (BlockContent::ToolResult(r), Outcome) => match r.outcome {
            ToolOutcome::Unknown { .. } => Ok(PresenceStatus::Missing),
            _ => encoded(&r.outcome.as_str(), sink),
        },
        (BlockContent::ToolResult(r), OutcomeReason) => match &r.outcome {
            ToolOutcome::Error {
                reason: Some(reason),
            } => encoded(&Reason(reason), sink),
            _ => Ok(PresenceStatus::Missing),
        },
        (BlockContent::ToolResult(r), ExitCode) => optional(r.exit_code.as_ref(), sink),
        (BlockContent::ToolResult(r), DurationMs) => optional(r.duration_ms.as_ref(), sink),
        (BlockContent::ToolResult(r), Output) => optional(r.output.as_ref(), sink),
        (BlockContent::Image(a) | BlockContent::File(a), Name) => optional(a.name.as_ref(), sink),
        (BlockContent::Image(a) | BlockContent::File(a), MediaType) => {
            optional(a.media_type.as_ref(), sink)
        }
        (BlockContent::Image(a) | BlockContent::File(a), Size) => optional(a.size.as_ref(), sink),
        (BlockContent::Image(a) | BlockContent::File(a), Url) => optional(a.url.as_ref(), sink),
        (BlockContent::Image(a) | BlockContent::File(a), ProducedBy) => {
            optional(a.produced_by.as_ref().map(BlockRef::from).as_ref(), sink)
        }
        (BlockContent::Image(a) | BlockContent::File(a), Direction) => {
            if a.direction == crate::AttachmentDirection::Unknown {
                Ok(PresenceStatus::Missing)
            } else {
                encoded(&a.direction, sink)
            }
        }
        (BlockContent::Other { data, .. }, OtherData) => encoded(data, sink),
        _ => Ok(PresenceStatus::Unsupported),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        IdKind, MessageId, Origin, Precision, Role, SessionId, Thinking, Timestamp, ToolCall,
        ToolResult, Usage, Usd, WrittenBy,
    };
    use serde_json::json;
    fn message() -> Message {
        let session = SessionId::new(&Origin::new("synthetic").unwrap(), "s:opaque").unwrap();
        Message::new(
            MessageId::new(session, "m:opaque").unwrap(),
            IdKind::Provider,
            Role::Assistant,
            WrittenBy::Model,
        )
    }
    fn bytes(value: Projection) -> Vec<u8> {
        match value {
            Projection::Encoded(b) => b,
            _ => panic!("expected encoded leaf"),
        }
    }
    fn block(content: &BlockContent, field: BlockField, max: usize) -> Result<Projection, Error> {
        let slot = BlockSlot::Item(7);
        encode(
            Subject::Block {
                slot: &slot,
                content,
            },
            &Slot::Block {
                block: slot.clone(),
                field,
            },
            max,
        )
    }
    #[test]
    fn usage_service_tier_is_a_typed_string_and_unknown_is_omitted() {
        let mut m = message();
        m.usage = Some(Usage::default());
        let slot = Slot::Usage(UsageField::ServiceTier);
        assert!(matches!(
            encode(Subject::Message(&m), &slot, 64).unwrap(),
            Projection::Missing
        ));
        assert!(
            serde_json::to_value(m.usage.as_ref().unwrap())
                .unwrap()
                .get("service_tier")
                .is_none()
        );
        for (value, expected) in [("standard", &b"\"standard\""[..]), ("", &b"\"\""[..])] {
            m.usage.as_mut().unwrap().service_tier = Some(value.into());
            assert_eq!(
                bytes(encode(Subject::Message(&m), &slot, 64).unwrap()),
                expected
            );
            assert_eq!(
                serde_json::to_value(m.usage.as_ref().unwrap()).unwrap()["service_tier"],
                value
            );
        }
        assert!(serde_json::from_str::<Usage>(r#"{"service_tier":17}"#).is_err());
    }
    #[test]
    fn typed_time_and_integer_money_are_not_raw_source_spelling() {
        let mut m = message();
        m.at = Some(ProviderTime::Intrinsic {
            at: Timestamp(1234),
            precision: Precision::Second,
            until: Some(Timestamp(2000)),
        });
        assert_eq!(
            bytes(encode(Subject::Message(&m), &Slot::Message(MessageField::At), 200).unwrap()),
            br#"{"at":1234,"precision":"second","time_quality":"intrinsic","until":2000}"#
        );
        m.usage = Some(Usage {
            input: Some(0),
            provider_cost_nusd: Some(Usd(9007199254740993)),
            ..Usage::default()
        });
        assert_eq!(
            bytes(
                encode(
                    Subject::Message(&m),
                    &Slot::Usage(UsageField::ProviderCostNusd),
                    100
                )
                .unwrap()
            ),
            b"9007199254740993"
        );
        assert_eq!(
            bytes(encode(Subject::Message(&m), &Slot::Usage(UsageField::Input), 1).unwrap()),
            b"0"
        );
        m.at = Some(ProviderTime::Unknown);
        assert_eq!(
            encode(Subject::Message(&m), &Slot::Message(MessageField::At), 100).unwrap(),
            Projection::Missing
        );
    }
    #[test]
    fn missing_unsupported_known_null_false_and_empty_are_distinct() {
        let m = message();
        assert_eq!(
            encode(
                Subject::Message(&m),
                &Slot::Message(MessageField::Model),
                10
            )
            .unwrap(),
            Projection::Missing
        );
        let c = BlockContent::ToolCall(ToolCall {
            tool: Some(String::new()),
            call_id: None,
            input: Some(Value::Null),
            command: None,
        });
        assert_eq!(bytes(block(&c, BlockField::Input, 4).unwrap()), b"null");
        assert_eq!(bytes(block(&c, BlockField::Tool, 2).unwrap()), b"\"\"");
        assert_eq!(
            block(&c, BlockField::Output, 100).unwrap(),
            Projection::Unsupported
        );
        let t = BlockContent::Thinking(Thinking {
            encrypted: Some(false),
            ..Thinking::default()
        });
        assert_eq!(
            bytes(block(&t, BlockField::Encrypted, 5).unwrap()),
            b"false"
        );
        assert_eq!(
            block(&t, BlockField::ThinkingSignature, 32).unwrap(),
            Projection::Missing
        );
        let signed = BlockContent::Thinking(Thinking {
            signature: Some("opaque-signature".into()),
            ..Thinking::default()
        });
        assert_eq!(
            bytes(block(&signed, BlockField::ThinkingSignature, 32).unwrap()),
            b"\"opaque-signature\""
        );
        let actual = BlockSlot::Item(7);
        assert_eq!(
            encode(
                Subject::Block {
                    slot: &actual,
                    content: &c
                },
                &Slot::Block {
                    block: BlockSlot::Item(8),
                    field: BlockField::Input
                },
                100
            )
            .unwrap(),
            Projection::Unsupported
        );
    }
    #[test]
    fn extras_are_one_immediate_opaque_value_including_empty_keys() {
        let values = json!({"":null,"nested":{"z":false,"a":[]}});
        assert_eq!(
            bytes(
                encode(
                    Subject::Extras {
                        block: None,
                        values: &values
                    },
                    &Slot::Extra {
                        block: None,
                        key: String::new()
                    },
                    4
                )
                .unwrap()
            ),
            b"null"
        );
        assert_eq!(
            bytes(
                encode(
                    Subject::Extras {
                        block: None,
                        values: &values
                    },
                    &Slot::Extra {
                        block: None,
                        key: "nested".into()
                    },
                    100
                )
                .unwrap()
            ),
            br#"{"a":[],"z":false}"#
        );
        assert_eq!(
            encode(
                Subject::Extras {
                    block: None,
                    values: &values
                },
                &Slot::Extra {
                    block: None,
                    key: "a".into()
                },
                100
            )
            .unwrap(),
            Projection::Missing
        );
    }
    #[test]
    fn unknown_outcome_is_not_success_and_reason_encoding_is_capped() {
        let mut result = ToolResult {
            data: None,
            call_id: None,
            outcome: ToolOutcome::Unknown {
                reason: crate::UnknownReason::NoResult,
            },
            exit_code: None,
            duration_ms: None,
            output: None,
        };
        assert_eq!(
            block(
                &BlockContent::ToolResult(result.clone()),
                BlockField::Outcome,
                100
            )
            .unwrap(),
            Projection::Missing
        );
        result.outcome = ToolOutcome::Error {
            reason: Some(ErrorReason::Denied("quote\"\n".into())),
        };
        assert_eq!(
            bytes(
                block(
                    &BlockContent::ToolResult(result.clone()),
                    BlockField::OutcomeReason,
                    100
                )
                .unwrap()
            ),
            serde_json::to_vec("denied:quote\"\n").unwrap()
        );
        result.outcome = ToolOutcome::Error {
            reason: Some(ErrorReason::Denied("x".repeat(65536))),
        };
        assert!(matches!(
            block(
                &BlockContent::ToolResult(result),
                BlockField::OutcomeReason,
                8
            ),
            Err(Error::Admission { maximum: 8 })
        ));
    }
    #[test]
    fn opaque_reference_relations_are_borrowed_without_id_splitting() {
        let mut m = message();
        let session = SessionId::from_stored("unrelated:session:opaque").unwrap();
        let parent =
            MessageId::from_stored("opaque:parent:without:prefix", session.clone()).unwrap();
        m.parent = Some(parent.clone());
        m.from_session = Some(session.clone());
        assert_eq!(
            bytes(
                encode(
                    Subject::Message(&m),
                    &Slot::Message(MessageField::Parent),
                    256
                )
                .unwrap()
            ),
            serde_json::to_vec(&parent).unwrap()
        );
        assert_eq!(
            bytes(
                encode(
                    Subject::Message(&m),
                    &Slot::Message(MessageField::FromSession),
                    256
                )
                .unwrap()
            ),
            serde_json::to_vec(&session).unwrap()
        );
        assert!(matches!(
            encode(
                Subject::Message(&m),
                &Slot::Message(MessageField::Parent),
                1
            ),
            Err(Error::Admission { maximum: 1 })
        ));
    }
    #[test]
    fn admitted_encoding_is_exact_and_no_unrequested_body_is_scanned() {
        let mut m = message();
        m.model = Some("a\"b".into());
        // This unrequested field would exceed the allowance by many orders.
        m.attrs = Some(json!({"unrequested":"x".repeat(65536)}));
        let slot = Slot::Message(MessageField::Model);
        let expected = serde_json::to_vec("a\"b").unwrap();
        assert!(
            matches!(encode(Subject::Message(&m),&slot,expected.len()-1),Err(Error::Admission {maximum}) if maximum==expected.len()-1)
        );
        assert_eq!(
            bytes(encode(Subject::Message(&m), &slot, expected.len()).unwrap()),
            expected
        );
        assert!(matches!(
            encode(Subject::Message(&m), &slot, 0),
            Err(Error::Admission { maximum: 0 })
        ));
    }
    fn equivalent(subject: Subject<'_>, slot: &Slot) {
        let collected = encode(subject, slot, 1024 * 1024).unwrap();
        let mut streamed = Vec::new();
        let status = write(subject, slot, &mut streamed).unwrap();
        match collected {
            Projection::Encoded(expected) => {
                assert_eq!(status, PresenceStatus::Written);
                assert_eq!(streamed, expected);
            }
            Projection::Missing => {
                assert_eq!(status, PresenceStatus::Missing);
                assert!(streamed.is_empty());
            }
            Projection::Unsupported => {
                assert_eq!(status, PresenceStatus::Unsupported);
                assert!(streamed.is_empty());
            }
        }
    }
    #[test]
    fn direct_sink_matches_all_selected_typed_leaf_branches() {
        use crate::{Attachment, AttachmentDirection, BlobHash, BlockId, PromptKind, ToolOutput};
        let mut m = message();
        m.prompt_kind = Some(PromptKind::Prose);
        m.model = Some("model\"\n".into());
        m.at = Some(ProviderTime::Intrinsic {
            at: Timestamp(10),
            precision: Precision::Bounded,
            until: Some(Timestamp(20)),
        });
        m.completed_at = Some(ProviderTime::FileMtime {
            at: Timestamp(30),
            precision: Precision::Instant,
            until: None,
        });
        m.parent = Some(MessageId::from_stored("opaque:parent", m.id.session().clone()).unwrap());
        m.from_session = Some(SessionId::from_stored("opaque:other:session").unwrap());
        m.from_name = Some(String::new());
        m.stop_reason = Some("stop".into());
        m.duration_ms = Some(0);
        m.usage = Some(Usage {
            input: Some(0),
            output: Some(1),
            cache_read: Some(2),
            cache_write: Some(3),
            cache_write_5m: Some(4),
            cache_write_1h: Some(5),
            reasoning: Some(6),
            service_tier: Some("standard".into()),
            provider_cost_nusd: Some(Usd(9007199254740993)),
        });
        m.attrs = Some(json!({"null":null,"nested":{"z":false,"a":[]}}));
        for field in [
            MessageField::PromptKind,
            MessageField::Model,
            MessageField::At,
            MessageField::CompletedAt,
            MessageField::Parent,
            MessageField::FromSession,
            MessageField::FromName,
            MessageField::StopReason,
            MessageField::DurationMs,
        ] {
            equivalent(Subject::Message(&m), &Slot::Message(field));
        }
        for field in [
            UsageField::Input,
            UsageField::Output,
            UsageField::CacheRead,
            UsageField::CacheWrite,
            UsageField::CacheWrite5m,
            UsageField::CacheWrite1h,
            UsageField::Reasoning,
            UsageField::ServiceTier,
            UsageField::ProviderCostNusd,
        ] {
            equivalent(Subject::Message(&m), &Slot::Usage(field));
        }
        for key in ["null", "nested", "absent"] {
            equivalent(
                Subject::Message(&m),
                &Slot::Extra {
                    block: None,
                    key: key.into(),
                },
            );
        }
        let attachment = Attachment {
            name: Some("file".into()),
            media_type: Some("synthetic/type".into()),
            size: Some(0),
            url: Some("synthetic://opaque".into()),
            direction: AttachmentDirection::Output,
            produced_by: Some(BlockId::new(m.id.clone(), 2)),
            ..Attachment::default()
        };
        let contents = vec![
            BlockContent::Text {
                text: "escaped\"\n☃".into(),
            },
            BlockContent::Thinking(Thinking {
                text: Some(String::new()),
                signature: Some("opaque replay signature".into()),
                encrypted: Some(false),
                data: Some("opaque encrypted bytes".into()),
                ..Thinking::default()
            }),
            BlockContent::ToolCall(ToolCall {
                tool: Some(String::new()),
                call_id: Some("opaque:call".into()),
                input: Some(Value::Null),
                command: Some("command".into()),
            }),
            BlockContent::ToolResult(ToolResult {
                data: None,
                call_id: Some("opaque:call".into()),
                outcome: ToolOutcome::Error {
                    reason: Some(ErrorReason::Denied("policy".into())),
                },
                exit_code: Some(0),
                duration_ms: Some(0),
                output: Some(ToolOutput::Text(String::new())),
            }),
            BlockContent::ToolResult(ToolResult {
                data: None,
                call_id: None,
                outcome: ToolOutcome::Unknown {
                    reason: crate::UnknownReason::NoResult,
                },
                exit_code: None,
                duration_ms: None,
                output: Some(ToolOutput::Blob(BlobHash::of(b"synthetic body"))),
            }),
            BlockContent::Image(attachment),
            BlockContent::File(Attachment::default()),
            BlockContent::Other {
                type_name: "synthetic".into(),
                data: json!({"z":false,"a":[],"null":null}),
            },
        ];
        let block = BlockSlot::Reasoning(7);
        for content in &contents {
            for field in [
                BlockField::Text,
                BlockField::ThinkingText,
                BlockField::Encrypted,
                BlockField::ThinkingData,
                BlockField::ThinkingSignature,
                BlockField::Tool,
                BlockField::CallId,
                BlockField::Input,
                BlockField::Command,
                BlockField::Outcome,
                BlockField::OutcomeReason,
                BlockField::ExitCode,
                BlockField::DurationMs,
                BlockField::Output,
                BlockField::Name,
                BlockField::MediaType,
                BlockField::Size,
                BlockField::Url,
                BlockField::Direction,
                BlockField::ProducedBy,
                BlockField::OtherData,
            ] {
                equivalent(
                    Subject::Block {
                        slot: &block,
                        content,
                    },
                    &Slot::Block {
                        block: block.clone(),
                        field,
                    },
                );
            }
        }
        let extras = json!({"":null,"opaque":{"z":1,"a":[]}});
        for key in ["", "opaque", "absent"] {
            equivalent(
                Subject::Extras {
                    block: Some(&block),
                    values: &extras,
                },
                &Slot::Extra {
                    block: Some(block.clone()),
                    key: key.into(),
                },
            );
        }
        equivalent(
            Subject::Message(&message()),
            &Slot::Message(MessageField::Model),
        );
        m.at = Some(ProviderTime::Unknown);
        equivalent(Subject::Message(&m), &Slot::Message(MessageField::At));
    }
    struct Chunks<F> {
        callback: F,
        flushes: usize,
    }
    impl<F: FnMut(&[u8]) -> io::Result<()>> Write for Chunks<F> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let n = bytes.len().min(65536);
            (self.callback)(&bytes[..n])?;
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }
    #[test]
    fn large_leaf_reaches_fixed_chunk_callback_without_encoded_vec() {
        let text = "x".repeat(1024 * 1024);
        let content = BlockContent::Text { text };
        let block = BlockSlot::Item(0);
        // Independent fixture oracle only; the production direct sink has no
        // whole encoded leaf buffer, and no resumable-serde/CPU/RSS claim.
        let expected = match &content {
            BlockContent::Text { text } => serde_json::to_vec(text).unwrap(),
            _ => unreachable!(),
        };
        let mut hash = blake3::Hasher::new();
        let mut calls = 0;
        let mut accepted = 0;
        let mut sink = Chunks {
            callback: |bytes: &[u8]| {
                assert!(bytes.len() <= 65536);
                calls += 1;
                accepted += bytes.len();
                hash.update(bytes);
                Ok(())
            },
            flushes: 0,
        };
        assert_eq!(
            write(
                Subject::Block {
                    slot: &block,
                    content: &content
                },
                &Slot::Block {
                    block: block.clone(),
                    field: BlockField::Text
                },
                &mut sink
            )
            .unwrap(),
            PresenceStatus::Written
        );
        assert_eq!(sink.flushes, 0);
        drop(sink);
        assert!(calls > 16);
        assert_eq!(accepted, expected.len());
        assert_eq!(hash.finalize(), blake3::hash(&expected));
    }
    #[derive(Debug)]
    struct Injected(std::sync::Arc<()>);
    impl fmt::Display for Injected {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("synthetic callback backpressure")
        }
    }
    impl std::error::Error for Injected {}
    #[test]
    fn first_and_middle_sink_failure_preserve_original_error_and_stop_writing() {
        let content = BlockContent::Text {
            text: "x".repeat(4 * 65536),
        };
        let block = BlockSlot::Item(0);
        for fail_at in [0, 2] {
            let identity = std::sync::Arc::new(());
            let mut calls = 0;
            let mut accepted = 0;
            let mut sink = Chunks {
                callback: |bytes: &[u8]| {
                    let call = calls;
                    calls += 1;
                    if call == fail_at {
                        return Err(io::Error::new(
                            io::ErrorKind::WouldBlock,
                            Injected(identity.clone()),
                        ));
                    }
                    accepted += bytes.len();
                    Ok(())
                },
                flushes: 0,
            };
            let error = write(
                Subject::Block {
                    slot: &block,
                    content: &content,
                },
                &Slot::Block {
                    block: block.clone(),
                    field: BlockField::Text,
                },
                &mut sink,
            )
            .unwrap_err();
            assert_eq!(sink.flushes, 0);
            drop(sink);
            match error {
                Error::Io(error) => {
                    assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
                    let payload = error.get_ref().unwrap().downcast_ref::<Injected>().unwrap();
                    assert!(std::sync::Arc::ptr_eq(&payload.0, &identity));
                }
                other => panic!("lost original sink error: {other}"),
            }
            assert_eq!(calls, fail_at + 1);
            assert!(accepted < 4 * 65536 + 2);
            if fail_at == 0 {
                assert_eq!(accepted, 0);
            } else {
                assert!(accepted > 0);
            }
        }
    }
    #[test]
    fn missing_and_unsupported_never_call_or_flush_the_sink() {
        let mut calls = 0;
        let mut sink = Chunks {
            callback: |_: &[u8]| {
                calls += 1;
                Err(io::Error::other("unexpected sink access"))
            },
            flushes: 0,
        };
        let m = message();
        assert_eq!(
            write(
                Subject::Message(&m),
                &Slot::Message(MessageField::Model),
                &mut sink
            )
            .unwrap(),
            PresenceStatus::Missing
        );
        assert_eq!(
            write(
                Subject::Message(&m),
                &Slot::Block {
                    block: BlockSlot::Item(0),
                    field: BlockField::Text
                },
                &mut sink
            )
            .unwrap(),
            PresenceStatus::Unsupported
        );
        assert_eq!(sink.flushes, 0);
        drop(sink);
        assert_eq!(calls, 0);
    }
    #[test]
    fn zero_progress_sink_is_a_typed_write_zero_failure() {
        struct Zero;
        impl Write for Zero {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Ok(0)
            }
            fn flush(&mut self) -> io::Result<()> {
                panic!("encoder must not flush")
            }
        }
        let mut m = message();
        m.model = Some("model".into());
        assert!(
            matches!(write(Subject::Message(&m),&Slot::Message(MessageField::Model),&mut Zero),Err(Error::Io(error)) if error.kind()==io::ErrorKind::WriteZero)
        );
    }
}
