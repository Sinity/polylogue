//! Declared structures, inspected through the complete JSON stream. Field
//! order and a long first value do not affect recognition.
use super::*;
use serde::Deserialize;
use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};
use std::collections::BTreeMap;

pub(super) const STRUCTURES: &[&str] = &[
    "ChatGPT export",
    "ChatGPT conversation",
    "claude.ai export",
    "claude.ai conversation",
    "Grok account export",
    "Google Takeout",
    "AI Studio prompt",
    "OpenRouter room",
    "OpenTelemetry GenAI",
    "Polylogue bundle",
    "live tool layout",
    "explicit format",
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Object,
    Array,
    String,
    Other,
}
impl<'de> Deserialize<'de> for Kind {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct ValueKind;
        impl<'de> Visitor<'de> for ValueKind {
            type Value = Kind;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Kind, M::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(Kind::Object)
            }
            fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Kind, S::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(Kind::Array)
            }
            fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<Kind, E> {
                Ok(Kind::String)
            }
            fn visit_string<E: serde::de::Error>(self, _: String) -> Result<Kind, E> {
                Ok(Kind::String)
            }
            fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Kind, E> {
                Ok(Kind::Other)
            }
            fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Kind, E> {
                Ok(Kind::Other)
            }
            fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Kind, E> {
                Ok(Kind::Other)
            }
            fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Kind, E> {
                Ok(Kind::Other)
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Kind, E> {
                Ok(Kind::Other)
            }
        }
        d.deserialize_any(ValueKind)
    }
}

#[derive(Default)]
struct Shape {
    keys: BTreeMap<String, Kind>,
}
impl<'de> Deserialize<'de> for Shape {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct ShapeVisitor;
        impl<'de> Visitor<'de> for ShapeVisitor {
            type Value = Shape;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an object or array")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Shape, M::Error> {
                let mut shape = Shape::default();
                while let Some(key) = map.next_key::<String>()? {
                    if [
                        "mapping",
                        "current_node",
                        "conversation_id",
                        "id",
                        "uuid",
                        "chat_messages",
                        "chunkedPrompt",
                        "resourceSpans",
                        "version",
                        "messages",
                        "characters",
                        "title",
                        "conversations",
                    ]
                    .contains(&key.as_str())
                    {
                        shape.keys.insert(key, map.next_value::<Kind>()?);
                    } else {
                        map.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(shape)
            }
            fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Shape, S::Error> {
                let shape = seq.next_element::<Shape>()?.unwrap_or_default();
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(shape)
            }
        }
        d.deserialize_any(ShapeVisitor)
    }
}

pub(super) fn declared(
    path: &Path,
    file: &mut std::fs::File,
    format: Option<&str>,
) -> std::io::Result<Option<&'static OriginSpec>> {
    use std::io::{Read, Seek};
    file.rewind()?;
    let mut head = [0; 64];
    let n = file.read(&mut head)?;
    file.rewind()?;
    if let Some(format) = format {
        return Ok(polylogue_sources::origin::by_name(format).filter(|o| {
            o.support == Support::Parsed
                && ((o.recognize)(path, &head[..n]) || o.arrival == Arrival::Import)
        }));
    }
    if polylogue_sources::polylogue_export::recognize_reader(&mut *file)? {
        file.rewind()?;
        return Ok(polylogue_sources::origin::by_name("polylogue-export"));
    }
    file.rewind()?;
    let shape = serde_json::from_reader::<_, Shape>(std::io::BufReader::new(&mut *file)).ok();
    file.rewind()?;
    let has = |keys: &[&str]| {
        shape
            .as_ref()
            .is_some_and(|s| keys.iter().all(|k| s.keys.contains_key(*k)))
    };
    let kind = |key: &str, kind: Kind| {
        shape
            .as_ref()
            .is_some_and(|s| s.keys.get(key) == Some(&kind))
    };
    let name = if kind("mapping", Kind::Object)
        && has(&["current_node"])
        && (kind("conversation_id", Kind::String) || kind("id", Kind::String))
    {
        Some("chatgpt")
    } else if kind("uuid", Kind::String) && kind("chat_messages", Kind::Array) {
        Some("claude-ai")
    } else if kind("chunkedPrompt", Kind::Object) {
        Some("ai-studio")
    } else if kind("resourceSpans", Kind::Array) {
        Some("otel-genai")
    } else if has(&["version"])
        && kind("messages", Kind::Object)
        && (has(&["id"])
            || has(&["characters"])
            || path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("OpenRouter Chat")))
    {
        Some("openrouter")
    } else if path
        .file_name()
        .is_some_and(|n| n == "prod-grok-backend.json")
        && kind("conversations", Kind::Array)
    {
        Some("grok")
    } else {
        None
    };
    if let Some(name) = name {
        return Ok(polylogue_sources::origin::by_name(name));
    }
    for name in ["browser-capture", "hook-spool", "hermes", "claude-design"] {
        if let Some(origin) = polylogue_sources::origin::by_name(name) {
            if (origin.recognize)(path, &head[..n]) {
                return Ok(Some(origin));
            }
        }
    }
    // Outside a tool's root its preserved layout is the declaration.
    Ok(ORIGINS
        .iter()
        .filter(|o| o.arrival != Arrival::Import && o.support == Support::Parsed)
        .find(|o| {
            o.locations.iter().any(|p| collect::ends_in(path, p)) && (o.recognize)(path, &head[..n])
        }))
}

pub(super) fn refuse(report: &mut ImportReport, path: &Path, reason: impl Into<String>) {
    report.refused.push(ImportRefusal {
        path: path.to_path_buf(),
        reason: reason.into(),
        structures_tried: STRUCTURES.iter().map(|s| (*s).into()).collect(),
    });
}
