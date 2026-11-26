//! Declared structures, inspected through the complete JSON stream. Field
//! order and a long first value do not affect recognition.
use super::*;
use polylogue_sources::document::{read_head, rewind_past_byte_order_mark};
use serde::Deserialize;
use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};
use std::collections::BTreeMap;

pub(super) const STRUCTURES: &[&str] = &[
    "ChatGPT export",
    "ChatGPT conversation",
    "Codex Cloud tasks",
    "Aider paired histories",
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
                        "docs",
                        "prompt_template",
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
    file: &mut (impl std::io::Read + std::io::Seek),
    named_origin: Option<&str>,
) -> std::io::Result<Option<&'static OriginSpec>> {
    file.rewind()?;
    let head = read_head(&mut *file)?;
    rewind_past_byte_order_mark(file)?;
    if named_origin == Some("polylogue-export") || retired_export(&mut *file)? {
        file.rewind()?;
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "retired polylogue-export input requires explicit private conversion to polylogue-bundle v2",
        ));
    }
    file.rewind()?;
    if polylogue_sources::polylogue_bundle::manifest(&mut *file).is_ok() {
        file.rewind()?;
        return Ok(polylogue_sources::origin::by_name(
            polylogue_sources::polylogue_bundle::ORIGIN,
        ));
    }
    rewind_past_byte_order_mark(file)?;
    if let Some(name) = named_origin {
        return Ok(polylogue_sources::origin::by_name(name).filter(|o| {
            importable(o) && ((o.recognize)(path, &head) || o.arrival == Arrival::Import)
        }));
    }
    if path.file_name().is_some_and(|name| name == "codex.json") {
        let mut recognized = true;
        let mut count = 0;
        let scanned = polylogue_sources::export::scan_array_or_object(
            &mut std::io::BufReader::new(&mut *file),
            |_, raw| {
                count += 1;
                let task: polylogue_model::Json = serde_json::from_slice(raw.get().as_bytes())
                    .map_err(|error| error.to_string())?;
                recognized &= polylogue_sources::codex_cloud::task_shape(&task);
                Ok(())
            },
        );
        rewind_past_byte_order_mark(file)?;
        if scanned.is_ok() && recognized && count > 0 {
            return Ok(polylogue_sources::origin::by_name("codex-cloud"));
        }
    }
    rewind_past_byte_order_mark(file)?;
    let shape = serde_json::from_reader::<_, Shape>(std::io::BufReader::new(&mut *file)).ok();
    rewind_past_byte_order_mark(file)?;
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
        && kind("conversation_id", Kind::String)
    {
        Some("chatgpt")
    } else if kind("uuid", Kind::String)
        && (kind("chat_messages", Kind::Array)
            || (claude_project_path(path)
                && (has(&["docs"]) || has(&["prompt_template"]))
                && polylogue_sources::origin::by_name("claude-ai")
                    .is_some_and(|origin| (origin.recognize)(path, &head))))
    {
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
    // Provider export names and stream signatures also identify incomplete
    // containers. Their scanner keeps unreadable input diagnostics;
    // a whole-JSON shape cannot identify JSONL or a malformed suffix.
    for name in ["chatgpt", "claude-ai", "openrouter", "otel-genai"] {
        // A standalone project needs its docs/template shape above; the
        // sidecar recognizer's path alone cannot declare neighboring objects.
        if let Some(origin) = polylogue_sources::origin::by_name(name)
            && (name != "claude-ai" || !claude_project_path(path))
            && (origin.recognize)(path, &head)
        {
            return Ok(Some(origin));
        }
    }
    // Hook spool and browser-capture files outside their own layouts are
    // read only when named with --format.
    for name in ["hermes", "claude-design"] {
        if let Some(origin) = polylogue_sources::origin::by_name(name)
            && (origin.recognize)(path, &head)
        {
            return Ok(Some(origin));
        }
    }
    // Outside a tool's root its preserved layout is the declaration.
    Ok(ORIGINS
        .iter()
        .filter(|o| o.arrival != Arrival::Import && importable(o))
        .find(|o| {
            o.locations.iter().any(|p| collect::ends_in(path, p)) && (o.recognize)(path, &head)
        }))
}

/// An origin whose files an import stores: a parsed one, or one whose
/// files are kept as bytes until it has a parser, as the live path keeps
/// them. A row-dump origin is a database read row by row, never a file.
fn importable(origin: &OriginSpec) -> bool {
    origin.support == Support::Parsed
        || (origin.support == Support::NoParser
            && matches!(origin.capture, polylogue_sources::Capture::Bytes))
}

// A standalone project path identifies that sidecar, not a whole
// account export or arbitrary neighboring files.
pub(super) fn claude_project_path(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension == "json")
        && (path.file_name().is_some_and(|name| name == "projects.json")
            || path
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|name| name == "projects"))
}

pub(super) fn refuse(report: &mut ImportReport, path: &Path, reason: impl Into<String>) {
    report.refused.push(ImportRefusal {
        path: path.to_path_buf(),
        reason: reason.into(),
        structures_tried: STRUCTURES.iter().map(|s| (*s).into()).collect(),
    });
}

// Recognizing a retired type field refuses it; it does not decode old data.
fn retired_export(reader: impl std::io::Read) -> std::io::Result<bool> {
    struct Header<'a>(&'a mut bool);
    impl<'de> Visitor<'de> for Header<'_> {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("an input header")
        }
        fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<(), M::Error> {
            while let Some(key) = map.next_key::<String>()? {
                if key == "format" {
                    *self.0 = map.next_value::<String>()? == "polylogue-export";
                    return Ok(());
                }
                map.next_value::<IgnoredAny>()?;
            }
            Ok(())
        }
    }
    let mut retired = false;
    let result = serde::Deserializer::deserialize_map(
        &mut serde_json::Deserializer::from_reader(std::io::BufReader::new(reader)),
        Header(&mut retired),
    );
    if let Err(error) = result
        && error.is_io()
    {
        return Err(error.into());
    }
    Ok(retired)
}
