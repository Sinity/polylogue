//! Identity canonicalization is independent of response presentation order.
use polylogue_model::{BlobHash, Json, Timestamp};

pub(super) fn native<'a>(
    value: &'a Json,
    first: &str,
    second: &str,
) -> Result<Option<&'a str>, &'static str> {
    let mut id = None;
    for field in [first, second] {
        match value.get(field) {
            Some(Json::String(text)) if !text.is_empty() => {
                if id.is_some_and(|id| id != text) {
                    return Err("conflicting Grok native identifiers");
                }
                id = Some(text.as_str());
            }
            Some(Json::Null) | None => {}
            _ => return Err("Grok native identifier is not a nonempty string"),
        }
    }
    Ok(id)
}

pub(super) fn time(value: &Json) -> Result<Option<Timestamp>, &'static str> {
    let mut at = None;
    for field in ["create_time", "createTime"] {
        let Some(value) = value.get(field).filter(|value| !value.is_null()) else {
            continue;
        };
        let parsed = if let Some(date) = value.get("$date") {
            // Only the declared Extended JSON shapes normalize. Extra fields
            // remain uninterpreted identity evidence instead of disappearing.
            if value.as_object().is_none_or(|fields| fields.len() != 1)
                || date.as_object().is_some_and(|fields| fields.len() != 1)
            {
                return Err("unrecognized Grok creation time; no time was inferred");
            }
            if let Some(millis) = date.get("$numberLong").and_then(Json::as_str) {
                millis.parse::<i64>().ok().and_then(Timestamp::from_unix_ms)
            } else if date.is_number() {
                crate::time::epoch_millis(date)
            } else {
                crate::time::rfc3339(date)
            }
        } else {
            crate::document::time_of(value)
        };
        let Some(parsed) = parsed else {
            return Err("unrecognized Grok creation time; no time was inferred");
        };
        if at.is_some_and(|at| at != parsed) {
            return Err("conflicting Grok creation times; no time was chosen");
        }
        at = Some(parsed);
    }
    Ok(at)
}

fn canonical(value: &Json) -> Json {
    let Some(mut fields) = value.as_object().cloned() else {
        return value.clone();
    };
    let mut raw = serde_json::Map::new();
    for field in ["create_time", "createTime"] {
        if let Some(value) = fields.remove(field) {
            raw.insert(field.to_owned(), value);
        }
    }
    let timestamp = match time(value) {
        Ok(Some(at)) => serde_json::json!(["instant", at.to_string()]),
        Ok(None) if value.get("create_time").is_none() && value.get("createTime").is_none() => {
            serde_json::json!(["absent"])
        }
        Ok(None) => serde_json::json!(["null"]),
        Err(_) => serde_json::json!(["uninterpreted", raw]),
    };
    serde_json::json!([fields, timestamp])
}

// JSON object member order is not source identity, including when another
// workspace crate enables serde_json's preserve_order feature.
fn ordered(value: &Json) -> Json {
    match value {
        Json::Object(fields) => {
            let mut fields: Vec<_> = fields.iter().collect();
            fields.sort_by(|left, right| left.0.cmp(right.0));
            Json::Object(
                fields
                    .into_iter()
                    .map(|(key, value)| (key.clone(), ordered(value)))
                    .collect(),
            )
        }
        Json::Array(values) => Json::Array(values.iter().map(ordered).collect()),
        value => value.clone(),
    }
}

fn digest(value: &Json) -> String {
    BlobHash::of(ordered(value).to_string().as_bytes()).to_string()
}

fn canonical_response(entry: &Json) -> Json {
    if let Some(response) = entry.get("response") {
        let mut extra = entry.as_object().cloned().unwrap_or_default();
        extra.remove("response");
        serde_json::json!([canonical(response), extra])
    } else {
        serde_json::json!([canonical(entry), {}])
    }
}

pub(super) fn response(entry: &Json) -> String {
    digest(&canonical_response(entry))
}

pub(super) fn conversation(metadata: &Json, responses: &[Json], wrapper: &Json) -> String {
    let mut responses: Vec<_> = responses.iter().map(canonical_response).collect();
    responses.sort_by_cached_key(|value| ordered(value).to_string());
    let mut extra = wrapper.as_object().cloned().unwrap_or_default();
    extra.remove("conversation");
    extra.remove("responses");
    digest(&serde_json::json!([canonical(metadata), responses, extra]))
}
