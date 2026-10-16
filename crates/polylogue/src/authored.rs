//! Bounded authored-note presentation over the shared archive reader.
use polylogue_model::{ErrorKind, Failure};
use polylogue_store::{
    ReadHandle,
    authored::{Family, Snapshot},
};
use serde_json::{Value, json};
use std::time::Instant;
#[cfg(test)]
mod tests;
fn fail(kind: ErrorKind, message: impl ToString) -> Value {
    json!({"outcome":"error","error":Failure::new(kind,message.to_string())})
}
pub(crate) fn store_error(error: polylogue_store::Error) -> Value {
    match error {
        polylogue_store::Error::Authored(e) => json!({"outcome":"error","error":e}),
        polylogue_store::Error::Cancelled => fail(
            ErrorKind::Timeout,
            "authored read cancelled or deadline exceeded",
        ),
        e => fail(ErrorKind::NotYetAvailable, e),
    }
}
pub(crate) fn family_value(
    family: Family,
    reader: &ReadHandle,
    until: Instant,
    cancel: &mut dyn FnMut() -> bool,
) -> Result<Value, Value> {
    let mut v = serde_json::to_value(family).map_err(|e| fail(ErrorKind::Io, e))?;
    for state in v["shown"].as_array_mut().unwrap() {
        let revision: polylogue_model::authored::Revision =
            serde_json::from_value(state["revision"].clone())
                .map_err(|e| fail(ErrorKind::Io, e))?;
        let mut statuses = vec![];
        let mut known = true;
        let mut stale = false;
        for t in &revision.targets {
            let basis =
                polylogue_engine::authored::current_basis(reader, &t.reference, until, cancel)
                    .map_err(store_error)?;
            let comparison = match (t.seen_hash, basis.seen_hash) {
                (Some(a), Some(b)) => Some(a != b),
                _ => None,
            };
            if comparison == Some(true) {
                stale = true;
            }
            if comparison.is_none() {
                known = false;
            }
            statuses.push(json!({"ref":t.reference,"attachment":basis.status,"stale":comparison,"current_basis":basis}));
        }
        if revision.targets.is_empty() {
            known = false;
        }
        state["ref"] = json!(revision.reference());
        state["target_status"] = json!(statuses);
        state["stale"] = json!(if stale {
            Some(true)
        } else if known {
            Some(false)
        } else {
            None
        });
    }
    Ok(v)
}
pub(crate) fn backlinks(
    snapshot: &Snapshot,
    reader: &ReadHandle,
    reference: &str,
    until: Instant,
    cancel: &mut dyn FnMut() -> bool,
) -> Result<Value, Value> {
    let families = snapshot.backlinks(reference).map_err(store_error)?;
    let families = families
        .into_iter()
        .map(|f| family_value(f, reader, until, cancel))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({"target":reference,"archive_counter":snapshot.archive_counter,"families":families}))
}
/// Same JSON encoder/masking policy as ordinary metadata, charged in full.
pub(crate) fn fit(value: Value, budget: u64, mcp: bool, until: Instant) -> Value {
    let mut bytes = vec![];
    let mut encoder = crate::output_json::Encoder::new(&mut bytes, until);
    if let Err(e) = encoder.value(&value).and_then(|_| encoder.finish()) {
        return fail(ErrorKind::NotYetAvailable, e);
    }
    let value: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => return fail(ErrorKind::Io, e),
    };
    let cost = if mcp {
        serde_json::to_vec(&json!({"structuredContent":&value,"content":[{"type":"text","text":crate::model_read::MCP_COMPANION}]})).unwrap().len()
    } else {
        bytes.len()
    };
    let tokens = (cost as u64).div_ceil(4);
    if bytes.len() >= 65536 || tokens > budget {
        let mut e = fail(
            ErrorKind::NotYetAvailable,
            "authored result exceeds admitted presentation; use a pinned revision or increase budget_tokens within the metadata allowance",
        );
        e["error"]["reason"] = json!("authored_presentation_admission");
        e["error"]["minimum_tokens"] = json!(tokens);
        return e;
    }
    value
}
pub(crate) fn read(
    request: &Value,
    reader: &ReadHandle,
    until: Instant,
    cancel: &mut dyn FnMut() -> bool,
) -> Value {
    let action = || -> Result<Value, Value> {
        if Instant::now() >= until || cancel() {
            return Err(fail(
                ErrorKind::Timeout,
                "authored read cancelled or deadline exceeded",
            ));
        }
        if request.as_object().is_none_or(|o| {
            o.keys().any(|k| {
                !matches!(
                    k.as_str(),
                    "command" | "reference" | "presentation" | "budget_tokens"
                )
            })
        }) {
            return Err(fail(
                ErrorKind::NotYetAvailable,
                "authored exact reads do not support continuation, views or general note queries",
            ));
        }
        let reference = request["reference"]
            .as_str()
            .ok_or_else(|| fail(ErrorKind::InvalidQuery, "missing exact note reference"))?;
        let snapshot = reader.authored_snapshot().map_err(store_error)?;
        let family = snapshot.family(reference).map_err(store_error)?;
        let family = family_value(family, reader, until, cancel)?;
        if Instant::now() >= until || cancel() {
            return Err(fail(
                ErrorKind::Timeout,
                "authored read cancelled or deadline exceeded",
            ));
        }
        let conflict = family["conflict"] == true;
        let mut notes = if snapshot.recovery.is_some() {
            vec![polylogue_model::RecoveryQualification::NOTE]
        } else {
            vec![]
        };
        if conflict {
            notes.push("authored family has conflicting current revisions");
        }
        Ok(
            json!({"outcome":if snapshot.recovery.is_some() || conflict{"degraded"}else{"ok"},"warning":crate::model_read::MCP_COMPANION,"reference":reference,"family":family,"view":{"authored_archive_counter":snapshot.archive_counter},"recovery":snapshot.recovery,"notes":notes,"missing":notes,"basis_evidence":"Supplied authored basis is evidence attributed to the action, not proof anyone read the underlying text."}),
        )
    };
    let mut action = action;
    let budget = request
        .get("budget_tokens")
        .map_or(Some(4000), Value::as_u64)
        .filter(|n| *n > 0 && *n <= 1_048_576);
    let Some(budget) = budget else {
        return fail(ErrorKind::InvalidQuery, "budget_tokens must be1..1048576");
    };
    let value = match action() {
        Ok(v) | Err(v) => v,
    };
    fit(value, budget, request["presentation"] == "mcp", until)
}
