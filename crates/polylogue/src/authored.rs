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
/// Transport loss or an unbound reply cannot establish whether an action committed.
pub fn uncertain_reply(request: &Value, kind: ErrorKind, message: &str) -> Value {
    let mut failure = Failure::new(kind, message);
    failure.retryable = true;
    failure.hint=Some("A write may have committed. Retry unchanged intent with the same request_id; never replace it merely because a reply was lost or mismatched.".into());
    json!({"outcome":"error","error":failure,"request_id":request["request_id"],"commit_state":"unknown"})
}
/// Both public writing clients bind the immutable effect to the submitted action.
/// A matching socket/JSON-RPC request number alone does not establish this.
pub fn bind_write_reply(request: &Value, mut reply: Value) -> Value {
    if reply["outcome"] == "error" {
        reply["request_id"] = request["request_id"].clone();
        return reply;
    }
    let id = request["request_id"].as_str().unwrap_or("");
    let root = request["relations"]
        .as_array()
        .and_then(|a| a.iter().find(|r| r["type"] == "revises"))
        .and_then(|r| r["ref"].as_str())
        .and_then(|r| polylogue_model::authored::note_reference(r).ok())
        .map_or(id, |(root, _)| root);
    let reference = format!("note:{root}@{id}");
    let effect_matches = matches!(reply["outcome"].as_str(), Some("ok" | "degraded"))
        && reply["commit_state"] == "committed"
        && reply["request_id"] == id
        && reply["note"] == id
        && reply["root"] == root
        && reply["ref"] == reference;
    let revision_matches = if let Some(revision) = reply.get("revision") {
        revision["id"] == id && revision["root"] == root
    } else {
        reply["outcome"] == "degraded" && reply["presentation"] == "retained_but_unpresentable"
    };
    if effect_matches && revision_matches {
        reply
    } else {
        uncertain_reply(
            request,
            ErrorKind::Io,
            "Authored reply does not bind its effect to the submitted action; commit outcome is unknown.",
        )
    }
}
/// A durable effect survives a presentation refusal. Return its safe action
/// identity with degraded qualification instead of losing the commit outcome.
pub(crate) fn write_reply(
    revision: polylogue_model::authored::Revision,
    recovery: Option<polylogue_model::RecoveryQualification>,
    until: Instant,
) -> Value {
    let mut notes = if recovery.is_some() {
        vec![polylogue_model::RecoveryQualification::NOTE]
    } else {
        vec![]
    };
    let identity = json!({"request_id":revision.id,"note":revision.id,"ref":revision.reference(),"root":revision.root,"commit_state":"committed","recovery":recovery});
    let identities = [&revision.author, &revision.machine]
        .into_iter()
        .map(String::as_str)
        .chain(revision.targets.iter().map(|t| t.reference.as_str()))
        .chain(revision.relations.iter().map(|r| r.other.as_str()));
    let presentable = identities
        .into_iter()
        .all(|id| crate::output_json::check_identity(id, until).is_ok());
    let mut reply = identity.clone();
    reply["outcome"] = json!(if recovery.is_some() { "degraded" } else { "ok" });
    reply["revision"] = json!(revision);
    reply["notes"] = json!(notes);
    reply["missing"] = json!(notes);
    // Preflight the same encoder before the first socket byte. Inheritance and
    // masking expansion can exceed the metadata response allowance even though
    // submitted request metadata was admitted.
    struct Capacity(usize);
    impl std::io::Write for Capacity {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self
                .0
                .checked_add(bytes.len())
                .ok_or_else(|| std::io::Error::other("metadata capacity"))?;
            if self.0 >= 65536 {
                return Err(std::io::Error::other("metadata capacity"));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut encoder = crate::output_json::Encoder::new(Capacity(1), until); // newline
    if presentable && encoder.value(&reply).and_then(|_| encoder.finish()).is_ok() {
        return reply;
    }
    notes.push("authored effect committed; its full immutable outcome cannot be presented within credential/metadata admission");
    let mut reply = identity;
    reply["outcome"] = json!("degraded");
    reply["presentation"] = json!("retained_but_unpresentable");
    reply["notes"] = json!(notes);
    reply["missing"] = json!(notes);
    reply["retry"] = json!(
        "Retain this request_id. Retry unchanged intent with the same ID; never create a replacement action merely because its reply could not be presented."
    );
    reply
}
pub(crate) fn family_value(
    family: Family,
    reader: &ReadHandle,
    until: Instant,
    cancel: &mut dyn FnMut() -> bool,
) -> Result<Value, Value> {
    for state in family.shown.iter().chain(&family.withdrawal_trail) {
        let revision = &state.revision;
        for identity in [&revision.author, &revision.machine]
            .into_iter()
            .map(String::as_str)
            .chain(revision.targets.iter().map(|t| t.reference.as_str()))
            .chain(revision.relations.iter().map(|r| r.other.as_str()))
        {
            crate::output_json::check_identity(identity,until).map_err(|_|fail(ErrorKind::NotYetAvailable,"authored identity requires credential masking; exact outcome cannot be presented"))?;
        }
    }
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
