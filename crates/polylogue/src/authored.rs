//! Note reads and write replies: note families, their history pages, and the
//! notes attached to a target.
use polylogue_model::{ErrorKind, Failure};
use polylogue_store::{
    ReadHandle,
    authored::{Family, PageKind, Snapshot},
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
            "note read cancelled or deadline exceeded",
        ),
        e => fail(ErrorKind::NotYetAvailable, e),
    }
}
/// After a lost or mismatched reply the note may or may not have been saved.
pub fn uncertain_reply(request: &Value, kind: ErrorKind, message: &str) -> Value {
    let mut failure = Failure::new(kind, message);
    failure.retryable = true;
    failure.hint = Some(
        "The note may have been saved. Retry the same note with the same request_id; do not send a new request_id because a reply was lost."
            .into(),
    );
    json!({"outcome":"error","error":failure,"request_id":request["request_id"],"commit_state":"unknown"})
}
/// Accept a write reply only when it names the submitted request ID, note and
/// revision. A matching transport request number is not enough.
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
        reply["outcome"] == "degraded" && reply["presentation"] == "saved_without_details"
    };
    if effect_matches && revision_matches {
        reply
    } else {
        uncertain_reply(
            request,
            ErrorKind::Io,
            "Reply does not match the submitted note; the note may or may not have been saved.",
        )
    }
}
/// A saved note whose full reply cannot be shown (a credential-like identity,
/// or more than the 64 KiB metadata limit after inheritance and masking) still
/// returns its request ID and reference, so the caller can retry safely.
pub(crate) fn write_reply(
    revision: polylogue_model::authored::Revision,
    recovery: Option<polylogue_model::RecoveryRecord>,
    until: Instant,
) -> Value {
    let mut notes = if recovery.is_some() {
        vec![polylogue_model::RecoveryRecord::NOTE]
    } else {
        vec![]
    };
    let identity = json!({"request_id":revision.id,"note":revision.id,"ref":revision.reference(),"root":revision.root,"commit_state":"committed","recovery":recovery});
    let showable = revision_identities(&revision, until).is_ok();
    let mut reply = identity.clone();
    reply["outcome"] = json!(if recovery.is_some() { "degraded" } else { "ok" });
    reply["revision"] = json!(revision);
    reply["notes"] = json!(notes);
    reply["missing"] = json!(notes);
    if showable && fits_metadata(&reply, until) {
        return reply;
    }
    notes.push(
        "note saved; its full reply needs credential masking or exceeds the 64 KiB metadata limit",
    );
    let mut reply = identity;
    reply["outcome"] = json!("degraded");
    reply["presentation"] = json!("saved_without_details");
    reply["notes"] = json!(notes);
    reply["missing"] = json!(notes);
    reply["retry"] = json!(
        "Keep this request_id and retry with it; do not create a new note because the reply could not be shown."
    );
    reply
}
/// Runs the reply encoder before the first socket byte, counting the trailing
/// newline, against the 64 KiB metadata response limit.
fn fits_metadata(reply: &Value, until: Instant) -> bool {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.saturating_add(bytes.len());
            if self.0 >= 65536 {
                return Err(std::io::Error::other("metadata limit"));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut encoder = crate::output_json::Encoder::new(Counter(1), until);
    encoder.value(reply).and_then(|_| encoder.finish()).is_ok()
}
fn view_tag(view: &str) -> Option<u8> {
    match view {
        "history" => Some(1),
        "withdrawals" => Some(2),
        "current" => Some(3),
        "annotations" => Some(4),
        _ => None,
    }
}
// Same process key as body continuations, with a separate MAC domain. Only
// hashes, counters and a page position go into the token.
fn next_call(
    snapshot: &Snapshot,
    reference: &str,
    view: &str,
    cutoff: u64,
    after: i64,
) -> Result<Value, Value> {
    let mut raw = Vec::from(*blake3::hash(reference.as_bytes()).as_bytes());
    raw.extend_from_slice(&snapshot.identity());
    raw.extend_from_slice(&cutoff.to_le_bytes());
    raw.extend_from_slice(&after.to_le_bytes());
    let tag = view_tag(view)
        .ok_or_else(|| fail(ErrorKind::InvalidQuery, "unsupported note page view"))?;
    raw.push(tag);
    let key = crate::model_read::key().map_err(|e| fail(ErrorKind::Io, e))?;
    let mut signed = b"authored-page-v1\0".to_vec();
    signed.extend_from_slice(&raw);
    let token = format!(
        "a1.{}.{}",
        crate::model_read::hex(&raw),
        blake3::keyed_hash(key, &signed).to_hex()
    );
    Ok(
        json!({"tool":"read","arguments":{"ref":reference,"view":view,"continuation":token,"budget_tokens":4000}}),
    )
}
fn position(
    snapshot: &Snapshot,
    reference: &str,
    view: &str,
    token: &str,
) -> Result<(u64, i64), Value> {
    let (raw, tag) = token
        .strip_prefix("a1.")
        .and_then(|s| s.split_once('.'))
        .ok_or_else(|| fail(ErrorKind::InvalidQuery, "invalid note continuation"))?;
    let raw = crate::model_read::unhex(raw)
        .filter(|v| v.len() == 81)
        .ok_or_else(|| fail(ErrorKind::InvalidQuery, "invalid note continuation"))?;
    let tag = crate::model_read::unhex(tag)
        .filter(|v| v.len() == 32)
        .ok_or_else(|| fail(ErrorKind::InvalidQuery, "invalid note continuation"))?;
    let key = crate::model_read::key().map_err(|e| fail(ErrorKind::Io, e))?;
    let mut signed = b"authored-page-v1\0".to_vec();
    signed.extend_from_slice(&raw);
    let tag = blake3::Hash::from_bytes(tag.try_into().unwrap());
    if blake3::keyed_hash(key, &signed) != tag
        || raw[..32] != *blake3::hash(reference.as_bytes()).as_bytes()
        || raw[32..64] != snapshot.identity()
    {
        return Err(fail(
            ErrorKind::InvalidQuery,
            "continuation is for another reference or archive, or has expired",
        ));
    }
    if Some(raw[80]) != view_tag(view) {
        return Err(fail(
            ErrorKind::InvalidQuery,
            "continuation is for another view",
        ));
    }
    let cutoff = u64::from_le_bytes(raw[64..72].try_into().unwrap());
    let after = i64::from_le_bytes(raw[72..80].try_into().unwrap());
    if cutoff > snapshot.archive_counter || after < 0 {
        return Err(fail(ErrorKind::InvalidQuery, "invalid note continuation"));
    }
    Ok((cutoff, after))
}
pub(crate) fn family_value(
    family: Family,
    snapshot: &Snapshot,
    reader: &ReadHandle,
    until: Instant,
    cancel: &mut dyn FnMut() -> bool,
) -> Result<Value, Value> {
    for state in family.shown.iter().chain(&family.withdrawal_trail) {
        let revision = &state.revision;
        revision_identities(revision, until).map_err(|_| {
            fail(
                ErrorKind::NotYetAvailable,
                "note contains a credential-like identity and cannot be shown",
            )
        })?;
    }
    let root = format!("note:{}", family.root);
    let cutoff = family.status_at_archive_counter;
    let nexts = [
        ("history_next", "history", family.history_after),
        ("withdrawals_next", "withdrawals", family.withdrawal_after),
        ("current_next", "current", family.current_after),
    ];
    let mut v = serde_json::to_value(family).map_err(|e| fail(ErrorKind::Io, e))?;
    for (field, view, after) in nexts {
        v[field] = match after {
            Some(after) => next_call(snapshot, &root, view, cutoff, after)?,
            None => Value::Null,
        };
    }
    for state in v["shown"].as_array_mut().unwrap() {
        let revision: polylogue_model::authored::Revision =
            serde_json::from_value(state["revision"].clone())
                .map_err(|e| fail(ErrorKind::Io, e))?;
        let mut statuses = vec![];
        let mut known = true;
        let mut stale = false;
        for t in &revision.targets {
            let current =
                polylogue_engine::authored::current_text(reader, &t.reference, until, cancel)
                    .map_err(store_error)?;
            let comparison = match (t.seen_hash, current.seen_hash) {
                (Some(a), Some(b)) => Some(a != b),
                _ => None,
            };
            if comparison == Some(true) {
                stale = true;
            }
            if comparison.is_none() {
                known = false;
            }
            statuses.push(json!({"ref":t.reference,"attachment":current.status,"stale":comparison,"current_hash":current}));
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
    association_page(
        snapshot,
        reader,
        reference,
        snapshot.archive_counter,
        0,
        16,
        until,
        cancel,
    )
}
fn association_page(
    snapshot: &Snapshot,
    reader: &ReadHandle,
    reference: &str,
    cutoff: u64,
    after: i64,
    limit: usize,
    until: Instant,
    cancel: &mut dyn FnMut() -> bool,
) -> Result<Value, Value> {
    let page = snapshot
        .backlinks_page(reference, cutoff, after, limit, until, cancel)
        .map_err(store_error)?;
    let next = page
        .after
        .map(|pos| next_call(snapshot, reference, "annotations", cutoff, pos))
        .transpose()?;
    let families = page
        .families
        .into_iter()
        .map(|f| family_value(f, snapshot, reader, until, cancel))
        .collect::<Result<Vec<_>, _>>()?;
    let complete = next.is_none()
        && families.iter().all(|f| {
            f["shown_current_complete"] == true && f["matching_target_revisions_complete"] != false
        });
    Ok(
        json!({"target":reference,"archive_counter":cutoff,"families":families,"complete":complete,"next":next}),
    )
}
/// Notes attached to a body read. They get at most 2 s and half the budget;
/// when they do not fit, the body is still shown and `next` pages the notes.
pub(crate) fn bounded_backlinks(
    snapshot: &Snapshot,
    reader: &ReadHandle,
    reference: &str,
    budget: u64,
    until: Instant,
    cancel: &mut dyn FnMut() -> bool,
) -> Value {
    let side_until = until.min(Instant::now() + std::time::Duration::from_secs(2));
    let value = backlinks(snapshot, reader, reference, side_until, cancel);
    if let Ok(value) = &value {
        let fitted = fit(value.clone(), budget / 2, false, until);
        if fitted["outcome"] != "error" {
            return fitted;
        }
    }
    let next = next_call(
        snapshot,
        reference,
        "annotations",
        snapshot.archive_counter,
        0,
    )
    .ok();
    json!({"target":reference,"archive_counter":snapshot.archive_counter,"families":[],"complete":false,"next":next,"unavailable":value.err(),"reason":"notes on this target did not fit this page; follow next"})
}
/// Masks like other metadata and charges the whole encoded reply to the budget.
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
            "note result exceeds budget_tokens or the 64 KiB limit; read a pinned revision or raise budget_tokens",
        );
        e["error"]["reason"] = json!("note_output_limit");
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
    let budget = request
        .get("budget_tokens")
        .map_or(Some(4000), Value::as_u64)
        .filter(|n| *n > 0 && *n <= 1_048_576);
    let Some(budget) = budget else {
        return fail(ErrorKind::InvalidQuery, "budget_tokens must be 1..1048576");
    };
    let mut action = || -> Result<Value, Value> {
        if Instant::now() >= until || cancel() {
            return Err(fail(
                ErrorKind::Timeout,
                "note read cancelled or deadline exceeded",
            ));
        }
        if request.as_object().is_none_or(|o| {
            o.keys().any(|k| {
                !matches!(
                    k.as_str(),
                    "command"
                        | "reference"
                        | "presentation"
                        | "budget_tokens"
                        | "view"
                        | "continuation"
                )
            })
        }) {
            return Err(fail(
                ErrorKind::NotYetAvailable,
                "unsupported note read field",
            ));
        }
        if request.get("command").is_some_and(|v| v != "model-read")
            || request.get("presentation").is_some_and(|v| v != "mcp")
        {
            return Err(fail(
                ErrorKind::InvalidQuery,
                "unsupported note read command or presentation",
            ));
        }
        let reference = request["reference"]
            .as_str()
            .ok_or_else(|| fail(ErrorKind::InvalidQuery, "missing reference"))?;
        let view = request
            .get("view")
            .map_or(Some("family"), Value::as_str)
            .ok_or_else(|| fail(ErrorKind::InvalidQuery, "invalid note view"))?;
        if !matches!(
            view,
            "family" | "history" | "withdrawals" | "current" | "value" | "annotations"
        ) {
            return Err(fail(ErrorKind::NotYetAvailable, "unsupported note view"));
        }
        let snapshot = reader.authored_snapshot().map_err(store_error)?;
        let (cutoff, after) = if let Some(token) = request.get("continuation") {
            position(
                &snapshot,
                reference,
                view,
                token
                    .as_str()
                    .ok_or_else(|| fail(ErrorKind::InvalidQuery, "invalid note continuation"))?,
            )?
        } else {
            (snapshot.archive_counter, 0)
        };
        let mut notes = vec![];
        if snapshot.recovery.is_some() {
            notes.push(polylogue_model::RecoveryRecord::NOTE);
        }
        if cutoff < snapshot.archive_counter {
            notes.push("status is as of this page's archive counter, not now");
        }
        let mut value = if view == "annotations" {
            let annotations = association_page(
                &snapshot, reader, reference, cutoff, after, 1, until, cancel,
            )?;
            if annotations["complete"] == false {
                notes.push("more notes on this target; follow next");
            }
            json!({"authored_notes":annotations,"next":annotations["next"]})
        } else if view == "value" {
            if request.get("continuation").is_some() {
                return Err(fail(
                    ErrorKind::InvalidQuery,
                    "value-only reads have no continuation",
                ));
            }
            let revision = snapshot.exact(reference, cutoff).map_err(store_error)?;
            revision_identities(&revision, until).map_err(|_| {
                fail(
                    ErrorKind::NotYetAvailable,
                    "note contains a credential-like identity and cannot be shown",
                )
            })?;
            notes.push("stored value only; current and withdrawn status not evaluated");
            json!({"revision":revision,"current":null,"withdrawn":null,"status_evaluated":false,"next":null})
        } else {
            let kind = match view {
                "history" => PageKind::History,
                "withdrawals" => PageKind::Withdrawals,
                "current" => PageKind::Current,
                _ => PageKind::Family,
            };
            let family = snapshot
                .family_page(reference, cutoff, kind, after, until, cancel)
                .map_err(store_error)?;
            if family.conflict {
                notes.push("note has conflicting current revisions");
            }
            if family.history_complete == Some(false)
                || family.withdrawal_trail_complete == Some(false)
                || family.shown_current_complete == Some(false)
            {
                notes.push(
                    "history, withdrawals or current revisions continue; follow the next calls",
                );
            }
            let family = family_value(family, &snapshot, reader, until, cancel)?;
            let next = match kind {
                PageKind::History => family["history_next"].clone(),
                PageKind::Withdrawals => family["withdrawals_next"].clone(),
                PageKind::Current => family["current_next"].clone(),
                _ => Value::Null,
            };
            json!({"family":family,"next":next})
        };
        if Instant::now() >= until || cancel() {
            return Err(fail(
                ErrorKind::Timeout,
                "note read cancelled or deadline exceeded",
            ));
        }
        value["outcome"] = json!(if notes.is_empty() { "ok" } else { "degraded" });
        value["warning"] = json!(crate::model_read::MCP_COMPANION);
        value["reference"] = json!(reference);
        value["view"] = json!({"kind":view,"authored_archive_counter":cutoff,"observed_archive_counter":snapshot.archive_counter});
        value["recovery"] = json!(snapshot.recovery);
        value["notes"] = json!(notes);
        value["missing"] = json!(notes);
        Ok(value)
    };
    let value = match action() {
        Ok(v) | Err(v) => v,
    };
    fit(value, budget, request["presentation"] == "mcp", until)
}

fn revision_identities(
    revision: &polylogue_model::authored::Revision,
    until: Instant,
) -> std::io::Result<()> {
    for identity in [&revision.author, &revision.machine] {
        crate::output_json::check_identity(identity, until)?;
    }
    for reference in revision
        .targets
        .iter()
        .map(|t| t.reference.as_str())
        .chain(revision.relations.iter().map(|r| r.other.as_str()))
    {
        crate::output_json::check_reference(reference, until)?;
    }
    Ok(())
}
