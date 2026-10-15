//! Shared typed Session listing/search over the same admission and body view.
use crate::session_qualification;
use polylogue_model::{SessionKind, query::SessionQuery};
use polylogue_store::{
    Error,
    contributions::{SessionCatalog, SessionHeader, SessionStep},
    prose_search::ProseSearch,
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    time::Instant,
};
fn check(until: Instant, cancelled: &mut dyn FnMut() -> bool) -> Result<(), Error> {
    if Instant::now() >= until || cancelled() {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
fn matches(plan: &SessionQuery, h: &SessionHeader) -> bool {
    plan.origin
        .as_deref()
        .is_none_or(|o| o == h.origin.as_str())
        && plan.id.as_deref().is_none_or(|id| id == h.id.as_str())
        && match plan.kind {
            Some(kind) => kind == h.kind,
            None => plan.all_kinds || plan.id.is_some() || h.kind != SessionKind::Subagent,
        }
}
fn row(h: &SessionHeader, known: u64) -> Value {
    let mut row = json!({"id":h.id.as_str(),"origin":h.origin.as_str(),"native_id":h.native_id,"kind":h.kind,"known_messages":known});
    for (key, value) in [
        ("title", &h.title),
        ("cwd", &h.cwd),
        ("repo", &h.repo),
        ("branch", &h.branch),
        ("harness", &h.harness),
    ] {
        if let Some(value) = value {
            row[key] = json!(value);
        }
    }
    for (key, value) in [
        ("started_at", h.started_at),
        ("last_activity_at", h.last_activity_at),
    ] {
        if let Some(value) = value {
            row[key] = json!(value);
        }
    }
    row
}
pub fn execute(
    catalog: &SessionCatalog,
    plan: &SessionQuery,
    until: Instant,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<Value, Error> {
    let qualification =
        session_qualification::qualify_catalog(catalog, plan, until, 1_048_576, cancelled)?;
    let mut cursor = None;
    let mut candidates = BTreeMap::new();
    let mut metadata_bytes = 0usize;
    let mut corpus = if plan.text.is_empty() {
        None
    } else {
        Some(ProseSearch::new()?)
    };
    let mut sessions = 0u64;
    let mut pending = 0u64;
    let mut unresolved_order = 0u64;
    let mut known_messages = 0u64;
    let mut unknown_authors = 0u64;
    loop {
        check(until, cancelled)?;
        let Some((next, session)) = catalog.next(cursor.as_ref(), 65536)? else {
            break;
        };
        cursor = Some(next);
        sessions += 1;
        if sessions > 20000 {
            return Err(Error::ParserAdmission {
                resource: "query Session catalog",
                required: sessions,
            });
        }
        let header = session.header()?;
        let selected = matches(plan, header);
        let mut after = None;
        let mut known = 0u64;
        let mut last_message_at: Option<i64> = None;
        loop {
            check(until, cancelled)?;
            match session.next(after.as_ref(), 65536)? {
                SessionStep::Ready {
                    cursor, message, ..
                } => {
                    if let Some(at) = message.provider_time()? {
                        last_message_at = Some(last_message_at.map_or(at, |old| old.max(at)));
                    }
                    if message.header()?.written_by == polylogue_model::WrittenBy::Unknown {
                        unknown_authors += 1;
                    }
                    known += 1;
                    known_messages += 1;
                    if let Some(corpus) = &mut corpus {
                        corpus.add(&message, until, cancelled)?;
                    }
                    after = Some(cursor);
                }
                SessionStep::Pending { cursor, .. } => {
                    pending += 1;
                    after = Some(cursor);
                }
                SessionStep::UnresolvedOrder { .. } => {
                    unresolved_order += 1;
                    break;
                }
                SessionStep::End { .. } => break,
            }
        }
        if selected {
            let mut value = row(header, known);
            value["known_last_message_at"] = json!(last_message_at);
            metadata_bytes = metadata_bytes.saturating_add(
                serde_json::to_vec(&value)
                    .map_err(|e| Error::Invariant(e.to_string()))?
                    .len(),
            );
            admit_metadata(metadata_bytes)?;
            candidates.insert(header.id.as_str().to_owned(), value);
        }
    }
    let candidate_count = candidates.len();
    let mut matched = BTreeMap::new();
    let mut ranks = HashMap::new();
    if let Some(corpus) = &corpus {
        for hit in corpus.hits(&plan.text, until, cancelled)? {
            check(until, cancelled)?;
            if let Some(row) = candidates.get(&hit.session) {
                metadata_bytes =
                    metadata_bytes.saturating_add(hit.message.len() + hit.session.len() + 128);
                admit_metadata(metadata_bytes)?;
                let result = matched.entry(hit.session.clone()).or_insert_with(|| {
                    let mut row = row.clone();
                    row["hits"] = json!([]);
                    row
                });
                // FTS yields best rank then provider time, so the first hit is
                // the Session's rank; no newest-row candidate window exists.
                ranks
                    .entry(hit.session.clone())
                    .or_insert((hit.rank, hit.at));
                result["hits"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"message":hit.message,"session":hit.session,"rank":hit.rank}));
            }
        }
    } else {
        matched = candidates;
    }
    let count = matched.len();
    let mut rows = matched.into_values().collect::<Vec<_>>();
    rows.sort_by(|a, b| {
        if corpus.is_some() && !plan.recent {
            let ar = ranks[a["id"].as_str().unwrap()];
            let br = ranks[b["id"].as_str().unwrap()];
            ar.0.total_cmp(&br.0)
                .then_with(|| br.1.cmp(&ar.1))
                .then_with(|| b["id"].as_str().cmp(&a["id"].as_str()))
        } else {
            b["known_last_message_at"]
                .as_i64()
                .cmp(&a["known_last_message_at"].as_i64())
                .then_with(|| b["id"].as_str().cmp(&a["id"].as_str()))
        }
    });
    if let Some(limit) = plan.limit {
        rows.truncate(limit);
    }
    let mut notes = vec![];
    if sessions > 0 || known_messages > 0 || qualification.interpreted_nonempty_inputs > 0 {
        notes.push("Only the known selected Session/Message catalog was searched; Session metadata reductions, linked/subagent expansion, inherited/copy classification and complete source populations remain pending.".to_owned());
    }
    if qualification.record_errors > 0 {
        notes.push(format!(
            "{} undecodable complete records in relevant retained inputs",
            qualification.record_errors
        ));
    }
    if qualification.pending_inputs > 0 {
        notes.push(format!(
            "{} retained inputs have pending interpretation",
            qualification.pending_inputs
        ));
    }
    if qualification.unavailable_inputs > 0 {
        notes.push(format!(
            "{} retained inputs have unavailable interpretation",
            qualification.unavailable_inputs
        ));
    }
    if qualification.unresolved_inputs > 0 {
        notes.push(format!(
            "{} retained input dependency shapes remain unresolved",
            qualification.unresolved_inputs
        ));
    }
    if corpus.is_some() && unknown_authors > 0 {
        notes.push(format!("{unknown_authors} selected Messages have unknown authors and are outside the human/model prose corpus"));
    }
    if pending > 0 {
        notes.push(format!(
            "{pending} known Messages have pending selected bodies"
        ));
    }
    if unresolved_order > 0 {
        notes.push(format!(
            "{unresolved_order} Sessions have unresolved Message order"
        ));
    }
    // Revisit current removal authority after corpus/rank computation. The
    // selected snapshot stays fixed; a revoked body must not reach output.
    let mut verify = None;
    while let Some((next, session)) = catalog.next(verify.as_ref(), 65536)? {
        check(until, cancelled)?;
        verify = Some(next);
        let mut after = None;
        loop {
            check(until, cancelled)?;
            match session.next(after.as_ref(), 65536)? {
                SessionStep::Ready { cursor, .. } | SessionStep::Pending { cursor, .. } => {
                    after = Some(cursor)
                }
                SessionStep::UnresolvedOrder { .. } | SessionStep::End { .. } => break,
            }
        }
    }
    let degraded = !notes.is_empty();
    let outcome = if degraded {
        "degraded"
    } else if count == 0 {
        "empty"
    } else {
        "ok"
    };
    let mut read_as = if corpus.is_some() {
        "Find known selected Sessions with all prose terms in the same human or model Message, returning retained Message references; snippets, complete population and linked subagent expansion remain unqualified.".to_owned()
    } else {
        "List known selected Sessions by newest known selected Message time; complete retained population remains unqualified.".to_owned()
    };
    if let Some(origin) = &plan.origin {
        read_as.push_str(&format!(" Origin equals {origin}."));
    }
    if let Some(id) = &plan.id {
        read_as.push_str(&format!(" Session ID equals {id}."));
    }
    if let Some(kind) = plan.kind {
        read_as.push_str(&format!(" Session kind is {kind:?}."));
    } else if plan.all_kinds || plan.id.is_some() {
        read_as.push_str(" Include every Session kind.");
    } else {
        read_as.push_str(" Exclude separate subagent rows.");
    }
    if let Some(limit) = plan.limit {
        read_as.push_str(&format!(" Select the first {limit} ranked rows."));
    }
    Ok(
        json!({"outcome":outcome,"unit":"sessions","rows":rows,"count":count,"next_cursor":null,"up_to_date":false,
        "read_as":read_as, "selection":{"limit":plan.limit},
        "view":{"index_generation":catalog.position().generation,"index_counter":catalog.position().counter,"archive_counter":catalog.source_archive_counter()},
        "defaults":{"exclude_subagents":plan.kind.is_none()&&!plan.all_kinds&&plan.id.is_none()},
        "missing":notes,"notes":notes,"unknown":{"record_errors":qualification.record_errors,"pending_inputs":qualification.pending_inputs,"unavailable_inputs":qualification.unavailable_inputs,"unresolved_inputs":qualification.unresolved_inputs,"unknown_authors":unknown_authors,"pending_messages":pending,"unresolved_message_order":unresolved_order},
        "searched":{"known_sessions":sessions,"candidate_sessions":candidate_count,"selected_messages":known_messages,"prose_messages":corpus.as_ref().map(ProseSearch::documents),"scope":"pinned retained admission and selected body catalog"}}),
    )
}

fn admit_metadata(bytes: usize) -> Result<(), Error> {
    if bytes > 4 * 1024 * 1024 {
        Err(Error::ParserAdmission {
            resource: "query result metadata bytes",
            required: bytes as u64,
        })
    } else {
        Ok(())
    }
}
