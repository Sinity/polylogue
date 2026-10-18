//! Imported tool calls: `session:ID` tool queries and `operation:` reads, shared
//! by the CLI and MCP.
use polylogue_store::ReadHandle;
use serde_json::{Value, json};
use std::time::Instant;
fn deadline(until: Instant) -> Result<(), polylogue_store::Error> {
    if Instant::now() >= until {
        Err(polylogue_store::Error::Invariant(
            "operation read deadline".into(),
        ))
    } else {
        Ok(())
    }
}
fn fail(reason: impl ToString) -> Value {
    fail_as(polylogue_model::ErrorKind::NotYetAvailable, reason)
}
fn fail_as(kind: polylogue_model::ErrorKind, reason: impl ToString) -> Value {
    json!({"outcome":"error","error":polylogue_model::Failure::new(kind,reason.to_string())})
}
pub fn query(
    reader: &ReadHandle,
    plan: &polylogue_model::query::ToolQuery,
    until: Instant,
) -> Result<Value, polylogue_store::Error> {
    let reader = reader.fork()?;
    let population = reader.tool_items_checked(&plan.session, || deadline(until))?;
    let mut rows = Vec::new();
    for item in &population.items {
        identities(item, until).map_err(|e| polylogue_store::Error::Invariant(e.to_string()))?;
        if plan.calls_only
            && (!item.operation.counted()
                || item.conflicting_representations
                || population.suppressed_wrappers.contains(&item.reference))
        {
            continue;
        }
        if plan.text.as_ref().is_some_and(|term| {
            item.operation
                .output
                .as_ref()
                .is_none_or(|output| !output.contains(term))
        }) {
            continue;
        }
        rows.push(json!({"id":item.reference,"ref":format!("{}@{}",item.reference,item.version_hash),"session":item.operation.session,"turn":item.operation.turn,"invocation":item.operation.invocation,"kind":item.operation.kind,"agent_class":item.operation.agent_class,"correspondence":item.operation.correspondence,"conflicting_representations":item.conflicting_representations,"result_present":item.operation.result_present,"output_present":item.operation.output.is_some(),"outcome":item.operation.outcome,"message":item.operation.message,"wrapper":item.operation.wrapper,"wrapper_overlaps":item.operation.wrapper_overlaps,"sources":item.sources,"representations":item.representations,"representations_complete":item.representations_complete,"associations":item.associations}));
        if rows.len() == plan.limit {
            break;
        }
    }
    let mut missing = vec!["general_session_coverage"];
    if rows
        .iter()
        .any(|row| row["representations_complete"] == false)
    {
        missing.push("operation_representation_history");
    }
    Ok(
        json!({"outcome":"degraded","unit":if plan.calls_only{"tool_calls"}else{"tool_items"},"count":rows.len(),"rows":rows,"population":{"total_items":null,"decoded_items":population.items.len(),"established_items":population.items.iter().filter(|i|i.operation.correspondence=="established" && !i.conflicting_representations).count(),"source_rows":population.framed_rows,"decoded_source_rows":population.decoded_rows,"unsupported_source_rows":population.unsupported_rows,"stored_representations":population.versions,"tool_calls":population.correspondence_complete.then_some(population.tool_calls),"failed":(population.correspondence_complete && population.outcomes_complete).then_some(population.failed),"outcomes_complete":population.outcomes_complete,"known_failed_items":population.failed,"known_agent_items":population.tool_calls,"correspondence_complete":population.correspondence_complete},"notes":["Only imported tool calls are listed; session coverage and tool schemas may be incomplete."],"missing":missing,"up_to_date":false}),
    )
}
pub fn read(reader: &ReadHandle, request: &Value, until: Instant) -> Value {
    if request.as_object().is_none_or(|o| {
        o.keys().any(|k| {
            !matches!(
                k.as_str(),
                "command" | "reference" | "budget_tokens" | "presentation"
            )
        })
    }) {
        return fail("unsupported operation read option");
    }
    if request.get("command").is_some_and(|v| v != "model-read")
        || request.get("presentation").is_some_and(|v| v != "mcp")
    {
        return fail("unsupported operation presentation or command");
    }
    let budget = match request.get("budget_tokens") {
        None => 4000,
        Some(v) => match v.as_u64().filter(|n| *n > 0 && *n <= 1_048_576) {
            Some(n) => n,
            None => {
                return fail_as(
                    polylogue_model::ErrorKind::InvalidQuery,
                    "budget_tokens must be 1..1048576",
                );
            }
        },
    };
    let Some(reference) = request["reference"].as_str() else {
        return fail("operation reference missing");
    };
    // A fork keeps file checks and decoded frames for this read only.
    let reader = match reader.fork() {
        Ok(reader) => reader,
        Err(error) => return fail(error),
    };
    let item = match reader.tool_item_checked(reference, || deadline(until)) {
        Ok(Some(v)) => v,
        Ok(None) => {
            return fail_as(
                polylogue_model::ErrorKind::NotFound,
                "operation or version not found",
            );
        }
        Err(e) => return fail(e),
    };
    if Instant::now() >= until {
        return fail("operation read deadline");
    }
    let mut missing = vec!["general_session_coverage"];
    if !item.representations_complete {
        missing.push("operation_representation_history");
    }
    let result = json!({"outcome":"degraded","reference":reference,"shown":[{"kind":"tool_item","selected":item.selected,"id":item.reference,"version_hash":item.version_hash,"operation":item.operation,"conflicting_representations":item.conflicting_representations,"sources":item.sources,"representations":item.representations,"representations_complete":item.representations_complete,"associations":item.associations}],"next":null,"notes":["Shows only what was imported; the session may have more history."],"missing":missing,"up_to_date":false});
    if let Err(e) = identities(&item, until) {
        return fail(e);
    }
    let mut encoded = Vec::new();
    let mut encoder = crate::output_json::Encoder::new(&mut encoded, until);
    if let Err(e) = encoder.value(&result).and_then(|_| encoder.finish()) {
        return fail(e);
    }
    let result: Value = match serde_json::from_slice(&encoded) {
        Ok(v) => v,
        Err(e) => return fail(e),
    };
    if serde_json::to_vec(&result).map_or(true, |v| v.len() as u64 > budget.saturating_mul(4)) {
        return fail("operation read exceeds the budget; raise --budget or budget_tokens");
    }
    result
}

fn identities(item: &polylogue_store::tool_items::Item, until: Instant) -> std::io::Result<()> {
    for identity in [
        &item.reference,
        &item.operation.invocation,
        &item.operation.native_session,
    ] {
        crate::output_json::check_identity(identity, until)?;
    }
    for identity in item
        .operation
        .turn
        .iter()
        .chain(item.operation.direct_anchors.iter())
        .chain(item.operation.wrapper.iter())
        .chain(item.operation.wrapper_overlaps.iter())
    {
        crate::output_json::check_identity(identity, until)?;
    }
    crate::output_json::check_reference(item.operation.session.as_str(), until)?;
    if let Some(anchor) = &item.operation.provisional_anchor {
        // A provisional anchor is a canonical source Block ID or an
        // irreversible source-record digest, never a native invocation key.
        crate::output_json::check_reference(anchor, until)?;
    }
    if let Some(message) = &item.operation.message {
        crate::output_json::check_reference(message, until)?;
    }
    for association in &item.associations {
        crate::output_json::check_reference(&association.message, until)?;
        for identity in [&association.input, &association.constructor] {
            crate::output_json::check_identity(identity, until)?;
        }
    }
    for citation in item.sources.iter().chain(&item.representations) {
        let mut identities = vec![citation.input.as_str(), citation.constructor.as_str()];
        if let polylogue_model::tool_item::ToolOrigin::History {
            schema,
            table,
            thread,
            turn,
            item,
            ..
        } = &citation.origin
        {
            identities.extend([
                schema.as_str(),
                table.as_str(),
                thread.as_str(),
                turn.as_str(),
                item.as_str(),
            ]);
        }
        for identity in identities {
            crate::output_json::check_identity(identity, until)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_operation_is_not_found_and_bad_budget_is_invalid() {
        let dir = tempfile::tempdir().unwrap();
        let writer = polylogue_store::Writer::open(&dir.path().join("archive")).unwrap();
        let reader = writer.read_handle().unwrap();
        let until = Instant::now() + std::time::Duration::from_secs(10);
        let absent = read(&reader, &json!({"reference":"operation:absent"}), until);
        assert_eq!(absent["error"]["kind"], "not_found", "{absent}");
        let budget = read(
            &reader,
            &json!({"reference":"operation:absent","budget_tokens":0}),
            until,
        );
        assert_eq!(budget["error"]["kind"], "invalid_query", "{budget}");
    }
}
