//! Operation-grain public reads shared by CLI queries and MCP.
use polylogue_store::ReadHandle;
use serde_json::{Value, json};
use std::time::Instant;
fn fail(reason: impl ToString) -> Value {
    json!({"outcome":"error","error":polylogue_model::Failure::new(polylogue_model::ErrorKind::NotYetAvailable,reason.to_string())})
}
pub fn query(
    reader: &ReadHandle,
    plan: &polylogue_model::query::ToolQuery,
    until: Instant,
) -> Result<Value, polylogue_store::Error> {
    let population = reader.tool_items(&plan.session)?;
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
        rows.push(json!({"id":item.reference,"ref":format!("{}@{}",item.reference,item.version_hash),"session":item.operation.session,"turn":item.operation.turn,"invocation":item.operation.invocation,"kind":item.operation.kind,"agent_class":item.operation.agent_class,"correspondence":item.operation.correspondence,"conflicting_representations":item.conflicting_representations,"result_present":item.operation.result_present,"output_present":item.operation.output.is_some(),"outcome":item.operation.outcome,"message":item.operation.message,"wrapper":item.operation.wrapper,"wrapper_overlaps":item.operation.wrapper_overlaps,"sources":item.sources,"representations":item.representations}));
        if rows.len() == plan.limit {
            break;
        }
    }
    Ok(
        json!({"outcome":"degraded","unit":if plan.calls_only{"tool_calls"}else{"tool_items"},"count":rows.len(),"rows":rows,"population":{"retained_items":null,"decoded_items":population.items.len(),"established_items":population.items.iter().filter(|i|i.operation.correspondence=="established" && !i.conflicting_representations).count(),"retained_source_rows":population.retained_rows,"decoded_source_rows":population.decoded_rows,"unsupported_source_rows":population.unsupported_rows,"retained_representations":population.retained_representations,"tool_calls":population.tool_calls,"failed":population.failed,"correspondence_complete":population.correspondence_complete},"notes":["Exact imported operation subset only; Session boundaries, source acquisition and general tool schemas remain incomplete."],"missing":["general_session_coverage"],"up_to_date":false}),
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
            None => return fail("invalid operation read budget"),
        },
    };
    let Some(reference) = request["reference"].as_str() else {
        return fail("operation reference missing");
    };
    let item = match reader.tool_item(reference) {
        Ok(Some(v)) => v,
        Ok(None) => return fail("operation or immutable version not found"),
        Err(e) => return fail(e),
    };
    if Instant::now() >= until {
        return fail("operation read deadline");
    }
    let result = json!({"outcome":"degraded","reference":reference,"shown":[{"kind":"tool_item","id":item.reference,"version_hash":item.version_hash,"operation":item.operation,"conflicting_representations":item.conflicting_representations,"sources":item.sources,"representations":item.representations}],"next":null,"notes":["Imported operation evidence does not certify complete Session or provider history."],"missing":["general_session_coverage"],"up_to_date":false});
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
        return fail("operation read exceeds display budget; increase --budget/budget_tokens");
    }
    result
}

fn identities(item: &polylogue_store::tool_items::Item, until: Instant) -> std::io::Result<()> {
    for identity in [
        &item.reference,
        item.operation.session.as_str(),
        &item.operation.invocation,
        &item.operation.native_session,
    ] {
        if let Err(e) = crate::output_json::check_identity(identity, until) {
            return Err(e);
        }
    }
    for identity in item
        .operation
        .turn
        .iter()
        .chain(item.operation.direct_anchors.iter())
        .chain(item.operation.wrapper.iter())
        .chain(item.operation.wrapper_overlaps.iter())
        .chain(item.operation.message.iter())
    {
        if let Err(e) = crate::output_json::check_identity(identity, until) {
            return Err(e);
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
            if let Err(e) = crate::output_json::check_identity(identity, until) {
                return Err(e);
            }
        }
    }
    Ok(())
}
