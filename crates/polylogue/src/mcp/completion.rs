//! MCP completion is an adapter over the same typed operation as HTTP and the shell.
use super::*;

pub(super) fn arguments(
    params: &Value,
) -> Result<crate::protocol::complete::Arguments, &'static str> {
    let text = params["argument"]["value"]
        .as_str()
        .ok_or("argument.value must contain the full query text")?;
    let position = match params["_meta"].get("polylogue/position") {
        None => None,
        Some(value) => Some(
            value
                .as_u64()
                .ok_or("polylogue/position must be a UTF-8 byte offset")?,
        ),
    };
    Ok(crate::protocol::complete::Arguments {
        text: text.into(),
        position,
    })
}

pub(super) fn answer(
    id: Value,
    archive: &Archive,
    arguments: crate::protocol::complete::Arguments,
    deadline_ms: Option<i64>,
    cancelled: &tokio_util::sync::CancellationToken,
    with_result_type: bool,
) -> Value {
    let object = serde_json::to_value(&arguments)
        .expect("completion arguments serialize")
        .as_object()
        .expect("completion arguments are an object")
        .clone();
    let answer = match call_with_limit(archive, "complete", object, deadline_ms, cancelled, None) {
        Ok((_, answer, _)) => answer,
        Err(message) => return error(id, -32603, &message),
    };
    if answer["outcome"] == "error" {
        let mut response = error(
            id,
            -32000,
            answer["error"]["message"]
                .as_str()
                .unwrap_or("completion failed"),
        );
        response["error"]["data"] = answer;
        return response;
    }
    let completions: crate::protocol::complete::Answer =
        match serde_json::from_value(answer["value"].clone()) {
            Ok(completions) => completions,
            Err(_) => return error(id, -32603, "invalid completion answer"),
        };
    let mut values = Vec::with_capacity(completions.len());
    for completion in &completions {
        if arguments
            .text
            .get(completion.replace.start..completion.replace.end)
            .is_none()
        {
            return error(
                id,
                -32603,
                "completion replacement is outside the query text",
            );
        }
        let mut text = arguments.text.clone();
        text.replace_range(
            completion.replace.start..completion.replace.end,
            &completion.value,
        );
        values.push(text);
    }
    let mut result = json!({
        "completion": {"total": values.len(), "values": values, "hasMore": false},
        "_meta": {"polylogue/completions": completions},
    });
    if with_result_type {
        result["resultType"] = json!("complete");
    }
    success(id, result)
}

#[cfg(test)]
mod tests;
