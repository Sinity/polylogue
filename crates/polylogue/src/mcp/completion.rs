//! MCP completion is an adapter over the same typed operation as HTTP and the shell.
use super::*;

/// The query text and the cursor: the full text is the argument's value, the
/// cursor a UTF-8 byte offset in `_meta["polylogue/position"]`.
pub(super) fn arguments(
    request: CompleteRequestParams,
    context: &RequestContext<RoleServer>,
) -> Result<crate::protocol::complete::Arguments, McpError> {
    let position = match context.meta.get("polylogue/position") {
        None => None,
        Some(value) => Some(value.as_u64().ok_or_else(|| {
            McpError::invalid_params("polylogue/position must be a UTF-8 byte offset", None)
        })?),
    };
    Ok(crate::protocol::complete::Arguments {
        text: request.argument.value,
        position,
    })
}

pub(super) async fn answer(
    archive: &Archive,
    arguments: crate::protocol::complete::Arguments,
    deadline_ms: Option<i64>,
    cancelled: &CancellationToken,
) -> Result<CompleteResult, McpError> {
    let deadline = deadline_ms.map_or(command_surface::Deadline::Default, |deadline| {
        command_surface::Deadline::Milliseconds(
            crate::daemon::execution::remaining(Some(deadline))
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX),
        )
    });
    let mut request = command_surface::RunnableRequest::new(
        "internal complete",
        serde_json::to_value(&arguments).expect("completion arguments serialize"),
        uuid::Uuid::now_v7().to_string(),
    );
    request.deadline = deadline;
    let structured = match run(archive, request, deadline_ms, cancelled, None).await {
        Ok((_, answer, _)) => answer,
        Err(message) => return Err(McpError::internal_error(message, None)),
    };
    let answer: command_surface::Answer<crate::protocol::complete::Answer> =
        serde_json::from_value(structured.clone())
            .map_err(|_| McpError::internal_error("invalid completion answer", None))?;
    let completions = crate::protocol::into_value(answer).map_err(|failure| {
        McpError::new(
            rmcp::model::ErrorCode(-32000),
            failure.message,
            Some(structured),
        )
    })?;
    let mut values = Vec::with_capacity(completions.len());
    for completion in &completions {
        if arguments
            .text
            .get(completion.replace.start..completion.replace.end)
            .is_none()
        {
            return Err(McpError::internal_error(
                "completion replacement is outside the query text",
                None,
            ));
        }
        let mut text = arguments.text.clone();
        text.replace_range(
            completion.replace.start..completion.replace.end,
            &completion.value,
        );
        values.push(text);
    }
    let result = json!({
        "completion": {"values": values, "total": values.len(), "hasMore": false},
        "_meta": {"polylogue/completions": completions},
    });
    let mut result: CompleteResult =
        serde_json::from_value(result).expect("completion result matches the transport schema");
    result.result_type = Some(rmcp::model::ResultType::COMPLETE);
    Ok(result)
}

#[cfg(test)]
mod tests;
