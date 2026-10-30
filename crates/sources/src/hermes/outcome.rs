use polylogue_model::{Block, ErrorReason, Json, ToolOutcome, UnknownReason};

fn unknown() -> ToolOutcome {
    ToolOutcome::Unknown {
        reason: UnknownReason::StatusFieldUnreadable,
    }
}

pub(super) fn result(
    content: Option<&Json>,
    call_id: Option<String>,
) -> (Vec<Block>, Option<&'static str>) {
    let raw = content.and_then(Json::as_str);
    let declared = raw.is_some_and(|text| text.starts_with("\0json:"));
    let text = raw.map(|text| text.strip_prefix("\0json:").unwrap_or(text));
    let decoded = text.and_then(|text| {
        let mut stream = serde_json::Deserializer::from_str(text).into_iter::<Json>();
        let value = stream.next()?.ok()?;
        Some((value, &text[stream.byte_offset()..]))
    });
    let (data, notes) = match decoded {
        Some((value, notes)) if value.is_object() => (value, Some(notes)),
        _ => {
            let reason = if declared {
                "json_marker_not_decoded"
            } else if raw.is_some_and(|text| text.starts_with('[')) {
                "output_not_recorded"
            } else {
                "result_without_status"
            };
            return (
                vec![Block::ToolResult {
                    reported_duration_ms: None,
                    call_id,
                    outcome: unknown(),
                    output: raw.map(str::to_owned),
                    data: content.cloned(),
                    attachments: vec![],
                }],
                Some(reason),
            );
        }
    };
    let (outcome, diagnostic) = verdict(&data);
    let output = data
        .get("output")
        .and_then(Json::as_str)
        .or_else(|| data.get("error").and_then(Json::as_str))
        .map(str::to_owned);
    let mut blocks = vec![Block::ToolResult {
        reported_duration_ms: None,
        call_id,
        outcome,
        output,
        data: Some(data),
        attachments: vec![],
    }];
    if let Some(notes) = notes.filter(|notes| !notes.is_empty()) {
        blocks.push(Block::Text {
            text: notes.to_owned(),
        });
    }
    (blocks, diagnostic)
}

fn verdict(data: &Json) -> (ToolOutcome, Option<&'static str>) {
    if let Some(error) = data.get("error").filter(|value| !value.is_null()) {
        return if error.as_str().is_some_and(|text| !text.is_empty()) {
            (ToolOutcome::Error { reason: None }, None)
        } else {
            (unknown(), Some("status_field_unreadable"))
        };
    }
    if let Some(success) = data.get("success") {
        return match success.as_bool() {
            Some(true) => (ToolOutcome::Ok, None),
            Some(false) => (ToolOutcome::Error { reason: None }, None),
            None => (unknown(), Some("status_field_unreadable")),
        };
    }
    if let Some(exit) = data.get("exit_code") {
        return match exit.as_i64() {
            Some(0) => (ToolOutcome::Ok, None),
            Some(code) => (
                ToolOutcome::Error {
                    reason: Some(ErrorReason::ExitCode(code)),
                },
                None,
            ),
            None => (unknown(), Some("status_field_unreadable")),
        };
    }
    match data.get("status").and_then(Json::as_str) {
        Some("success" | "unchanged") => (ToolOutcome::Ok, None),
        Some("error" | "not_found") => (ToolOutcome::Error { reason: None }, None),
        Some("timeout") => (
            ToolOutcome::Error {
                reason: Some(ErrorReason::TimedOut),
            },
            None,
        ),
        Some("blocked") => (
            ToolOutcome::Error {
                reason: Some(ErrorReason::Denied("blocked".into())),
            },
            None,
        ),
        Some("interrupted" | "killed") => (
            ToolOutcome::Error {
                reason: Some(ErrorReason::Interrupted),
            },
            None,
        ),
        Some("running" | "dispatched") => (
            ToolOutcome::Unknown {
                reason: UnknownReason::RunningInBackground,
            },
            Some("background_never_reported"),
        ),
        Some(_) => (unknown(), Some("unrecognized_status")),
        None => (
            unknown(),
            Some(if data.get("status").is_some() {
                "status_field_unreadable"
            } else {
                "result_without_status"
            }),
        ),
    }
}
