use super::*;

#[test]
fn feed_refuses_unsupported_envelopes_before_production() {
    let daemon = crate::test_daemon::start("stream acceptance source");
    let operation = "dev feed";
    let arguments = json!({});
    let original = request(operation, arguments, "feed-identity");
    let row = crate::commands::command(operation).unwrap();
    let crate::commands::Run::Feed { registration, .. } = row.run else {
        panic!("feed row")
    };
    for deadline in [
        command_surface::Deadline::Default,
        command_surface::Deadline::None,
        command_surface::Deadline::Milliseconds(73),
    ] {
        let mut allowed = original.clone();
        allowed.deadline = deadline;
        registration.admit(&allowed).unwrap();
    }
    let mut refused = Vec::new();
    let mut selected = original.clone();
    selected.cursor = Some("ordinary-page".into());
    refused.push((selected, "cursor"));
    let mut selected = original.clone();
    selected.budget_tokens = command_surface::TokenBudget::Tokens(10);
    refused.push((selected, "budget_tokens"));
    let mut selected = original.clone();
    selected.budget_tokens = command_surface::TokenBudget::Unlimited;
    refused.push((selected, "budget_tokens"));
    let mut selected = original.clone();
    selected.fresh = true;
    refused.push((selected, "fresh"));
    let mut selected = original.clone();
    selected.format = command_surface::Format::Json;
    refused.push((selected, "format"));
    let mut selected = original.clone();
    selected.request_id.clear();
    refused.push((selected, "request_id"));
    let mut selected = original.clone();
    selected.arguments = json!([]);
    refused.push((selected, "object"));
    let mut selected = original.clone();
    selected
        .arguments
        .as_object_mut()
        .unwrap()
        .insert("author".into(), json!("operator"));
    refused.push((selected, "author"));
    for (envelope, field) in refused {
        let failure = registration.admit(&envelope).unwrap_err();
        assert_eq!(failure.error.kind, command_surface::FailureKind::BadRequest);
        assert!(failure.error.message.contains(field), "{failure:?}");
        let failure = daemon.daemon.ready_call(&envelope).unwrap_err();
        assert_eq!(failure.kind, command_surface::FailureKind::BadRequest);
        assert!(failure.message.contains(field), "{failure:?}");
        let mut produced = false;
        let result = crate::client::stream::run_call(
            &daemon.daemon.archive,
            &envelope,
            Some(std::time::Duration::from_secs(5)),
            |_| {
                produced = true;
                Ok(true)
            },
        );
        let Err(crate::client::stream::Error::Answer(answer)) = result else {
            panic!("expected pre-stream refusal: {result:?}")
        };
        let command_surface::Outcome::Error(error) = answer.outcome else {
            panic!("not an error")
        };
        assert_eq!(error.kind, command_surface::FailureKind::BadRequest);
        assert!(error.message.contains(field), "{error:?}");
        assert!(!produced, "a refused envelope produced stream events");
    }
}
