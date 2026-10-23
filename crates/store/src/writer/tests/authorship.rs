use super::*;

fn publish(origin: &str, messages: Vec<Message>) -> World {
    let w = world();
    let session = SessionId::parse(&format!("{origin}:s")).unwrap();
    let rows = messages
        .into_iter()
        .enumerate()
        .map(|(i, mut message)| {
            message.id = MessageId::parse(&format!("{origin}:{}", message.id.native())).unwrap();
            message.session = session.clone();
            MessageRow {
                offset: i as u64 * 10,
                end: i as u64 * 10 + 1,
                message,
            }
        })
        .collect();
    let mut b = batch("authorship", 0, 0, rows, &["s"]);
    b.input.format = origin.into();
    b.parse[0].output.sessions = vec![Session::new(session, SessionKind::Main)];
    w.writer.send(b).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    w
}

#[test]
fn an_explicit_harness_marker_keeps_native_roles_and_harness_authorship_in_queries() {
    for origin in ["claude-code", "openai-chat"] {
        let roles = [Role::Assistant, Role::User, Role::System];
        let messages = roles
            .iter()
            .enumerate()
            .map(|(i, &role)| {
                let mut message = text_message(&format!("m{i}"), "s", role, "Harness notice", None);
                message.prompt_kind = Some(PromptKind::Harness);
                message
            })
            .collect();
        let w = publish(origin, messages);
        let reads = ReadPool::open(&w.archive).unwrap();
        let reader = reads.reader().unwrap();
        let rows = reader
            .query(
                "SELECT role,written_by,prompt_kind,model FROM current_message ORDER BY id",
                &[],
            )
            .unwrap();
        assert_eq!(rows.len(), roles.len());
        for (row, role) in rows.iter().zip(roles) {
            assert_eq!(row["role"], serde_json::to_value(role).unwrap());
            assert_eq!(row["written_by"], "harness");
            assert_eq!(row["prompt_kind"], "harness");
            assert_eq!(row["model"], Json::Null);
        }
        assert!(reader.query("SELECT model FROM current_message WHERE written_by='model' AND model IS NOT NULL", &[]).unwrap().is_empty());
        let stored = reader
            .messages(&SessionId::parse(&format!("{origin}:s")).unwrap(), false)
            .unwrap();
        assert_eq!(stored.len(), roles.len());
        for (message, role) in stored.iter().zip(roles) {
            assert_eq!(message.role, role);
            assert_eq!(message.prompt_kind, Some(PromptKind::Harness));
            assert!(message.model.is_none());
        }
    }
}

#[test]
fn ordinary_role_defaults_do_not_use_model_names_or_other_prompt_kinds_as_harness_markers() {
    let cases = [
        (Role::Assistant, None, None, "model"),
        (Role::Assistant, Some(PromptKind::Prose), None, "model"),
        (Role::Assistant, Some(PromptKind::Command), None, "model"),
        (Role::Assistant, None, Some("<synthetic>"), "model"),
        (Role::User, Some(PromptKind::Prose), None, "human"),
        (Role::User, Some(PromptKind::Command), None, "human"),
        (Role::User, Some(PromptKind::Model), None, "model"),
        (Role::User, None, None, "unknown"),
        (Role::System, Some(PromptKind::Prose), None, "harness"),
    ];
    let messages = cases
        .iter()
        .enumerate()
        .map(|(i, &(role, kind, model, _))| {
            let mut message = text_message(&format!("m{i}"), "s", role, "Ordinary material", None);
            message.prompt_kind = kind;
            message.model = model.map(str::to_owned);
            message
        })
        .collect();
    let w = publish("openai-chat", messages);
    let reads = ReadPool::open(&w.archive).unwrap();
    let rows = reads
        .reader()
        .unwrap()
        .query(
            "SELECT role,written_by,model FROM current_message ORDER BY id",
            &[],
        )
        .unwrap();
    assert_eq!(rows.len(), cases.len());
    for (row, (role, _, model, expected)) in rows.iter().zip(cases) {
        assert_eq!(row["role"], serde_json::to_value(role).unwrap());
        assert_eq!(row["written_by"], expected);
        assert_eq!(row["model"], serde_json::json!(model));
    }
}
