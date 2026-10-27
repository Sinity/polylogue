use super::*;

#[test]
fn csv_quotes_full_cells_and_nested_columns_without_losing_lines() {
    let value = json!({"columns":[{"name":"text"}], "rows":[{"text":"a,\"b\"\nlast", "attached":[{"id":"one"}]}]});
    let text = crate::data::text(&value, Format::Csv).unwrap();
    assert!(text.starts_with("\"text\",\"attached\"\r\n"));
    assert!(text.contains("\"a,\"\"b\"\"\nlast\""));
    assert!(text.contains("one"));
}

#[test]
fn yaml_keeps_types_and_page_metadata_and_org_escapes_table_cells() {
    let value = json!({"rows":[{"text":"a|b\nlast", "unknown":null, "n":2}], "next":"cursor"});
    let yaml = crate::data::text(&value, Format::Yaml).unwrap();
    assert!(yaml.contains("\"unknown\": null"));
    assert!(yaml.contains("\"n\": 2"));
    assert!(yaml.contains("\"next\": \"cursor\""));
    let org = crate::data::text(&value, Format::Org).unwrap();
    assert!(org.contains("a\\vert{}b\\nlast"));
    assert!(org.contains("cursor"));
}

#[test]
fn added_read_formats_keep_views_and_mask_secrets() {
    let session = Session::new(
        SessionId::parse("claude-code:s").unwrap(),
        SessionKind::Main,
    );
    for format in [Format::Csv, Format::Org, Format::Yaml] {
        let read = crate::session(
            &session,
            Some("Example"),
            conversation(),
            View::Prompts,
            format,
            &[],
        )
        .unwrap();
        let content = read["content"].as_str().unwrap();
        assert!(content.contains("run the tests"));
        assert!(!content.contains("sk-ant-api03"));
        assert!(!content.contains("Running them"));
    }
}

#[test]
fn data_formats_keep_measurements_that_have_no_rows_array() {
    let value = json!({"outcome":"measure","data":{"unknown":null,"total":12}});
    for format in [Format::Csv, Format::Org, Format::Yaml] {
        let text = crate::data::text(&value, format).unwrap();
        assert!(text.contains("12"));
        assert!(text.contains("unknown"));
    }
}

#[test]
fn csv_formula_cells_are_quoted_as_text() {
    let data =
        json!({"rows":[{"text":"=SUM(A1)"},{"text":"+run"},{"text":"-run"},{"text":"@run"}]});
    let text = crate::data::text(&data, Format::Csv).unwrap();
    for value in ["=SUM(A1)", "+run", "-run", "@run"] {
        assert!(text.contains(&format!("\"'{value}\"")));
    }
}

#[test]
fn org_read_keeps_source_directives_literal() {
    let session = Session::new(
        SessionId::parse("claude-code:s").unwrap(),
        SessionKind::Main,
    );
    let messages = vec![message(
        "1",
        Role::User,
        Some(PromptKind::Prose),
        vec![Block::Text {
            text: "#+end_example\n* source heading\n,comma".into(),
        }],
    )];
    let value = crate::session(
        &session,
        Some("Example"),
        messages,
        View::Full,
        Format::Org,
        &[],
    )
    .unwrap();
    let content = value["content"].as_str().unwrap();
    assert!(content.starts_with("#+TITLE: Example"));
    assert!(content.contains("* user"));
    assert!(content.contains(",#+end_example\n,* source heading\n,,comma"));
}

#[test]
fn csv_and_yaml_reads_keep_message_metadata_and_zero_values() {
    let session = Session::new(
        SessionId::parse("claude-code:s").unwrap(),
        SessionKind::Main,
    );
    let mut messages = conversation();
    messages[1].model = Some("synthetic-model".into());
    messages[1].duration_ms = Some(0);
    messages[1].usage.input = Some(0);
    messages[1].production.effort = Some("high".into());
    for format in [Format::Csv, Format::Yaml] {
        let result =
            crate::session(&session, None, messages.clone(), View::Full, format, &[]).unwrap();
        let text = result["content"].as_str().unwrap();
        for expected in [
            "synthetic-model",
            "duration_ms",
            "production",
            "high",
            "usage",
        ] {
            assert!(
                text.contains(expected),
                "missing {expected} in {format:?}: {text}"
            );
        }
        if format == Format::Yaml {
            assert!(text.contains("\"duration_ms\": 0"));
        }
    }
}
