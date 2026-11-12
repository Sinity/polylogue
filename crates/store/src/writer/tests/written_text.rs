use super::*;

fn call(text: Option<&str>) -> Block {
    Block::ToolCall {
        call_id: Some("write".into()),
        tool: Some("Write".into()),
        action: polylogue_model::ToolAction::FileWrite,
        input: serde_json::json!({"content": "original input", "file_path": "/home/op/needle-path"}),
        written_text: text.map(str::to_owned),
        command: None,
        path: Some("/home/op/needle-path".into()),
        run_by_script: false,
    }
}

#[test]
fn written_tool_text_searches_output_and_snippets_and_retracts_without_losing_input() {
    for first_build in [true, false] {
        let w = test_archive();
        if !first_build {
            w.writer.derive().unwrap();
        }
        let message = |text| {
            let mut m = text_message("a", "s", Role::Assistant, "ordinary prose", None);
            m.blocks.push(call(text));
            m
        };
        let stored = w
            .store_reading(
                batch(
                    "i",
                    Route::Live,
                    0,
                    vec![row(0, message(Some("written needle Łódź")))],
                    &["s"],
                ),
                Some(&SOURCE_RECORD),
            )
            .unwrap();
        w.writer.derive().unwrap();
        {
            let pool = ReadPool::open(&w.archive).unwrap();
            let r = pool.reader().unwrap();
            assert_eq!(
                r.query(
                    "SELECT count(*) AS n FROM output_fts WHERE output_fts MATCH 'needle'",
                    &[]
                )
                .unwrap()[0]["n"],
                1
            );
            assert_eq!(
                r.query(
                    "SELECT count(*) AS n FROM text_fts WHERE text_fts MATCH 'needle'",
                    &[]
                )
                .unwrap()[0]["n"],
                0
            );
            let options = crate::query::Options {
                clock: query_language::time::Clock::new(
                    "2026-11-01T00:00:00Z".parse().unwrap(),
                    query_language::time::DaySetting::new(
                        jiff::tz::TimeZone::UTC,
                        jiff::civil::time(0, 0, 0, 0),
                    )
                    .unwrap(),
                ),
                origins: vec!["claude-code".into()],
                page_size: None,
                random_seed: None,
                embeddings: false,
                self_session: None,
                saved: Default::default(),
                kept: Default::default(),
                fields: vec![],
                chat_pages: vec![],
            };
            for unit in ["sessions", "messages", "tool_calls"] {
                let program =
                    crate::query::compile(&format!("{unit} where output:needle"), &options)
                        .unwrap();
                assert_eq!(r.query_count(&program).unwrap(), Some((1, 0)), "{unit}");
            }
            let program = crate::query::compile("messages where needle", &options).unwrap();
            assert_eq!(r.query_count(&program).unwrap(), Some((0, 0)));
            let block = r.query("SELECT b.pk,m.pk AS message_pk FROM block b JOIN message m ON m.pk=b.message_pk WHERE b.kind='tool_call'", &[]).unwrap();
            let decoded =
                crate::ordered_blocks::one(r.conn().unwrap(), block[0]["pk"].as_i64().unwrap())
                    .unwrap();
            assert_eq!(decoded, call(Some("written needle Łódź")));
            let snippets = r
                .snippets(
                    &[crate::search::SnippetQuery {
                        index: crate::search::SnippetIndex::Output,
                        fts: "needle".into(),
                        words: vec!["needle"],
                    }],
                    &[crate::search::Within::Message(
                        block[0]["message_pk"].as_i64().unwrap(),
                    )],
                )
                .unwrap();
            assert_eq!(snippets[0].as_ref().unwrap().text, "written needle Łódź");
        }
        let mut replacement = batch("i", Route::Live, 0, vec![row(0, message(None))], &["s"]);
        replacement.change = Change::Reparse {
            chain: stored.chain,
        };
        replacement.observation = None;
        replacement.parse[0].reading_version = "synthetic=2".into();
        w.store_reading(replacement, None).unwrap();
        w.writer.derive().unwrap();
        let pool = ReadPool::open(&w.archive).unwrap();
        let r = pool.reader().unwrap();
        assert_eq!(
            r.query(
                "SELECT count(*) AS n FROM output_fts WHERE output_fts MATCH 'needle'",
                &[]
            )
            .unwrap()[0]["n"],
            0
        );
        assert_eq!(
            r.query(
                "SELECT count(*) AS n FROM text_fts WHERE text_fts MATCH 'ordinary'",
                &[]
            )
            .unwrap()[0]["n"],
            1
        );
    }
}
