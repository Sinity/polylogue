use super::*;

#[test]
fn older_chains_keep_text_and_input_excisions_after_the_source_is_gone() {
    for whole in [false, true] {
        let w = ClaudeHome::new();
        w.append(
            "history.jsonl",
            &(user("old", "history", "removed historical content")
                + &user("keep", "history", "surviving sibling")),
        );
        {
            let (mut e, _lock) = w.engine(64 << 20);
            e.converge().unwrap();
            fs::write(
                w.file("history.jsonl"),
                user("new", "history", "current sibling"),
            )
            .unwrap();
            e.converge().unwrap();
            let target = if whole {
                let input = w
                    .reads()
                    .reader()
                    .unwrap()
                    .known_inputs(&e.config.machine)
                    .unwrap()
                    .into_iter()
                    .find(|i| Path::new(&i.path) == w.file("history.jsonl"))
                    .unwrap()
                    .input_id;
                format!("input:{input}")
            } else {
                "text:removed historical content".into()
            };
            excise(&mut e, &target, false);
        }
        fs::remove_file(w.file("history.jsonl")).unwrap();
        delete_index(&w.root);
        let (mut e, _lock) = w.engine(64 << 20);
        let pass = e.converge().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        if whole {
            assert!(w.texts("history").is_empty());
        } else {
            let texts = w.texts("history");
            assert!(texts.iter().any(|t| t == "surviving sibling"));
            assert!(texts.iter().any(|t| t == "current sibling"));
            assert!(
                texts
                    .iter()
                    .all(|t| !t.contains("removed historical content"))
            );
        }
    }
}
