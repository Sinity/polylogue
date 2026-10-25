use super::*;

#[test]
fn an_input_excision_removes_older_chains_and_stays_after_the_source_is_gone() {
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
        excise(&mut e, &format!("input:{input}"), false);
    }
    assert!(!stored(&w.root, "removed historical content"));
    fs::remove_file(w.file("history.jsonl")).unwrap();
    delete_index(&w.root);
    let (mut e, _lock) = w.engine(64 << 20);
    let pass = e.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert!(w.texts("history").is_empty());
}
