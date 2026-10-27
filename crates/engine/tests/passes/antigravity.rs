use super::*;
use serde_json::json;

#[test]
fn antigravity_exports_and_trajectory_databases_are_read_by_the_real_loop_and_rebuild() {
    let w = ClaudeHome::new();
    let dir = w.home.join(".gemini/antigravity/conversations");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("export.json"),serde_json::to_vec(&json!({"source":"antigravity_language_server","cascadeId":"exported", "markdown":"### User Input\nMarkdown prompt\n### Planner Response\nMarkdown answer"})).unwrap()).unwrap();
    let db = dir.join("trajectory.db");
    let connection = rusqlite::Connection::open(&db).unwrap();
    connection.execute_batch("CREATE TABLE trajectory_meta(trajectory_id TEXT,cascade_id TEXT); CREATE TABLE steps(idx INTEGER,step_type TEXT,step_format TEXT,step_payload TEXT); INSERT INTO trajectory_meta VALUES('trajectory','cascade');").unwrap();
    connection
        .execute(
            "INSERT INTO steps VALUES(0,'user','v1',?1)",
            [br#"{"text":"Database prompt"}"#.as_slice()],
        )
        .unwrap();
    drop(connection);
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        let pool = w.reads();
        let reader = pool.reader().unwrap();
        let exported = reader
            .messages(&SessionId::parse("antigravity:exported").unwrap())
            .unwrap();
        assert_eq!(
            exported.iter().map(Message::text).collect::<Vec<_>>(),
            ["Markdown prompt", "Markdown answer"]
        );
        let trajectory = reader
            .messages(&SessionId::parse("antigravity:trajectory").unwrap())
            .unwrap();
        assert_eq!(
            trajectory.iter().map(Message::text).collect::<Vec<_>>(),
            ["Database prompt"]
        );
    }
    fs::remove_file(&db).unwrap();
    fs::remove_file(dir.join("export.json")).unwrap();
    delete_index(&w.root);
    let (mut engine, _lock) = w.engine(64 << 20);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let pool = w.reads();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .messages(&SessionId::parse("antigravity:trajectory").unwrap())
            .unwrap()[0]
            .text(),
        "Database prompt"
    );
    assert_eq!(
        reader
            .messages(&SessionId::parse("antigravity:exported").unwrap())
            .unwrap()
            .len(),
        2
    );
}
