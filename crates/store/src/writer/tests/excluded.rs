use super::*;

#[test]
fn background_only_groups_use_the_exclusion_lookup() {
    let archive_conn = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&archive_conn).unwrap();
    archive_conn
        .execute(
            "INSERT INTO excision(id, target,created_at,commit_seq) VALUES ('remove',?1,1,1)",
            [format!("session:{}", sid("removed"))],
        )
        .unwrap();
    archive_conn
        .execute(
            "INSERT INTO excised(excision_id,kind,value) VALUES ('remove','session',?1)",
            [sid("removed").to_string()],
        )
        .unwrap();
    let mut b = batch("source", 0, 0, vec![], &[]);
    b.parse[0]
        .output
        .background
        .push(polylogue_model::BackgroundUpdate {
            session: sid("removed"),
            task_id: "task".into(),
            call_id: None,
            outcome: polylogue_model::ToolOutcome::Ok,
            output_file: None,
        });
    {
        let group = [Box::new(b)];
        let hits = matching(&group, &archive_conn).unwrap();
        assert_eq!(
            hits.sessions,
            std::collections::HashSet::from([sid("removed").to_string()])
        );
        let filtered = without_refused(&group[0], &hits);
        assert!(filtered.parse[0].output.background.is_empty());
    }
}
