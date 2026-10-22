use super::*;

#[test]
fn background_and_snapshot_only_groups_use_the_same_exclusion_lookup() {
    let canonical = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&canonical).unwrap();
    canonical
        .execute(
            "INSERT INTO excision(id,scope,created_at,commit_seq) VALUES ('remove',?1,1,1)",
            [format!("session:{}", sid("removed"))],
        )
        .unwrap();
    let cache = crate::excise::Refused::load(&canonical).unwrap();
    for snapshot in [false, true] {
        let mut b = batch("source", 0, 0, vec![], &[]);
        if snapshot {
            b.parse[0].snapshot = Some(polylogue_model::Snapshot {
                start: 0,
                end: 100,
                covered: vec![sid("removed"), sid("retained")],
                complete: true,
            });
        } else {
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
        }
        let group = [Box::new(b)];
        let hits = matching(&group, &cache).unwrap();
        assert_eq!(
            hits.sessions,
            std::collections::HashSet::from([sid("removed").to_string()])
        );
        let filtered = without_refused(&group[0], &hits);
        assert!(filtered.parse[0].output.background.is_empty());
        if snapshot {
            assert_eq!(
                filtered.parse[0].snapshot.as_ref().unwrap().covered,
                [sid("retained")]
            );
        }
    }
}
