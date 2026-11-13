use super::*;

fn member(position: i64, id: &str, member_index: u64) -> Confirmed {
    let mut event = input(position, id);
    let ConfirmedChange::Live(product) = &mut event.change else {
        unreachable!()
    };
    let Product::Input { rows, .. } = product.as_mut() else {
        unreachable!()
    };
    rows.observation = Some(crate::input_rows::ObservationRow {
        id: format!("observation-{id}"),
        machine: "machine".into(),
        host: "host".into(),
        path: format!("/synthetic/{id}"),
        path_bytes: Some(format!("/synthetic/{id}").into_bytes()),
        class: "first".into(),
        revision: None,
        chain: None,
        length: None,
        device: None,
        inode: None,
        size: None,
        mtime_ns: None,
        ctime_ns: None,
        polylogue_version: "synthetic".into(),
        seen_at: 1,
        details_json: None,
        container: Some(crate::InContainer {
            input: "container".into(),
            member_index,
            kind: "zip".into(),
            path_bytes: vec![b'a' + member_index as u8, 0xff],
        }),
    });
    event
}

#[test]
fn complete_positions_rebuild_container_memberships_with_their_original_observations() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    index.pragma_update(None, "foreign_keys", true).unwrap();
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(
        &crate::text_rows::TextIds::new(&index, false).unwrap(),
        &mut index,
        &resolver,
        &[
            input(1, "container"),
            member(2, "first", 0),
            member(3, "second", 1),
            barrier(3),
        ],
    )
    .unwrap();
    for (position, change, expected_first) in [
        (4, ConfirmedChange::Archived, false),
        (5, ConfirmedChange::Restored, true),
        (6, ConfirmedChange::Tombstoned, false),
    ] {
        apply(
            &crate::text_rows::TextIds::new(&index, false).unwrap(),
            &mut index,
            &resolver,
            &[
                Confirmed {
                    native: None,
                    position,
                    output_index: 0,
                    event: "event-first".into(),
                    change,
                },
                barrier(position),
            ],
        )
        .unwrap();
        let actual = index
            .prepare("SELECT c.observation_id,c.container_input,c.member_index,c.kind,c.path_bytes,o.commit_seq FROM source_observation_container c JOIN source_observation o ON o.id=c.observation_id ORDER BY c.observation_id")
            .unwrap()
            .query_map([], |row| Ok((row.get::<_,String>(0)?,row.get::<_,String>(1)?,row.get::<_,i64>(2)?,row.get::<_,String>(3)?,row.get::<_,Vec<u8>>(4)?,row.get::<_,i64>(5)?)))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        let mut expected = Vec::new();
        if expected_first {
            expected.push((
                "observation-first".into(),
                "container".into(),
                0,
                "zip".into(),
                vec![b'a', 0xff],
                2,
            ));
        }
        expected.push((
            "observation-second".into(),
            "container".into(),
            1,
            "zip".into(),
            vec![b'b', 0xff],
            3,
        ));
        assert_eq!(actual, expected);
        assert_eq!(checkpoint(&index).unwrap(), Some(position));
        assert_eq!(
            index
                .query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            0
        );
    }
}
