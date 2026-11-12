use super::*;

#[test]
fn root_and_depth_queries_use_current_explicit_lineage_before_and_after_derivation() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for id in ["test:root", "test:fork", "test:resume", "test:reference"] {
        f.session(SessionFacts {
            id,
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    }
    f.link("test:root", "test:fork", "fork", None).unwrap();
    f.link("test:fork", "test:resume", "resume", None).unwrap();
    f.link("test:root", "test:reference", "reference", Some("mention"))
        .unwrap();
    for derived in [false, true] {
        if derived {
            f.derive().unwrap();
        }
        let pool = f.reads().unwrap();
        let reader = pool.reader().unwrap();
        for (query, expected) in [
            (
                "sessions where root:test:root | sort by id",
                vec!["test:fork", "test:resume", "test:root"],
            ),
            (
                "sessions where depth=2 | sort by depth",
                vec!["test:resume"],
            ),
        ] {
            let mut opts = options();
            opts.page_size = Some(20);
            opts.fields = vec!["id".into(), "root".into(), "depth".into()];
            let program = crate::query::compile(query, &opts).unwrap();
            let page = reader.query_page(&program, None).unwrap();
            let ids: Vec<_> = page
                .rows
                .iter()
                .map(|row| row["id"].as_str().unwrap())
                .collect();
            assert_eq!(ids, expected, "{query}, derived={derived}");
            for row in &page.rows {
                assert_eq!(row["root"], "test:root");
                assert!(row["depth"].is_number());
            }
        }
    }
    // A known cached root cannot leak through a newly conflicting parent.
    f.link("test:reference", "test:fork", "spawn", None)
        .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut opts = options();
    opts.page_size = Some(20);
    opts.fields = vec!["id".into(), "root".into(), "depth".into()];
    let program = crate::query::compile("sessions where root:test:root", &opts).unwrap();
    let page = reader.query_page(&program, None).unwrap();
    assert_eq!(page.rows.len(), 1);
    assert_eq!(page.rows[0]["id"], "test:root");
}
