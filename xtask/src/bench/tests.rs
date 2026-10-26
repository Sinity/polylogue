use super::*;

#[test]
fn first_build_refuses_ingestion_errors_or_empty_work() {
    let good = json!({"errors":0,"input_mb":8.0,"seconds":1.0,"messages_written":10,"unreadable_root_count":0});
    assert_eq!(
        first_build_result(&format!("progress\n{good}\n"), true).unwrap(),
        good
    );
    for (field, value) in [
        ("errors", json!(2)),
        ("unreadable_root_count", json!(1)),
        ("messages_written", json!(0)),
        ("input_mb", json!(0)),
        ("seconds", json!(0)),
    ] {
        let mut bad = good.clone();
        bad[field] = value;
        assert!(first_build_result(&bad.to_string(), true).is_err(), "{bad}");
    }
    let mut idle = good.clone();
    idle["messages_written"] = json!(0);
    assert_eq!(first_build_result(&idle.to_string(), false).unwrap(), idle);
    idle["errors"] = json!(1);
    assert!(first_build_result(&idle.to_string(), false).is_err());
    let mut unknown = good;
    unknown.as_object_mut().unwrap().remove("errors");
    assert!(first_build_result(&unknown.to_string(), true).is_err());
}

#[test]
fn a_pair_needs_two_labelled_builds() {
    assert_eq!(
        labelled("base=/bin/a").unwrap(),
        ("base".to_owned(), PathBuf::from("/bin/a"))
    );
    for bad in ["base", "=/bin/a", "base="] {
        assert!(labelled(bad).is_err(), "{bad}");
    }
}
