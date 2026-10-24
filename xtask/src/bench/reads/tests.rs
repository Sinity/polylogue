use super::*;

#[test]
fn native_measurement_requires_completed_work_and_matching_population() {
    let walk = json!({"pages":2,"source_messages_verified":2,"rendered_bytes":128,"rendered_blake3":"a".repeat(64),"rss_after":{"rss_bytes":1024}});
    let good = json!({"population":{"id":POPULATION,"large_messages":2,"body_bytes":64},"repetitions":2,"answers_verified":true,"cancelled_client_recovered":true,"read":{"walks":[walk.clone(),walk.clone(),walk]}});
    assert_eq!(read_result(&good.to_string(), 2).unwrap(), good);
    for field in [
        "population",
        "answers_verified",
        "cancelled_client_recovered",
        "read",
    ] {
        let mut bad = good.clone();
        bad.as_object_mut().unwrap().remove(field);
        assert!(read_result(&bad.to_string(), 2).is_err(), "{field}");
    }
    for (field, value) in [
        ("pages", json!(0)),
        ("source_messages_verified", json!(1)),
        ("rendered_bytes", json!(64)),
        ("rendered_blake3", json!("b".repeat(64))),
        ("rss_after", json!({})),
    ] {
        let mut bad = good.clone();
        bad["read"]["walks"][1][field] = value;
        assert!(read_result(&bad.to_string(), 2).is_err(), "{field}");
    }
    assert!(read_result(&good.to_string(), 3).is_err());
}
