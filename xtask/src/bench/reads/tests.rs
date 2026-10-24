use super::*;

fn result(full: bool) -> Value {
    let walk = json!({"pages":2,"source_messages_verified":2,"rendered_bytes":128,"rendered_blake3":"a".repeat(64),"rss_after":{"rss_bytes":1024},"reference_slice_verified":true});
    let walks = json!([walk.clone(), walk.clone(), walk]);
    let read = if full {
        json!({"walks":walks})
    } else {
        json!({"reference":{"source_messages_verified":2,"rendered_bytes":128},"windows":{"first":{"walks":walks},"middle":{"walks":walks},"tail":{"walks":walks}}})
    };
    json!({"population":{"id":POPULATION,"large_messages":2,"body_bytes":64},"mode":if full {"full_walk"}else{"windows"},"repetitions":2,"answers_verified":true,"cancelled_client_recovered":true,"read":read})
}

#[test]
fn native_measurement_requires_completed_work_and_matching_population() {
    for full in [false, true] {
        let good = result(full);
        assert_eq!(read_result(&good.to_string(), 2, full).unwrap(), good);
        for field in [
            "population",
            "answers_verified",
            "cancelled_client_recovered",
            "read",
        ] {
            let mut bad = good.clone();
            bad.as_object_mut().unwrap().remove(field);
            assert!(read_result(&bad.to_string(), 2, full).is_err(), "{field}");
        }
        let mut bad = good.clone();
        let walks = if full {
            &mut bad["read"]["walks"]
        } else {
            &mut bad["read"]["windows"]["tail"]["walks"]
        };
        walks[1]["rendered_blake3"] = json!("b".repeat(64));
        assert!(read_result(&bad.to_string(), 2, full).is_err());
        assert!(read_result(&good.to_string(), 3, full).is_err());
        assert!(read_result(&good.to_string(), 2, !full).is_err());
    }
    let mut bad = result(false);
    bad["read"]["reference"]["source_messages_verified"] = json!(1);
    assert!(read_result(&bad.to_string(), 2, false).is_err());
    let mut bad = result(false);
    bad["read"]["windows"]["middle"] = Value::Null;
    assert!(read_result(&bad.to_string(), 2, false).is_err());
}
