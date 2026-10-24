use super::*;

#[test]
fn mcp_attachment_descriptors_are_masked_and_cannot_request_reveal() {
    let world = test_daemon::start("synthetic transport fixture");
    let message = test_daemon::assets::seed(&world);
    let archive = Archive::at(&world.daemon.archive.root);
    let response = call(
        &archive,
        "attachment",
        args(json!({"message":message,"block":0})),
    )
    .unwrap();
    let descriptor: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert_eq!(descriptor["state"], "available");
    for secret in [
        "synthetic-secret-file-bytes",
        "synthetic-private-label",
        "content_base64",
    ] {
        assert!(!response.contains(secret), "{response}");
    }
    let refused = call(
        &archive,
        "attachment",
        args(json!({"message":message,"block":0,"reveal":true,"out":"synthetic.bin"})),
    )
    .unwrap_err();
    assert!(refused.starts_with("error (bad_request)"), "{refused}");
}
