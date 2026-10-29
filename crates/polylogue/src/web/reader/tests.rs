use super::*;
#[test]
fn object_urls_forward_exact_target_and_canonical_parameters() {
    let parameters = vec![
        ("view".into(), "dialogue".into()),
        ("cursor".into(), "sealed +λ&?".into()),
    ];
    assert_eq!(
        target_url("message:test:session:message", &parameters).unwrap(),
        "/api/v1/read?target=message%3Atest%3Asession%3Amessage&view=dialogue&cursor=sealed%20%2B%CE%BB%26%3F"
    );
}
#[test]
fn a_path_target_cannot_be_overridden_by_query_parameters() {
    assert!(
        target_url(
            "session:test:session",
            &[("target".into(), "another selection".into())]
        )
        .is_err()
    );
}
