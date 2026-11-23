use super::*;
#[test]
fn object_urls_forward_target_and_parameters() {
    let parameters = vec![
        ("view".into(), "dialogue".into()),
        ("cursor".into(), "sealed +λ&?".into()),
    ];
    assert_eq!(
        target_url("message:test:session:message", &parameters).unwrap(),
        "/api/v1/rows?selection=message%3Atest%3Asession%3Amessage&steps=read&view=dialogue&cursor=sealed%20%2B%CE%BB%26%3F"
    );
}
#[test]
fn a_path_target_cannot_be_overridden_by_query_parameters() {
    assert_eq!(
        target_url(
            "session:test:session",
            &[("target".into(), "another selection".into())]
        ),
        Err("the object URL already supplies selection and its read step".to_owned())
    );
}
