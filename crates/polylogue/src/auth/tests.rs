use super::*;

#[test]
fn pkce_uses_the_rfc7636_s256_vector() {
    assert_eq!(
        pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    );
}

#[test]
fn callback_refuses_denial_mismatched_state_and_duplicate_values() {
    for (query, kind) in [
        ("state=other&code=x", io::ErrorKind::PermissionDenied),
        (
            "state=expected&error=access_denied",
            io::ErrorKind::PermissionDenied,
        ),
        (
            "state=expected&code=first&code=second",
            io::ErrorKind::InvalidData,
        ),
        (
            "state=expected&state=expected&code=x",
            io::ErrorKind::PermissionDenied,
        ),
        ("state=expected", io::ErrorKind::InvalidData),
    ] {
        assert_eq!(
            callback_code(query, "expected").unwrap_err().kind(),
            kind,
            "{query}"
        );
    }
    assert_eq!(
        callback_code("state=expected&code=a%2Bb%2Fc", "expected").unwrap(),
        "a+b/c"
    );
}

#[test]
fn callback_failure_names_the_providers_reason() {
    let error = callback_code("state=expected&error=invalid_scope", "expected").unwrap_err();
    assert_eq!(
        error.to_string(),
        "Drive authorization failed: invalid_scope"
    );
}

#[test]
fn provider_reason_joins_error_and_description() {
    let body = serde_json::json!({"error": "invalid_grant", "error_description": "Bad Request"});
    assert_eq!(provider_reason(&body), ": invalid_grant (Bad Request)");
    assert_eq!(provider_reason(&serde_json::Value::Null), "");
}
