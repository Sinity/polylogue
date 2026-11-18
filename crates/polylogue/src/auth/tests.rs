use super::*;

#[test]
fn pkce_uses_the_rfc7636_s256_vector() {
    assert_eq!(
        challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
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
