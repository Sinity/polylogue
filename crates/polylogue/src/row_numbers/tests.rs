use super::*;

fn rows() -> Result<Vec<String>, String> {
    Ok(vec!["claude-code:a1".into(), "codex:b2".into()])
}

#[test]
fn positions_become_the_listed_ids() {
    assert_eq!(resolve("@2", &rows).unwrap(), "codex:b2");
    assert_eq!(
        resolve("sessions where (parent:@1) since:7d", &rows).unwrap(),
        "sessions where (parent:claude-code:a1) since:7d"
    );
    // `@` inside a word is not a position.
    assert_eq!(
        resolve("note:n1@n2 file:x@HEAD", &rows).unwrap(),
        "note:n1@n2 file:x@HEAD"
    );
    assert!(resolve("@3", &rows).unwrap_err().contains("has 2 rows"));
    assert!(
        resolve("@1.2", &rows)
            .unwrap_err()
            .contains("hit within a row")
    );
    // Text without a position never reads the rows.
    let none = || -> Result<Vec<String>, String> { Err("unread".into()) };
    assert_eq!(resolve("repo:x", &none).unwrap(), "repo:x");
}

#[test]
fn a_leading_position_moves_after_its_verb() {
    let dir = tempfile::tempdir().unwrap();
    let args: Vec<String> = ["polylogue", "read", "x"].map(String::from).to_vec();
    assert_eq!(rewrite_args(args.clone(), dir.path()).unwrap(), args);
    assert!(is_position("@12"));
    assert!(!is_position("@"));
    assert!(!is_position("@1x"));
}
