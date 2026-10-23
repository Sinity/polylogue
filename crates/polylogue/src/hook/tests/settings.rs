use super::*;
use std::os::unix::fs::PermissionsExt;

#[test]
fn malformed_or_conflicting_settings_refuse_without_changing_bytes_or_permissions() {
    for text in [
        "not json",
        "null",
        "[]",
        "42",
        r#"{"hooks":null}"#,
        r#"{"hooks":[]}"#,
        r#"{"hooks":{"Stop":{}}}"#,
        r#"{"hooks":{"Stop":[null]}}"#,
        r#"{"hooks":{"Stop":[{}]}}"#,
        r#"{"hooks":{"Stop":[{"hooks":{}}]}}"#,
        r#"{"hooks":{"Stop":[{"hooks":[null]}]}}"#,
        r#"{"hooks":{"Stop":[{"hooks":[{"type":"command","command":42}]}]}}"#,
        r#"{"hooks":{},"hooks":{"Stop":[]}}"#,
        r#"{"hooks":{"Stop":[],"Stop":[]}}"#,
        r#"{"future":{"keep":1,"keep":2},"hooks":{}}"#,
    ] {
        for on in [true, false] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("settings.json");
            let original = format!(" \n{text}\n ");
            std::fs::write(&path, &original).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
            let result = install(&path, on);
            assert!(result.is_err(), "on={on}, input={text}, result={result:?}");
            assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o640
            );
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        }
    }
}

#[test]
fn mixed_hook_entries_keep_other_commands_matchers_and_unknown_fields() {
    for on in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let other = json!({"type":"command","command":"notify-send done","timeout":7});
        let prompt = json!({"type":"prompt","prompt":"synthetic prompt","future":{"value":7}});
        let entry = json!({"matcher":"Bash|Read","future":{"keep":true},"hooks":[
            {"type":"command","command":"polylogue hook Stop","timeout":2},other.clone(),prompt.clone()
        ]});
        let original = json!({"model":"example","hooks":{"Stop":[entry],"UnknownFutureEvent":[]},"future":{"a":[{"key":1},{"key":2}]}});
        std::fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
        install(&path, on).unwrap();
        let once: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(
            once["hooks"]["Stop"][0],
            json!({"matcher":"Bash|Read","future":{"keep":true},"hooks":[other,prompt]})
        );
        assert_eq!(once["hooks"]["UnknownFutureEvent"], json!([]));
        assert_eq!(once["model"], original["model"]);
        assert_eq!(once["future"], original["future"]);
        install(&path, on).unwrap();
        let twice: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(once, twice);
    }
}

#[test]
fn unknown_hook_kinds_and_native_json_numbers_survive_setup() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    let original = r#"{"large":900719925474099312345,"real":1.23456789012345,"hooks":{"Stop":[{"hooks":[{"type":"future","command":"polylogue hook Stop","opaque":null}]}]}}"#;
    std::fs::write(&path, original).unwrap();
    let expected: Value = serde_json::from_str(original).unwrap();
    install(&path, true).unwrap();
    install(&path, false).unwrap();
    let actual: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(actual, expected);
}
