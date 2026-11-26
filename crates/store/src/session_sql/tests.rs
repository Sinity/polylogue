use super::*;

#[test]
fn path_ranges_agree_with_rust_on_generated_paths() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    let parts = ["", "/", "a", "bc", "_", "%", "é"];
    let paths: Vec<String> = parts
        .iter()
        .flat_map(|a| parts.iter().map(move |b| format!("{a}/{b}")))
        .collect();
    for directory in &paths {
        let mut values = Vec::new();
        let condition = path_under_sql("?1", directory, |value| {
            values.push(value.to_owned());
            format!("?{}", values.len() + 1)
        });
        for path in &paths {
            let actual: bool = conn
                .query_row(
                    &format!("SELECT {condition}"),
                    rusqlite::params![path, values[0], values[1], values[2]],
                    |r| r.get(0),
                )
                .unwrap();
            let column: bool = conn
                .query_row(
                    &format!("SELECT {}", path_under_column("?1", "?2")),
                    rusqlite::params![path, directory],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(
                (actual, column),
                (path_under(path, directory), actual),
                "{path:?} under {directory:?}"
            );
        }
    }
}
