use polylogue_store::archive::{
    open_read,
    writer::agreement::{Agreement, recorded},
};
use polylogue_testkit::{ClaudeHome, claude_user};
use std::{collections::BTreeMap, fs};

#[test]
fn file_rewrites_record_the_longest_shared_prefix() {
    let home = ClaudeHome::new();
    let path = home.file("agreement.jsonl");

    // Each line has 127 bytes of JSON and newline with a three-digit ID,
    // plus 897 bytes of text. The original has 200 × 1,024 = 204,800 bytes.
    let text = "a".repeat(897);
    let original: String = (0..200)
        .map(|n| claude_user(&format!("{n:03}"), "agreement", &text))
        .collect();
    assert_eq!(original.len(), 204_800);

    // Shorten to 100 lines and change byte 97 × 1,024 + 672 = 100,000,
    // inside the text of line 98. The change is 2,400 bytes from the new
    // end, so restoring the original changes the stored tail as it grows.
    let mut rewritten = original[..102_400].to_owned();
    rewritten.replace_range(100_000..100_001, "b");
    // JSON allows leading space. The earlier versions all start with `{`.
    let different = format!(" {}", &original[..1_024]);

    let (mut engine, _lock) = home.engine(1 << 20);
    for bytes in [&original, &rewritten, &original, &different] {
        fs::write(&path, bytes).unwrap();
        assert_eq!(engine.run_pass().unwrap().errors, []);
    }

    let input = home
        .read(|reader| {
            reader.known_watched_input_at("test-machine", path.as_os_str().as_encoded_bytes())
        })
        .expect("the engine stored the watched file");
    let conn = open_read(&engine.archive.archive_db()).unwrap();
    // The typed agreement reader expects the archive table under its source view name.
    conn.execute_batch("CREATE TEMP VIEW source_chain AS SELECT * FROM main.chain")
        .unwrap();
    let agreements = recorded(&conn, &input.input_id).unwrap();
    let no_agreement = Agreement {
        begins_like: None,
        agrees_to: 0,
    };
    assert_eq!(
        agreements.0,
        BTreeMap::from([
            (1, no_agreement),
            (
                2,
                Agreement {
                    begins_like: Some(1),
                    agrees_to: 100_000,
                },
            ),
            // Version 3 repeats all of version 1, the earlier version it
            // begins like for longest.
            (
                3,
                Agreement {
                    begins_like: Some(1),
                    agrees_to: 204_800,
                },
            ),
            (4, no_agreement),
        ]),
        "stored agreement for the four file versions"
    );
}
