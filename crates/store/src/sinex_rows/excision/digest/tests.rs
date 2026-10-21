use super::*;
use crate::sinex_rows::Unit;

#[test]
fn digest_matches_fixed_preexisting_manifest_bytes() {
    let publication = ExcisionPublication {
        version: 1,
        replacements: vec![Replacement {
            unit: Unit {
                commit_seq: 7,
                kind: "note".into(),
                key: "a\n\"é".into(),
            },
            batch: None,
            bytes: 0,
            rows: 0,
            blobs: vec![],
        }],
        retired_batches: vec![],
        digest: "excluded".into(),
    };
    let bytes = r#"{"replacements":[{"batch":null,"blobs":[],"bytes":0,"rows":0,"unit":{"commit_seq":7,"key":"a\n\"é","kind":"note"}}],"retired_batches":[],"version":1}"#;
    assert_eq!(
        compute(&publication).unwrap(),
        format!("blake3:{}", blake3::hash(bytes.as_bytes()))
    );
}

#[test]
fn streamed_digest_matches_the_former_sorted_json_for_all_field_shapes() {
    for size in 0..80 {
        let publication = ExcisionPublication {
            version: size as u32,
            replacements: (0..size)
                .map(|n| Replacement {
                    unit: Unit {
                        commit_seq: i64::MAX - n as i64,
                        kind: ["note", "revision", "excision", "setting", "delivery"][n % 5].into(),
                        key: format!("{n}:\0\t\n\"\\é🦀"),
                    },
                    batch: (n % 2 == 0).then(|| format!("batch:{n}")),
                    bytes: u64::MAX - n as u64,
                    rows: n as u64,
                    blobs: (0..n % 4).map(|b| format!("blob:{b}")).collect(),
                })
                .collect(),
            retired_batches: (0..size % 7).map(|n| format!("old:{n}")).collect(),
            digest: "never included".into(),
        };
        let mut old = serde_json::json!({"version":publication.version,
            "replacements":publication.replacements,"retired_batches":publication.retired_batches});
        old.sort_all_objects();
        assert_eq!(
            compute(&publication).unwrap(),
            format!(
                "blake3:{}",
                blake3::hash(&serde_json::to_vec(&old).unwrap())
            ),
            "size {size}"
        );
    }
}

#[cfg(target_os = "linux")]
#[test]
fn hashing_a_large_manifest_does_not_clone_its_content() {
    fn peak() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find_map(|line| {
                line.strip_prefix("VmHWM:")
                    .map(|v| v.split_whitespace().next().unwrap().parse().unwrap())
            })
            .unwrap()
    }
    let publication = ExcisionPublication {
        version: 1,
        replacements: (0..12_000)
            .map(|n| Replacement {
                unit: Unit {
                    commit_seq: n + 1,
                    kind: "note".into(),
                    key: "x".repeat(2048),
                },
                batch: None,
                bytes: 0,
                rows: 0,
                blobs: vec![],
            })
            .collect(),
        retired_batches: vec![],
        digest: String::new(),
    };
    let before = peak();
    let hash = compute(&publication).unwrap();
    assert!(hash.starts_with("blake3:"));
    let growth = peak().saturating_sub(before);
    eprintln!("12,000 manifest replacements: hashing high-water growth {growth}KiB");
    assert!(growth < 8 * 1024, "manifest hashing grew by {growth}KiB");
}
