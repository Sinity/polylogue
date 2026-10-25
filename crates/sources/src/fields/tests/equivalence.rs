use super::*;
use std::time::Instant;

// The original scanner is the reference, including literal path spelling.
fn reference(value: &Json, rules: &[(&str, Rule)], checks: &mut usize) -> Vec<String> {
    fn deeper(path: &str, rules: &[(&str, Rule)], checks: &mut usize) -> bool {
        rules.iter().any(|(p, _)| {
            *checks += 1;
            p.len() > path.len()
                && p.starts_with(path)
                && matches!(p.as_bytes()[path.len()], b'.' | b'[')
        })
    }
    fn scan(
        value: &Json,
        path: &str,
        rules: &[(&str, Rule)],
        out: &mut Vec<String>,
        checks: &mut usize,
    ) {
        match value {
            Json::Object(map) => {
                for (key, child) in map {
                    let child_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    let declared = rules.iter().any(|(p, _)| {
                        *checks += 1;
                        *p == child_path
                    });
                    if declared {
                        if deeper(&child_path, rules, checks) {
                            scan(child, &child_path, rules, out, checks);
                        }
                    } else if deeper(&child_path, rules, checks) {
                        scan(child, &child_path, rules, out, checks);
                    } else {
                        out.push(child_path);
                    }
                }
            }
            Json::Array(items) => {
                let element = format!("{path}[]");
                if deeper(&element, rules, checks) {
                    for item in items {
                        scan(item, &element, rules, out, checks);
                    }
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    scan(value, "", rules, &mut out, checks);
    out
}

#[test]
fn compiled_paths_match_the_original_scanner_for_nested_and_literal_keys() {
    const PATHS: &[&str] = &[
        "a", "a.b", "a[].x", "a[]", "ab", "a.b[].x", "a[odd", "a[", "[].x", "☺.x", "", "a[][].x",
    ];
    let cases = [
        json!({"a":{"b":1,"x":2,"other":null},"ab":{"y":[]},"unknown":{"x":1}}),
        json!({"a":[{"x":1,"unknown":2},{"x":null,"unknown":3},[{"x":4}]],"ab":false}),
        json!({"a.b":[{"x":1,"u":2}],"a":{"b":[{"x":3,"u":4}]}}),
        json!({"a[]":{"x":1,"u":2},"a[odd":1,"a[":{"x":2},"a":{"b.c":4}}),
        json!([{"x":1,"y":2},{"x":3,"y":4}]),
        json!({"☺":{"x":"yes","extra":[]},"☺.x":false,"":null}),
        json!({"a":[],"ab":{},"a.b":null}),
        json!(null),
    ];
    for mask in 0u32..1 << PATHS.len() {
        let mut rules = PATHS
            .iter()
            .enumerate()
            .filter(|(n, _)| mask & (1 << n) != 0)
            .map(|(n, path)| {
                (
                    *path,
                    match n % 3 {
                        0 => Rule::Read,
                        1 => Rule::Kept("synthetic"),
                        _ => Rule::Excluded("synthetic"),
                    },
                )
            })
            .collect::<Vec<_>>();
        if let Some(first) = rules.first().copied() {
            rules.push((first.0, Rule::Excluded("duplicate")));
        }
        let fields = Fields::new(Box::leak(rules.into_boxed_slice()));
        for value in &cases {
            assert_eq!(
                unaccounted(value, &fields),
                reference(value, fields.rules, &mut 0),
                "mask={mask},value={value}"
            );
        }
    }
}

#[test]
fn parsing_threads_share_one_compiled_declaration() {
    static SHARED: Fields = Fields::new(&[("a[].x", Rule::Read), ("whole", Rule::Kept("opaque"))]);
    std::thread::scope(|scope| {
        let threads: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    for _ in 0..1000 {
                        assert_eq!(
                            unaccounted(
                                &json!({"a":[{"x":1,"extra":2},{"extra":3}],"whole":{"unknown":4}}),
                                &SHARED
                            ),
                            ["a[].extra", "a[].extra"]
                        );
                    }
                    SHARED.compiled() as *const Compiled as usize
                })
            })
            .collect();
        let addresses: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert!(addresses.iter().all(|address| *address == addresses[0]));
    });
}

#[test]
fn repeated_records_measure_lookup_work_separately_from_pipeline_time() {
    let mut rules = (0..160)
        .map(|n| {
            let path: &'static str =
                Box::leak(format!("metadata{n}.nested.field").into_boxed_str());
            (path, Rule::Kept("synthetic"))
        })
        .collect::<Vec<_>>();
    rules.extend_from_slice(&[
        ("message.content[].text", Rule::Read),
        ("message.role", Rule::Read),
        ("type", Rule::Read),
    ]);
    let fields = Fields::new(Box::leak(rules.into_boxed_slice()));
    let value = json!({"type":"assistant","message":{"role":"assistant","content":[{"text":"one","unknown":1},{"text":"two","unknown":2}]},"metadata99":{"nested":{"field":42}},"unknown":0});
    let expected = reference(&value, fields.rules, &mut 0);
    let count = 5000;
    let start = Instant::now();
    let mut comparisons = 0;
    for _ in 0..count {
        assert_eq!(
            reference(std::hint::black_box(&value), fields.rules, &mut comparisons),
            expected
        );
    }
    let old = start.elapsed();
    let start = Instant::now();
    for _ in 0..count {
        assert_eq!(unaccounted(std::hint::black_box(&value), &fields), expected);
    }
    eprintln!(
        "field accounting component: {count} records; reference={old:?}, compiled={:?}; reference declaration comparisons={comparisons}; one declaration lookup initialization",
        start.elapsed()
    );
    assert!(comparisons > count * fields.rules.len());
}
