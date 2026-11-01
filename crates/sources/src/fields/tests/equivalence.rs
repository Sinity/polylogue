use super::*;
use std::time::Instant;

// Independent declaration scan used to check the compiled lookup.
fn reference(value: &Json, rules: &[(&str, Rule)], checks: &mut usize) -> Vec<String> {
    fn scan(
        value: &Json,
        path: &str,
        rules: &[(&str, Rule)],
        checks: &mut usize,
        out: &mut Vec<String>,
    ) {
        *checks += rules.len();
        let exact = rules.iter().rev().find(|(p, _)| *p == path).map(|(_, r)| r);
        if matches!(
            exact,
            Some(Rule::Indexed(_) | Rule::Kept(_) | Rule::Excluded(_))
        ) {
            return;
        }
        match value {
            Json::Object(map) if !map.is_empty() => {
                for (k, v) in map {
                    scan(
                        v,
                        &if path.is_empty() {
                            k.clone()
                        } else {
                            format!("{path}.{k}")
                        },
                        rules,
                        checks,
                        out,
                    );
                }
            }
            Json::Array(items) if !items.is_empty() => {
                for v in items {
                    scan(v, &format!("{path}[]"), rules, checks, out);
                }
            }
            _ => {
                let deeper = rules.iter().any(|(p, _)| {
                    p.strip_prefix(path)
                        .is_some_and(|s| s.starts_with('.') || s.starts_with('['))
                });
                if exact.is_none() && !deeper && !path.is_empty() {
                    out.push(path.into());
                }
            }
        }
    }
    let mut out = Vec::new();
    scan(value, "", rules, checks, &mut out);
    out
}

#[test]
fn compiled_paths_match_the_original_scanner_for_nested_and_literal_keys() {
    const PATHS: &[&str] = &[
        "a", "a.b", "a[].x", "a[]", "ab", "a.b[].x", "a[odd", "a[", "[].x", "☺.x", "a[][].x",
    ];
    let cases = [
        json!({"a":{"b":1,"x":2,"other":null},"ab":{"y":[]},"unknown":{"x":1}}),
        json!({"a":[{"x":1,"unknown":2},{"x":null,"unknown":3},[{"x":4}]],"ab":false}),
        json!({"a.b":[{"x":1,"u":2}],"a":{"b":[{"x":3,"u":4}]}}),
        json!({"a[]":{"x":1,"u":2},"a[odd":1,"a[":{"x":2},"a":{"b.c":4}}),
        json!([{"x":1,"y":2},{"x":3,"y":4}]),
        json!({"☺":{"x":"yes","extra":[]},"☺.x":false}),
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
                    match n % 4 {
                        0 => Rule::Read,
                        1 => Rule::Kept("synthetic"),
                        2 => Rule::Excluded("synthetic"),
                        _ => Rule::Indexed("synthetic"),
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
        "field counts component: {count} records; reference={old:?}, compiled={:?}; reference declaration comparisons={comparisons}; one declaration lookup initialization",
        start.elapsed()
    );
    assert!(comparisons > count * fields.rules.len());
}
