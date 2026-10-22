use super::*;
use std::io::Write;

fn compare(file: &File, bytes: &[u8], ranges: &[Range]) {
    let slice = super::super::plan(bytes, ranges);
    let streamed = plan_file(file, ranges);
    match (slice, streamed) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "{ranges:?}"),
        (Err(_), Err(Error::Records(_))) => (),
        (a, b) => panic!("ranges={ranges:?}, bytes={bytes:?}: slice={a:?}, file={b:?}"),
    }
}

#[test]
fn every_pair_of_endpoints_matches_the_slice_planner() {
    for bytes in [
        br#"[1,2] {"b":3}"#.as_slice(),
        b" \n{\"a\":1}\t {\"b\":2}\r\n",
        b"123\n{\"b\":3}",
        br#"{"a":[{"x":1},{"x":2}],"b":3} [true,null]"#,
        br#"["quote\" and }", "\\", "\u263a", [1,2]]"#,
        b"[1, 2, 3]\n[4,5]\n[6]",
        b"true false null 123 \"text\"",
        b"[1,2]\n{malformed suffix",
        b" ",
        b"",
        b"{",
        b"[1] trailing",
    ] {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(bytes).unwrap();
        compare(&file, bytes, &[]);
        for start in 0..=bytes.len() + 1 {
            for end in start..=bytes.len() + 1 {
                compare(&file, bytes, &[(start, end)]);
                if end <= bytes.len() {
                    compare(&file, bytes, &[(start, end), (0, end), (start, end)]);
                }
            }
        }
    }
}

#[test]
fn disjoint_nested_removals_across_roots_match_exact_separator_choices() {
    let bytes = br#"[1, 2, 3] {"nested":[4,5,6]} [7,8,9]"#;
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(bytes).unwrap();
    for mask in 0..512 {
        let ranges: Vec<_> = (b'1'..=b'9')
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, n)| {
                let at = bytes.iter().position(|&b| b == n).unwrap();
                (at, at + 1)
            })
            .collect();
        compare(&file, bytes, &ranges);
    }
}

#[test]
fn chunk_boundary_whitespace_and_many_records_match_without_a_whole_file_buffer() {
    let mut file = tempfile::tempfile().unwrap();
    let padding = vec![b' '; 70_000];
    file.write_all(&padding).unwrap();
    let first = padding.len();
    for n in 0..20_000 {
        writeln!(file, "{{\"n\":{n}}}").unwrap();
    }
    let end = file.metadata().unwrap().len() as usize;
    file.write_all(&padding).unwrap();
    let result = plan_file(&file, &[(0, end + padding.len())]).unwrap();
    assert_eq!(result.documents, 20_000);
    assert_eq!(result.erased_documents, 20_000);
    assert_eq!(result.ranges, [(first, end - 1)]);
}

#[test]
fn read_errors_stay_distinct_from_invalid_record_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let directory = File::open(dir.path()).unwrap();
    assert!(matches!(plan_file(&directory, &[]), Err(Error::Io(_))));
}
