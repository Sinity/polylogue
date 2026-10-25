use super::super::*;
use crate::excise::record_ranges::{Erasure, Error, plan};
use std::fs::File;
use std::io::Write;

type Range = (usize, usize);

fn compare(owner: &Erasures, file: &File, bytes: &[u8], ranges: &[Range]) {
    let slice = plan(bytes, ranges);
    let streamed = test_plan(owner, file, ranges);
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
        let owner = Erasures::new().unwrap();
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(bytes).unwrap();
        compare(&owner, &file, bytes, &[]);
        for start in 0..=bytes.len() + 1 {
            for end in start..=bytes.len() + 1 {
                compare(&owner, &file, bytes, &[(start, end)]);
                if end <= bytes.len() {
                    compare(
                        &owner,
                        &file,
                        bytes,
                        &[(start, end), (0, end), (start, end)],
                    );
                }
            }
        }
    }
}

#[test]
fn disjoint_nested_removals_across_roots_match_exact_separator_choices() {
    let bytes = br#"[1, 2, 3] {"nested":[4,5,6]} [7,8,9]"#;
    let owner = Erasures::new().unwrap();
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
        compare(&owner, &file, bytes, &ranges);
    }
}

#[test]
fn chunk_boundary_whitespace_and_many_records_match_without_a_whole_file_buffer() {
    let owner = Erasures::new().unwrap();
    let mut file = tempfile::tempfile().unwrap();
    let padding = vec![b' '; 70_000];
    file.write_all(&padding).unwrap();
    let first = padding.len();
    for n in 0..20_000 {
        writeln!(file, "{{\"n\":{n}}}").unwrap();
    }
    let end = file.metadata().unwrap().len() as usize;
    file.write_all(&padding).unwrap();
    let result = test_plan(&owner, &file, &[(0, end + padding.len())]).unwrap();
    assert_eq!(result.documents, 20_000);
    assert_eq!(result.erased_documents, 20_000);
    assert_eq!(result.ranges, [(first, end - 1)]);
}

#[test]
fn read_errors_stay_distinct_from_invalid_record_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let directory = File::open(dir.path()).unwrap();
    let owner = Erasures::new().unwrap();
    assert!(matches!(
        test_plan(&owner, &directory, &[]),
        Err(Error::Io(_))
    ));
}

fn test_plan(
    owner: &Erasures,
    file: &File,
    ranges: &[Range],
) -> std::result::Result<Erasure, Error> {
    owner.conn.execute("DELETE FROM requested", [])?;
    for &(start, end) in ranges {
        owner.request(b"test", start, end).map_err(Error::Store)?;
    }
    let (documents, erased_documents) = owner.plan_blob(b"test", file)?;
    let mut ranges = Vec::new();
    owner
        .visit_ranges(b"test", |start, end| {
            ranges.push((start, end));
            Ok(())
        })
        .map_err(Error::Store)?;
    Ok(Erasure {
        ranges,
        documents,
        erased_documents,
    })
}

#[test]
fn many_ranges_in_one_root_or_many_roots_stay_in_the_existing_disk_owner() {
    for array in [true, false] {
        let owner = Erasures::new().unwrap();
        let mut file = tempfile::tempfile().unwrap();
        if array {
            file.write_all(b"[").unwrap();
        }
        for n in 0..25_000 {
            let start = file.metadata().unwrap().len() as usize;
            owner.request(b"many", start, start + 1).unwrap();
            // Duplicate citations do not create a second removal.
            owner.request(b"many", start, start + 1).unwrap();
            if array {
                file.write_all(if n == 24_999 { b"1,0]" } else { b"1,0," })
                    .unwrap();
            } else {
                file.write_all(b"1\n0\n").unwrap();
            }
        }
        assert_eq!(
            owner.plan_blob(b"many", &file).unwrap(),
            if array { (1, 0) } else { (50_000, 25_000) }
        );
        let mut count = 0;
        owner
            .visit_ranges(b"many", |start, end| {
                assert_eq!(start, count * 4 + usize::from(array));
                assert_eq!(end - start, if array { 2 } else { 1 });
                count += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(count, 25_000);
        let bytes: i64 = owner
            .conn
            .query_row(
                "SELECT page_count*page_size FROM pragma_page_count(),pragma_page_size()",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(bytes > 1 << 20);
        assert_eq!(owner.conn.path(), Some(""));
    }
}

#[test]
fn malformed_eof_cannot_expose_partial_ranges_on_a_retry() {
    let owner = Erasures::new().unwrap();
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(b"[1,0,1,0]\n{").unwrap();
    for (start, end) in [(1, 2), (5, 6)] {
        owner.request(b"late", start, end).unwrap();
    }
    for _ in 0..2 {
        assert!(matches!(
            owner.plan_blob(b"late", &file),
            Err(Error::Records(_))
        ));
        assert_eq!(owner.counts(b"late").unwrap(), None);
        assert!(owner.has_unchecked().unwrap());
        let partial: i64 = owner
            .conn
            .query_row(
                "SELECT count(*) FROM ranges WHERE hash=?1",
                [b"late"],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(partial, 1);
        let mut calls = 0;
        assert!(
            owner
                .visit_ranges(b"late", |_, _| {
                    calls += 1;
                    Ok(())
                })
                .is_err()
        );
        owner
            .visit(|_| {
                calls += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(calls, 0);
    }
}

#[cfg(feature = "faults")]
#[test]
fn scratch_failures_stay_structured_and_retry_clears_uncommitted_ranges() {
    use crate::fault::{self, Action};
    for (point, skip) in [
        ("excision_ranges.read", 1),
        ("excision_ranges.write", 0),
        ("excision_ranges.complete", 0),
    ] {
        fault::clear();
        let owner = Erasures::new().unwrap();
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"[1,0,1,0]\n").unwrap();
        for (start, end) in [(1, 2), (5, 6)] {
            owner.request(b"retry", start, end).unwrap();
        }
        fault::arm(point, "", Action::Fail, skip, 1);
        assert!(
            matches!(owner.plan_blob(b"retry", &file), Err(Error::Store(_))),
            "{point}"
        );
        assert_eq!(owner.counts(b"retry").unwrap(), None);
        let mut calls = 0;
        assert!(
            owner
                .visit_ranges(b"retry", |_, _| {
                    calls += 1;
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(calls, 0);
        assert_eq!(owner.plan_blob(b"retry", &file).unwrap(), (1, 0));
        let mut ranges = vec![];
        owner
            .visit_ranges(b"retry", |start, end| {
                ranges.push((start, end));
                Ok(())
            })
            .unwrap();
        assert_eq!(ranges, [(1, 3), (5, 7)]);
    }
    fault::clear();
}
