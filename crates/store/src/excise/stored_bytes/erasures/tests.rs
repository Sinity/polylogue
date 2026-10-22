use super::*;
use crate::excise::record_ranges::Erasure;
mod records;

impl Erasures {
    pub fn requested(&self, hash: &[u8]) -> Result<Vec<(usize, usize)>> {
        self.read_ranges("requested", hash)
    }

    pub fn checked(&self, hash: &[u8], plan: &Erasure) -> Result<()> {
        let changed = !plan.ranges.is_empty();
        self.conn
            .prepare_cached("INSERT INTO checked VALUES(?1,?2,?3,?4)")?
            .execute(params![
                hash,
                plan.documents as i64,
                plan.erased_documents as i64,
                changed
            ])?;
        for &(start, end) in &plan.ranges {
            self.conn
                .prepare_cached("INSERT OR IGNORE INTO ranges VALUES(?1,?2,?3)")?
                .execute(params![hash, start as i64, end as i64])?;
        }
        Ok(())
    }

    fn read_ranges(&self, table: &str, hash: &[u8]) -> Result<Vec<(usize, usize)>> {
        let mut stmt = self.conn.prepare_cached(&format!(
            "SELECT start,end FROM {table} WHERE hash=?1 ORDER BY start,end"
        ))?;
        Ok(stmt
            .query_map([hash], |r| {
                Ok((r.get::<_, i64>(0)? as usize, r.get::<_, i64>(1)? as usize))
            })?
            .collect::<rusqlite::Result<_>>()?)
    }
}

#[test]
fn requests_and_checked_plans_are_exact_ordered_and_operation_local() {
    let plan = Erasures::new().unwrap();
    for n in (0..10_000_u32).rev() {
        let hash = blake3::hash(&n.to_le_bytes());
        for (start, end) in [(12, 24), (1, 9), (12, 24)] {
            plan.request(hash.as_bytes(), start, end).unwrap();
        }
        assert_eq!(plan.requested(hash.as_bytes()).unwrap(), [(1, 9), (12, 24)]);
        plan.checked(
            hash.as_bytes(),
            &Erasure {
                ranges: if n % 2 == 0 {
                    vec![(1, 10), (12, 24)]
                } else {
                    vec![]
                },
                documents: 3,
                erased_documents: usize::from(n % 2 == 0),
            },
        )
        .unwrap();
        assert_eq!(
            plan.counts(hash.as_bytes()).unwrap(),
            Some((3, usize::from(n % 2 == 0)))
        );
    }
    assert!(!plan.has_unchecked().unwrap());
    let mut count = 0;
    let mut previous: Option<Vec<u8>> = None;
    plan.visit(|hash| {
        let ranges = plan.read_ranges("ranges", hash)?;
        assert!(previous.as_deref().is_none_or(|p| p < hash));
        previous = Some(hash.to_vec());
        assert_eq!(ranges, [(1, 10), (12, 24)]);
        count += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 5000);
    assert!(
        plan.visit(|_| Err(crate::StoreError::Invalid("stop".into())))
            .is_err()
    );
    let bytes: i64 = plan
        .conn
        .query_row(
            "SELECT page_count*page_size FROM pragma_page_count(),pragma_page_size()",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(bytes > 1 << 20);
    assert_eq!(plan.conn.path(), Some(""));
    drop(plan);
}

#[test]
fn a_checked_unrequested_piece_does_not_hide_an_unmatched_citation() {
    let plan = Erasures::new().unwrap();
    plan.request(b"missing", 1, 4).unwrap();
    plan.checked(
        b"unchanged",
        &Erasure {
            ranges: vec![],
            documents: 2,
            erased_documents: 0,
        },
    )
    .unwrap();
    assert_eq!(plan.counts(b"unchanged").unwrap(), Some((2, 0)));
    assert_eq!(plan.counts(b"missing").unwrap(), None);
    assert!(plan.has_unchecked().unwrap());
    assert!(
        plan.checked(
            b"unchanged",
            &Erasure {
                ranges: vec![],
                documents: 3,
                erased_documents: 0
            }
        )
        .is_err()
    );
    let mut visits = 0;
    plan.visit(|_| {
        visits += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(visits, 0);
}
