use super::*;
use proptest::prelude::*;

/// The true common prefix of two byte strings.
fn common(a: &[u8], b: &[u8]) -> u64 {
    a.iter().zip(b).take_while(|(x, y)| x == y).count() as u64
}

/// Stores `versions` as the chains of one input in order, each recorded as
/// the writer records it, and returns the recorded chains.
fn record(versions: &[Vec<u8>]) -> Chains {
    let mut chains = Chains::default();
    for (n, bytes) in versions.iter().enumerate() {
        let number = n as i64 + 1;
        let latest = chains.0.keys().next_back().copied();
        let agreement = chains
            .agreement(latest, |other, from| {
                first_difference(
                    bytes.as_slice(),
                    versions[other as usize - 1].as_slice(),
                    from,
                )
            })
            .unwrap();
        chains.0.insert(number, agreement);
    }
    chains
}

/// Byte strings over a small alphabet, so that versions share prefixes.
fn versions() -> impl Strategy<Value = Vec<Vec<u8>>> {
    prop::collection::vec(prop::collection::vec(0u8..3, 0..24), 1..7)
}

/// Whether some chain's bytes are a prefix of an earlier chain's.
fn stored_as_a_prefix(versions: &[Vec<u8>]) -> bool {
    versions.iter().enumerate().any(|(n, later)| {
        versions[..n]
            .iter()
            .any(|earlier| earlier.starts_with(later))
    })
}

proptest! {
    /// How far two chains are known to agree, by the smallest agreement on
    /// the path between them, never exceeds their true common prefix, and
    /// equals it whenever no chain was first stored as a prefix of an
    /// earlier one. What a chain records against the chain it begins like is
    /// exact.
    #[test]
    fn recorded_agreement_never_exceeds_the_true_prefix(versions in versions()) {
        let chains = record(&versions);
        let exact = !stored_as_a_prefix(&versions);
        for a in 1..=versions.len() as i64 {
            for b in 1..=versions.len() as i64 {
                if a == b {
                    continue;
                }
                let truth = common(&versions[a as usize - 1], &versions[b as usize - 1]);
                let known = chains.known(a, b).unwrap_or(0);
                prop_assert!(known <= truth, "{a} {b}: {known} > {truth}");
                if exact {
                    prop_assert_eq!(known, truth, "{} {}", a, b);
                }
            }
            let recorded = chains.0[&a];
            if let Some(like) = recorded.begins_like {
                let truth = common(&versions[a as usize - 1], &versions[like as usize - 1]);
                prop_assert_eq!(recorded.agrees_to, truth);
            }
        }
    }
}

#[test]
fn a_new_chain_begins_like_the_earlier_chain_it_shares_most_with() {
    let versions = [
        b"header|alpha|one".to_vec(),
        b"header|beta".to_vec(),
        b"header|alpha|two".to_vec(),
    ];
    let chains = record(&versions);
    assert_eq!(chains.0[&1], Agreement::NONE);
    assert_eq!(
        chains.0[&2],
        Agreement {
            begins_like: Some(1),
            agrees_to: 7
        }
    );
    // Equal to the latest through `header|` and so compared with the first
    // from there: it agrees with the first further.
    assert_eq!(
        chains.0[&3],
        Agreement {
            begins_like: Some(1),
            agrees_to: 13
        }
    );
    assert_eq!(chains.known(2, 3), Some(7));
}
