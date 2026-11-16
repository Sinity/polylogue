use super::*;

#[test]
fn observed_empty_sinex_differs_from_an_unobserved_cache() {
    let current = IndexFreshness::primary(Some(0), Some(0), Some(0.), Some(true), None);
    assert!(!current.degraded());
    let unobserved = IndexFreshness::primary(None, Some(0), Some(0.), Some(true), None);
    assert!(unobserved.degraded());
    assert!(matches!(
        unobserved,
        IndexFreshness::Primary {
            lag_positions: None,
            state: FreshnessState::Unknown,
            ..
        }
    ));
}

#[test]
fn stale_or_older_sinex_position_has_no_lag() {
    for (sinex, age, reachable, state) in [
        (12, 0., false, FreshnessState::Unavailable),
        (12, 6., true, FreshnessState::Unknown),
        (8, 0., true, FreshnessState::Unknown),
    ] {
        let freshness =
            IndexFreshness::primary(Some(10), Some(sinex), Some(age), Some(reachable), None);
        assert!(
            matches!(freshness,IndexFreshness::Primary{lag_positions:None,state:actual,..} if actual==state)
        );
    }
    let behind = IndexFreshness::primary(Some(10), Some(12), Some(0.2), Some(true), None);
    assert!(matches!(
        behind,
        IndexFreshness::Primary {
            lag_positions: Some(2),
            state: FreshnessState::Behind,
            ..
        }
    ));
}
