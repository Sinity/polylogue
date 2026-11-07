use super::*;

#[test]
fn observed_empty_owner_differs_from_an_unobserved_cache() {
    let current = IndexCurrency::primary(Some(0), Some(0), Some(0.), Some(true), None);
    assert!(!current.degraded());
    let unobserved = IndexCurrency::primary(None, Some(0), Some(0.), Some(true), None);
    assert!(unobserved.degraded());
    assert!(matches!(
        unobserved,
        IndexCurrency::Primary {
            lag_positions: None,
            state: CurrencyState::Unknown,
            ..
        }
    ));
}

#[test]
fn outage_stale_or_older_owner_never_claims_zero_or_negative_lag() {
    for (owner, age, reachable, state) in [
        (12, 0., false, CurrencyState::Unavailable),
        (12, 6., true, CurrencyState::Unknown),
        (8, 0., true, CurrencyState::Unknown),
    ] {
        let currency =
            IndexCurrency::primary(Some(10), Some(owner), Some(age), Some(reachable), None);
        assert!(
            matches!(currency,IndexCurrency::Primary{lag_positions:None,state:actual,..} if actual==state)
        );
    }
    let behind = IndexCurrency::primary(Some(10), Some(12), Some(0.2), Some(true), None);
    assert!(matches!(
        behind,
        IndexCurrency::Primary {
            lag_positions: Some(2),
            state: CurrencyState::Behind,
            ..
        }
    ));
}
