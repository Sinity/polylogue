use super::*;

#[test]
fn completeness_reasons_keep_their_wire_names_and_order() {
    let wire = concat!(
        r#"["missing_declaration","missing_checkpoint","parse_version_changed","parser_behind","#,
        r#""capture_pending","parsing_coverage_pending","refused","parse_failed","derivation_behind","search_pending","index_building","#,
        r#""stream_not_applied","stream_behind"]"#,
    );
    assert_eq!(
        serde_json::to_string(CompletenessReason::ALL).unwrap(),
        wire
    );
    assert_eq!(
        serde_json::from_str::<Vec<CompletenessReason>>(wire).unwrap(),
        CompletenessReason::ALL
    );
}

#[test]
fn primary_observation_age_decodes_from_tagged_json() {
    for age in ["0", "2", "0.436835857", "4.2e-1", "null"] {
        let json =
            format!(r#"{{"mode":"primary","sinex_position_seconds_ago":{age},"state":"current"}}"#);
        let freshness: IndexFreshness = serde_json::from_str(&json).unwrap();
        let IndexFreshness::Primary {
            sinex_position_seconds_ago,
            ..
        } = freshness
        else {
            panic!("primary freshness expected")
        };
        assert_eq!(
            sinex_position_seconds_ago,
            serde_json::from_str::<Option<f64>>(age).unwrap()
        );
        assert_eq!(
            serde_json::from_str::<IndexFreshness>(&serde_json::to_string(&freshness).unwrap())
                .unwrap(),
            freshness
        );
    }
    let absent: IndexFreshness =
        serde_json::from_str(r#"{"mode":"primary","state":"unknown"}"#).unwrap();
    assert!(matches!(
        absent,
        IndexFreshness::Primary {
            sinex_position_seconds_ago: None,
            ..
        }
    ));
    for age in [r#""0.2""#, "{}", "[]", "true", "1e400", "NaN", "Infinity"] {
        let json =
            format!(r#"{{"mode":"primary","sinex_position_seconds_ago":{age},"state":"current"}}"#);
        assert!(
            serde_json::from_str::<IndexFreshness>(&json).is_err(),
            "{age}"
        );
    }
}

#[test]
fn observed_empty_sinex_differs_from_an_unobserved_cache() {
    let current = IndexFreshness::primary(Some(0), Some(0), Some(0.), Some(true), None);
    assert!(!current.degraded());
    let unobserved = IndexFreshness::primary(None, Some(0), Some(0.), Some(true), None);
    assert!(unobserved.degraded());
    assert_eq!(
        unobserved,
        IndexFreshness::Primary {
            scope: None,
            applied_position: None,
            sinex_position: Some(0),
            sinex_position_seconds_ago: Some(0.),
            reachable: Some(true),
            lag_positions: None,
            state: FreshnessState::Unknown,
            error: None,
        }
    );
}

#[test]
fn unreachable_stale_or_older_sinex_position_has_no_lag() {
    for (sinex, age, reachable, state) in [
        (12, 0., false, FreshnessState::Unavailable),
        (12, 6., true, FreshnessState::Unknown),
        (8, 0., true, FreshnessState::Unknown),
    ] {
        let freshness =
            IndexFreshness::primary(Some(10), Some(sinex), Some(age), Some(reachable), None);
        assert_eq!(
            freshness,
            IndexFreshness::Primary {
                scope: None,
                applied_position: Some(10),
                sinex_position: Some(sinex),
                sinex_position_seconds_ago: Some(age),
                reachable: Some(reachable),
                lag_positions: None,
                state,
                error: None,
            }
        );
    }
    let failed = IndexFreshness::primary(
        Some(10),
        Some(12),
        Some(0.),
        Some(false),
        Some("unreachable".into()),
    );
    assert!(matches!(
        failed,
        IndexFreshness::Primary {
            error: Some(ref error),
            state: FreshnessState::Unavailable,
            ..
        } if error == "unreachable"
    ));
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
