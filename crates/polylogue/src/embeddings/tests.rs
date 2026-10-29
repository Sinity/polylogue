use super::*;

#[test]
fn long_texts_are_chunked_and_normalized() {
    let short = chunks("cafe\u{301}");
    assert_eq!(short, ["café"]);
    let long = "x".repeat(CHUNK_CHARS * 2 + 5);
    assert_eq!(chunks(&long).len(), 3);
    let v = unit(vec![3.0, 4.0]);
    assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
}

mod cancellation;

#[test]
fn provider_consumes_typed_settings_and_captured_credentials() {
    use polylogue_model::settings::{self, Current, Owner, Snapshot};
    let snapshot = |provider: &str| {
        Snapshot::new(
            settings::CATALOG
                .iter()
                .filter(|spec| !spec.key.ends_with('*'))
                .map(|spec| Current {
                    key: spec.key.into(),
                    kind: spec.kind,
                    default: None,
                    scope: spec.scope,
                    owner: "polylogue".into(),
                    value: settings::current(
                        spec.key,
                        if spec.key == "embed.provider" {
                            Some(provider)
                        } else {
                            None
                        },
                        Owner::Standalone,
                    )
                    .unwrap(),
                    authored: None,
                })
                .collect::<Vec<_>>(),
        )
    };
    let credentials = Credentials::new(Some("synthetic-secret".into()));
    assert!(!format!("{credentials:?}").contains("synthetic-secret"));
    assert!(
        Provider::from_settings(&snapshot("off"), &credentials)
            .unwrap()
            .is_none()
    );
    let local = Provider::from_settings(&snapshot("local"), &credentials)
        .unwrap()
        .unwrap();
    assert_eq!(local.endpoint, "http://127.0.0.1:11434/v1");
    assert_eq!(local.price_per_mtok, Some(0.0));
    let voyage = Provider::from_settings(&snapshot("voyage"), &credentials)
        .unwrap()
        .unwrap();
    assert_eq!(voyage.model, "voyage-4");
    assert!(!format!("{voyage:?}").contains("synthetic-secret"));
    assert!(Provider::from_settings(&Snapshot::new(Vec::new()), &credentials).is_err());
}
