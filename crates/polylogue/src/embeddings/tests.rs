use super::*;

#[test]
fn long_texts_are_chunked_and_normalized() {
    let short = chunks("cafe\u{301}");
    assert_eq!(short, ["café"]);
    let long = "x".repeat(CHUNK_CHARS * 2 + 5);
    assert_eq!(chunks(&long).len(), 3);
    let v = unit(vec![3.0, 4.0]).unwrap();
    assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
}

mod cancellation;
mod proxy;

#[test]
fn provider_consumes_typed_settings_and_captured_credentials() {
    use polylogue_model::settings::{self, Current, Owner, Snapshot};
    let snapshot_with = |provider: &str, endpoint: Option<&str>| {
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
                        match spec.key {
                            "embed.provider" => Some(provider),
                            "embed.endpoint" => endpoint,
                            _ => None,
                        },
                        Owner::Standalone,
                    )
                    .unwrap(),
                    authored: None,
                    conflicts: Vec::new(),
                })
                .collect::<Vec<_>>(),
        )
    };
    let snapshot = |provider: &str| snapshot_with(provider, None);
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
    // A Voyage provider never points its key at another origin.
    let elsewhere = Provider::from_settings(
        &snapshot_with("voyage", Some("http://127.0.0.1:11434/v1")),
        &credentials,
    )
    .unwrap_err();
    assert!(elsewhere.contains("not Voyage's origin"), "{elsewhere}");
    assert!(
        Provider::from_settings(
            &snapshot_with("voyage", Some("https://api.voyageai.com/v1")),
            &credentials,
        )
        .unwrap()
        .is_some()
    );
}

#[test]
fn embedding_selection_refuses_aggregate_and_incomplete_session_results() {
    use polylogue_model::authorship::{Author, Authorship, Channel};
    let daemon = crate::test_daemon::start("synthetic selected text");
    let operation = daemon
        .daemon
        .operation_context(&Authorship {
            author: Author::Operator,
            via: Channel::Cli,
        })
        .unwrap();
    let reader = daemon.daemon.reads.reader().unwrap();
    let selected = query_sessions(&reader, "sessions", &operation).unwrap();
    assert_eq!(selected, ["claude-code:s1"]);
    for (query, expected) in [
        ("sessions | group by origin | count", "bad_request"),
        (
            "sessions where similar:\"synthetic selected text\"",
            "not_yet_available",
        ),
    ] {
        let response = query_sessions(&reader, query, &operation).unwrap_err();
        assert!(response.code() == expected, "{query}: {response:?}");
    }
}

mod batch;

#[test]
fn only_voyages_own_origin_counts_as_voyage() {
    assert!(is_voyage_origin("https://api.voyageai.com/v1"));
    assert!(is_voyage_origin("https://api.voyageai.com:443/v1/"));
    assert!(!is_voyage_origin("http://api.voyageai.com/v1"));
    assert!(!is_voyage_origin("https://api.voyageai.com.example.net/v1"));
    assert!(!is_voyage_origin("https://api.voyageai.com:8443/v1"));
    assert!(!is_voyage_origin("http://127.0.0.1:11434/v1"));
    assert!(!is_voyage_origin("not a url"));
}

#[test]
fn a_voyage_provider_pointed_elsewhere_sends_no_key() {
    use std::io::Read;
    use std::net::TcpListener;
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let provider = Provider {
        kind: ProviderKind::Voyage,
        endpoint: format!("http://{}/v1", listener.local_addr().unwrap()),
        model: "voyage-4".into(),
        price_per_mtok: None,
        credentials: Credentials::new(Some("synthetic-voyage-key".into())),
    };
    let error = provider.embed(&["text".into()], false).unwrap_err();
    assert!(matches!(error, EmbedError::Config(_)), "{error:?}");
    // The refusal happens before any connection: nothing reached the
    // listener, so no request could have carried the key.
    match listener.accept() {
        Ok((mut stream, _)) => {
            let mut received = String::new();
            let _ = stream.read_to_string(&mut received);
            panic!("the provider connected to a non-Voyage origin: {received:?}");
        }
        Err(error) => assert_eq!(error.kind(), std::io::ErrorKind::WouldBlock),
    }
}

#[test]
fn the_local_provider_only_uses_loopback_addresses() {
    assert!(is_loopback("http://127.0.0.1:11434/v1"));
    assert!(is_loopback("http://127.8.9.10:80/v1"));
    assert!(is_loopback("http://[::1]:11434/v1"));
    assert!(is_loopback("http://[::ffff:127.0.0.1]:11434/v1"));
    assert!(!is_loopback("http://[::ffff:192.0.2.1]:11434/v1"));
    assert!(is_loopback("http://localhost:11434/v1"));
    assert!(!is_loopback("http://192.168.1.5:11434/v1"));
    assert!(!is_loopback("https://example.com/v1"));
    assert!(!is_loopback("http://localhost.example.com/v1"));
    assert!(!is_loopback("not a url"));
    let provider = Provider {
        kind: ProviderKind::Local,
        endpoint: "http://192.0.2.1:11434/v1".into(),
        model: "synthetic".into(),
        price_per_mtok: Some(0.0),
        credentials: Credentials::new(None),
    };
    let error = provider.embed(&["private text".into()], false).unwrap_err();
    assert!(matches!(error, EmbedError::Config(_)), "{error:?}");
}
