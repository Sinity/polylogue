use super::*;
use std::net::TcpListener;
use std::time::Duration;

fn provider(replies: Vec<(u16, String)>) -> (Provider, std::thread::JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        replies
            .into_iter()
            .map(|(status, body)| {
                let until = Instant::now() + Duration::from_secs(5);
                let peer = loop {
                    match listener.accept() {
                        Ok((peer, _)) => break peer,
                        Err(error)
                            if error.kind() == std::io::ErrorKind::WouldBlock
                                && Instant::now() < until =>
                        {
                            std::thread::sleep(Duration::from_millis(1))
                        }
                        Err(error) => {
                            panic!("synthetic provider did not receive its next request: {error}")
                        }
                    }
                };
                crate::test_daemon::embeddings::response(peer, status, &body)
            })
            .collect()
    });
    (
        Provider {
            kind: ProviderKind::Local,
            endpoint,
            model: "batch-test".into(),
            price_per_mtok: Some(polylogue_model::Rate::ZERO),
            credentials: Credentials::new(None),
        },
        server,
    )
}

fn answer(data: Value) -> String {
    json!({"data":data}).to_string()
}

#[test]
fn malformed_provider_batches_are_refused_as_a_whole() {
    let cases = [
        ("data array", r#"{"data":{}}"#.into()),
        ("count", answer(json!([{"index":0,"embedding":[1,0]}]))),
        (
            "duplicate",
            answer(json!([{"index":0,"embedding":[1,0]},{"index":0,"embedding":[0,1]}])),
        ),
        (
            "out of range",
            answer(json!([{"index":0,"embedding":[1,0]},{"index":2,"embedding":[0,1]}])),
        ),
        (
            "input index",
            answer(json!([{"index":0,"embedding":[1,0]},{"embedding":[0,1]}])),
        ),
        (
            "input index",
            answer(json!([{"index":null,"embedding":[1,0]},{"index":1,"embedding":[0,1]}])),
        ),
        (
            "input index",
            answer(json!([{"index":-1,"embedding":[1,0]},{"index":1,"embedding":[0,1]}])),
        ),
        (
            "input index",
            answer(json!([{"index":"0","embedding":[1,0]},{"index":1,"embedding":[0,1]}])),
        ),
        (
            "input index",
            answer(json!([{"index":0.5,"embedding":[1,0]},{"index":1,"embedding":[0,1]}])),
        ),
        (
            "out of range",
            answer(json!([{"index":u64::MAX,"embedding":[1,0]},{"index":1,"embedding":[0,1]}])),
        ),
        (
            "embedding array",
            answer(json!([{"index":0},{"index":1,"embedding":[0,1]}])),
        ),
        (
            "finite number",
            answer(json!([{"index":0,"embedding":["x",1]},{"index":1,"embedding":[0,1]}])),
        ),
        (
            "finite number",
            answer(json!([{"index":0,"embedding":[null,1]},{"index":1,"embedding":[0,1]}])),
        ),
        (
            "f32 range",
            answer(json!([{"index":0,"embedding":[1e300,1]},{"index":1,"embedding":[0,1]}])),
        ),
        (
            "norm",
            answer(json!([{"index":0,"embedding":[0,0]},{"index":1,"embedding":[0,1]}])),
        ),
        (
            "norm",
            answer(json!([{"index":0,"embedding":[]},{"index":1,"embedding":[]}])),
        ),
        (
            "dimensions",
            answer(json!([{"index":0,"embedding":[1,0]},{"index":1,"embedding":[0,1,0]}])),
        ),
    ];
    for (expected, body) in cases {
        let (provider, server) = provider(vec![(200, body)]);
        let result = provider.embed(&["first".into(), "second".into()], false);
        let requests = server.join().unwrap();
        assert_eq!(requests[0]["input"], json!(["first", "second"]));
        match result {
            Err(EmbedError::Transport(error)) => assert!(error.contains(expected), "{error}"),
            other => panic!("expected {expected} refusal, got {other:?}"),
        }
    }
    for body in [
        r#"{"data":[{"index":0,"embedding":[NaN]}]}"#,
        r#"{"data":[{"index":0,"embedding":[1e400]}]}"#,
    ] {
        let (provider, server) = provider(vec![(200, body.into())]);
        let result = provider.embed(&["first".into()], false);
        server.join().unwrap();
        assert!(matches!(result, Err(EmbedError::Transport(_))));
    }
}

#[test]
fn valid_reordered_and_extreme_finite_vectors_keep_their_input_identity() {
    let body = answer(json!([{"index":1,"embedding":[0,2]},{"index":0,"embedding":[3,4]}]));
    let tiny = f32::from_bits(1);
    let extreme = answer(json!([
        {"index":0,"embedding":[f32::MAX,f32::MAX]},
        {"index":1,"embedding":[tiny,0]},
    ]));
    let (provider, server) = provider(vec![(200, body), (200, extreme)]);
    let vectors = provider
        .embed(&["first".into(), "second".into()], false)
        .unwrap();
    assert!((vectors[0][0] - 0.6).abs() < 1e-6);
    assert!((vectors[0][1] - 0.8).abs() < 1e-6);
    assert_eq!(vectors[1], [0.0, 1.0]);
    let vectors = provider
        .embed(&["large".into(), "tiny".into()], false)
        .unwrap();
    assert!(
        vectors[0]
            .iter()
            .all(|value| (*value - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6)
    );
    assert_eq!(vectors[1], [1.0, 0.0]);
    assert_eq!(server.join().unwrap().len(), 2);
}

#[test]
fn a_malformed_success_response_never_stores_its_valid_prefix() {
    let daemon = crate::test_daemon::start(&"x".repeat(CHUNK_CHARS + 4));
    let body = answer(json!([{"index":0,"embedding":[1,0]},{"index":0,"embedding":[0,1]}]));
    let (provider, server) = provider(vec![(200, body)]);
    let reader = daemon.daemon.reads.reader().unwrap();
    let result = build(&reader, &provider, None, None);
    let requests = server.join().unwrap();
    assert_eq!(requests[0]["input"].as_array().unwrap().len(), 2);
    assert!(matches!(result, Err(EmbedError::Transport(_))));
    assert!(reader.vectors().unwrap().status().unwrap().keys.is_empty());
}

#[test]
fn a_partial_purchase_resumes_only_the_missing_final_chunk() {
    let text = format!("{}tail", "x".repeat(CHUNK_CHARS * 32));
    let daemon = crate::test_daemon::start(&text);
    let first = answer(Value::Array(
        (0..32)
            .map(|index| json!({"index":index,"embedding":[1,0]}))
            .collect(),
    ));
    let last = answer(json!([{"index":0,"embedding":[0,1]}]));
    let (provider, server) = provider(vec![(200, first), (503, "{}".into()), (200, last)]);
    let reader = daemon.daemon.reads.reader().unwrap();
    assert!(matches!(
        build(&reader, &provider, None, None),
        Err(EmbedError::Transport(_))
    ));
    drop(reader);
    let reader = daemon.daemon.reads.reader().unwrap();
    let vectors = reader.vectors().unwrap();
    let key = provider.key(2);
    assert_eq!(vectors.load(&key).unwrap().vectors(), 32);
    let revision = vectors.revision(&key).unwrap();
    drop(vectors);
    let remaining = preview(&reader, &provider, None, None).unwrap();
    assert_eq!(
        (
            remaining.texts,
            remaining.chunks,
            remaining.characters,
            remaining.tokens_estimate
        ),
        (1, 1, 4, 1)
    );
    assert_eq!(remaining.already_embedded, 0);
    let resumed = build(&reader, &provider, None, None).unwrap();
    assert_eq!(resumed.vectors, 1);
    drop(reader);
    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0]["input"].as_array().unwrap().len(), 32);
    assert_eq!(requests[1]["input"], json!(["tail"]));
    assert_eq!(requests[2]["input"], json!(["tail"]));
    let reader = daemon.daemon.reads.reader().unwrap();
    let vectors = reader.vectors().unwrap();
    assert_eq!(vectors.load(&key).unwrap().vectors(), 33);
    assert_eq!(vectors.revision(&key).unwrap(), revision + 1);
    drop(vectors);
    assert_eq!(
        preview(&reader, &provider, None, None)
            .unwrap()
            .already_embedded,
        1
    );
}

#[test]
fn singleton_fallback_dimensions_are_checked_before_any_put() {
    let daemon = crate::test_daemon::start(&"x".repeat(CHUNK_CHARS + 4));
    let (provider, server) = provider(vec![
        (400, "{}".into()),
        (200, answer(json!([{"index":0,"embedding":[1,0]}]))),
        (200, answer(json!([{"index":0,"embedding":[1,0,0]}]))),
    ]);
    let reader = daemon.daemon.reads.reader().unwrap();
    let result = build(&reader, &provider, None, None);
    let requests = server.join().unwrap();
    assert_eq!(
        requests
            .iter()
            .map(|request| request["input"].as_array().unwrap().len())
            .collect::<Vec<_>>(),
        [2, 1, 1]
    );
    assert!(
        matches!(result, Err(EmbedError::Transport(error)) if error.contains("dimensions across requests"))
    );
    assert!(reader.vectors().unwrap().status().unwrap().keys.is_empty());
}

#[test]
fn resumed_provider_dimensions_cannot_relabel_purchased_chunks() {
    let text = "x".repeat(CHUNK_CHARS + 4);
    let daemon = crate::test_daemon::start(&text);
    let (provider, server) = provider(vec![(
        200,
        answer(json!([{"index":0,"embedding":[1,0,0]}])),
    )]);
    let reader = daemon.daemon.reads.reader().unwrap();
    let input = reader
        .prose_texts(0, PAGE, None)
        .unwrap()
        .into_iter()
        .find(|input| input.text == text)
        .unwrap();
    let key = provider.key(2);
    let mut vectors = reader.vectors_writer().unwrap();
    vectors
        .put(
            &key,
            &[Vector {
                text_hash: input.hash.clone(),
                chunk: 0,
                values: vec![1.0, 0.0],
            }],
        )
        .unwrap();
    let revision = vectors.revision(&key).unwrap();
    drop(vectors);
    let result = build(&reader, &provider, None, None);
    let requests = server.join().unwrap();
    assert_eq!(requests[0]["input"], json!(["xxxx"]));
    assert!(
        matches!(result, Err(EmbedError::Transport(error)) if error.contains("selected embedding key"))
    );
    let vectors = reader.vectors().unwrap();
    assert_eq!(vectors.status().unwrap().keys.len(), 1);
    assert_eq!(vectors.revision(&key).unwrap(), revision);
    assert_eq!(
        vectors
            .missing_chunks(&key, &[(input.hash.clone(), 2)])
            .unwrap()
            .remove(&input.hash)
            .unwrap(),
        std::collections::HashSet::from([1])
    );
}
