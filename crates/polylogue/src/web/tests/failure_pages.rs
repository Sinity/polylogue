use super::*;

#[test]
fn unknown_paths_have_a_reader_page() {
    let r = reader();
    rt().block_on(async {
        let (status, headers, body) =
            send(&r, authed(get("/nosuch")).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(headers[header::CONTENT_TYPE], "text/html; charset=utf-8");
        assert!(body.contains(
            "<p>/nosuch is not a page. Use Home, Search or Cost and usage in the navigation.</p>"
        ));
        assert!(body.contains("href=\"/cost\""));
        assert!(body.contains("href=\"/search\""));
    });
}

#[test]
fn a_late_reader_page_keeps_the_deadline_hint_and_next_command() {
    let r = reader();
    rt().block_on(async {
        let (status, _, body) = send(
            &r,
            authed(get("/rows?selection=sessions"))
                .header("x-polylogue-deadline-ms", "0")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
        let ids = body.split("--request-id ").skip(1)
            .filter_map(|text| text.split_once(' ').map(|(id, _)| id.to_owned()))
            .collect::<Vec<_>>();
        let mut body = body;
        for id in ids {
            body = body.replace(&format!("--request-id {id}"), "--request-id REQUEST");
        }
        assert!(
            body.contains("<p>the read deadline was exceeded</p>"),
            "{body}"
        );
        assert!(
            body.contains("<p>rerun with `--deadline none`, or narrow to one session</p>"),
            "{body}"
        );
        assert!(
            body.contains(
                "complete without a deadline: <code>polylogue --request-id REQUEST --deadline none -- sessions</code>"
            ),
            "{body}"
        );
        assert!(
            body.contains("Command: <code>polylogue --request-id REQUEST -- sessions</code>"),
            "{body}"
        );
    });
}

#[test]
fn stale_sign_in_keeps_the_requested_session() {
    let r = reader();
    rt().block_on(async {
        let (status, headers, _) = send(&r, get("/s/claude-code:s1?view=dialogue")
            .header(header::COOKIE, "polylogue_abc123def456=stale").body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(headers[header::LOCATION], "/auth");
        let next_cookie = headers[header::SET_COOKIE].to_str().unwrap();
        assert_eq!(next_cookie, "polylogue_abc123def456_page=%2Fs%2Fclaude-code%3As1%3Fview%3Ddialogue; Path=/auth; HttpOnly; SameSite=Strict");
        let (_, _, body) = send(&r, get(headers[header::LOCATION].to_str().unwrap()).body(Body::empty()).unwrap()).await;
        assert!(body.contains("Sign-in is needed. The daemon may have restarted. Run <code>polylogue web</code> again to sign in and return to your page."));
        let (status, signed_in, _) = send(&r, own_origin(post("/auth"))
            .header(header::COOKIE, next_cookie.split(';').next().unwrap())
            .body(Body::from(format!("token={TOKEN}"))).unwrap()).await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        assert_eq!(signed_in["x-polylogue-return-to"], "%2Fs%2Fclaude-code%3As1%3Fview%3Ddialogue");
        assert_eq!(signed_in.get_all(header::SET_COOKIE).iter().nth(1).unwrap(), "polylogue_abc123def456_page=; Path=/auth; HttpOnly; SameSite=Strict; Max-Age=0");
        let token_cookie = signed_in[header::SET_COOKIE].to_str().unwrap().split(';').next().unwrap();
        let (status, _, body) = send(&r, get("/s/claude-code:s1?view=dialogue")
            .header(header::COOKIE, token_cookie).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    });
}

#[test]
fn partial_pages_show_all_missing_reasons_notes_and_next_calls() {
    let answer = command_surface::Answer {
        outcome: command_surface::Outcome::Degraded {
            value: serde_json::json!({"reply":{"columns":[],"rows":[],"count":0}}),
            missing: vec![
                "The index is catching up.".into(),
                "Usage is unavailable.".into(),
            ],
        },
        envelope: command_surface::Envelope {
            notes: vec!["Counts are lower bounds.".into()],
            ..Default::default()
        },
    }
    .with_next(
        "Read sessions",
        command_surface::RunnableRequest::new(
            "rows",
            serde_json::json!({"selection":"sessions"}),
            "next",
        ),
    );
    let response = answer_page(&[(&answer, "polylogue sessions")], || {
        crate::web::pages::search("sessions", &answer.value().unwrap()["reply"], None)
    });
    rt().block_on(async {
        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap();
        assert!(body.contains("<p>Partial results</p><ul><li>The index is catching up.</li><li>Usage is unavailable.</li></ul>"), "{body}");
        assert!(body.contains("<p class=\"note\">Counts are lower bounds.</p>"));
        assert!(body.contains("<p>Read sessions: <code>polylogue --request-id next -- sessions</code></p>"));
    });
}

#[test]
fn listings_and_read_pages_name_an_unparsed_file() {
    let r = reader();
    let bytes = polylogue_testkit::claude_user("pending", "s-pending", "not parsed yet");
    for (origin, path) in [
        ("synthetic-unregistered", "/home/op/unknown.jsonl"),
        ("claude-code", "/home/op/pending.jsonl"),
    ] {
        let mut batch =
            polylogue_store::fixture::writer::step(origin, origin, path, Default::default(), 0, 0);
        batch.length = bytes.len() as u64;
        batch.parse.clear();
        let prepared = r.w.daemon.archive.prepare_bytes(bytes.as_bytes()).unwrap();
        r.w.daemon
            .writes
            .as_ref()
            .unwrap()
            .store_bytes(prepared, batch)
            .unwrap();
    }
    rt().block_on(async {
        for path in [
            "/search?q=sessions",
            "/rows?selection=sessions",
            "/s/claude-code:s1",
            "/cost",
            "/day/2026-03-01",
            "/",
        ] {
            let (status, _, body) = send(&r, authed(get(path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::OK, "{path}: {body}");
            let details = body
                .split_once("aria-label=\"Answer details\">")
                .unwrap()
                .1
                .split_once("</section>")
                .unwrap()
                .0;
            assert!(
                details.contains("<p>Partial results</p>"),
                "{path}: {details}"
            );
            assert!(
                details
                    .contains("<li>some stored inputs have no recorded reading declaration</li>"),
                "{path}: {details}"
            );
            assert!(
                details.contains("<li>some stored inputs have not been parsed</li>"),
                "{path}: {details}"
            );
        }
    });
}

#[test]
fn failure_pages_show_the_suggested_shorter_period() {
    let failure = crate::protocol::failure(
        polylogue_model::Code::DeadlineExceeded,
        "the read deadline was exceeded",
    )
    .with_hint("restrict the query to a shorter time range or one session")
    .with_suggestion(command_surface::Suggestion::Query(
        "sessions since:7d".into(),
    ));
    let answer = Answer::<Value>::error(failure);
    let status = Answer::ok(Value::Null);
    let response = answer_page(
        &[
            (&status, "polylogue status"),
            (&answer, "polylogue -- sessions"),
        ],
        String::new,
    );
    rt().block_on(async {
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        let body = String::from_utf8(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert_copy_command(&body, "polylogue -- sessions");
        assert!(body.contains("<p><code>deadline_exceeded</code></p>"));
        assert!(body.contains("<p>restrict the query to a shorter time range or one session</p>"));
        assert!(body.contains("<p>Try <a href=\"/search?q=sessions%20since%3A7d\"><code>sessions since:7d</code></a></p>"));
    });
}

#[test]
fn failed_search_commands_keep_option_words_in_the_selection() {
    let r = reader();
    rt().block_on(async {
        let (status, _, body) = send(
            &r,
            authed(get("/search?q=--bad%20%28"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_copy_command(&body, "polylogue --why -- '--bad ('");
    });
}
