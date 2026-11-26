use super::*;

/// A reply delivered one byte per read.
struct Trickle<'a>(&'a [u8]);
impl Read for Trickle<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let count = output.len().min(self.0.len()).min(1);
        output[..count].copy_from_slice(&self.0[..count]);
        self.0 = &self.0[count..];
        Ok(count)
    }
}
impl Receive for Trickle<'_> {
    fn wait_until(&mut self, _: Option<Instant>) -> io::Result<()> {
        Ok(())
    }
}

fn refusal() -> Answer<Value> {
    command_surface::Answer {
        outcome: command_surface::Outcome::Error(crate::protocol::failure(
            polylogue_model::Code::ScopeChanged,
            "scope changed: λ",
        )),
        envelope: command_surface::Envelope {
            notes: vec!["acquire a new reader".into()],
            next: vec![],
            ..Default::default()
        },
    }
}

fn frames(events: &[(&str, Value)], answer: Option<&Answer<Value>>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (name, value) in events {
        crate::protocol::stream::event(&mut bytes, name, value).unwrap();
    }
    if let Some(answer) = answer {
        crate::protocol::stream::event(&mut bytes, "answer", answer).unwrap();
    }
    bytes
}

#[test]
fn a_finite_reply_is_one_answer_frame_kept_whole_across_split_reads() {
    let bytes = frames(&[], Some(&refusal()));
    let answer = receive(&mut Trickle(&bytes), None, |_| panic!("no events")).unwrap();
    assert_eq!(answer, Some(refusal()));
}

#[test]
fn events_arrive_in_order_before_the_answer_that_ends_the_reply() {
    let bytes = frames(
        &[
            ("begin", serde_json::json!({"title":"λ"})),
            ("record", serde_json::json!({"n":1})),
        ],
        Some(&refusal()),
    );
    let mut events = Vec::new();
    let answer = receive(&mut Trickle(&bytes), None, |event| {
        events.push(event.event);
        Ok(false)
    })
    .unwrap();
    assert_eq!(events, ["begin", "record"]);
    assert_eq!(answer, Some(refusal()));
}

#[test]
fn a_reply_that_ends_without_its_answer_reports_the_daemon_gone() {
    let bytes = frames(&[("begin", serde_json::json!({}))], None);
    let answer = receive(&mut Trickle(&bytes), None, |_| Ok(false))
        .unwrap()
        .unwrap();
    let failure = answer.failure().unwrap();
    assert_eq!(failure.code.as_deref(), Some("not_running"));
    assert_eq!(answer.exit_code(), 6);
}

#[test]
fn a_caller_that_stops_at_an_event_reads_no_further() {
    let bytes = frames(
        &[
            ("record", serde_json::json!(1)),
            ("record", serde_json::json!(2)),
        ],
        Some(&refusal()),
    );
    let mut seen = 0;
    let answer = receive(&mut Trickle(&bytes), None, |_| {
        seen += 1;
        Ok(true)
    })
    .unwrap();
    assert_eq!((seen, answer), (1, None));
}

#[test]
fn an_async_caller_that_cancels_hangs_up_its_connection() {
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::at(dir.path());
    std::fs::create_dir_all(archive.socket().parent().unwrap()).unwrap();
    let listener = std::os::unix::net::UnixListener::bind(archive.socket()).unwrap();
    // A daemon that takes the request and never answers; its read ends when
    // the caller hangs up.
    let daemon = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut rest = Vec::new();
        (&stream).read_to_end(&mut rest).unwrap();
    });
    let request = request("status", serde_json::json!({}), "cancelled-call".into());
    let cancelled = tokio_util::sync::CancellationToken::new();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let answer = runtime
        .block_on(async {
            let call = call_ready_as_bounded(&archive, &request, &cancelled, this_process(), None);
            let cancel = async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                cancelled.cancel();
                std::future::pending::<()>().await
            };
            tokio::select! {
                answer = call => answer,
                () = cancel => unreachable!(),
            }
        })
        .unwrap();
    assert_eq!(
        answer.failure().and_then(|failure| failure.code.as_deref()),
        Some("cancelled"),
        "{answer:?}"
    );
    daemon.join().unwrap();
}
