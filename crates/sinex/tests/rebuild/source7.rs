//! Source7's exact owner declaration was generated from Polylogue6c70b4d9e249.
//! These are producer inputs, never a production reader for the retired bodies.
use super::*;
use polylogue_model::{BlobHash, Message, MessageRow, SessionId};
use polylogue_store::primary::{Material, Product, Reading};
use polylogue_store::{InputKey, ParseStep};
use sinex_protocol::{Ingress, ProducerEvent};

fn declaration() -> sinex_model::SourceDeclaration {
    let declaration: sinex_model::SourceDeclaration =
        serde_json::from_str(include_str!("source7.json")).unwrap();
    assert_eq!(declaration.source.version, 7);
    for (name, version) in [
        ("polylogue.message", 6),
        ("polylogue.session", 5),
        ("polylogue.parse_progress", 6),
        ("polylogue.input_observation", 6),
    ] {
        assert_eq!(
            declaration
                .kinds
                .iter()
                .find(|kind| kind.kind.name == name)
                .unwrap()
                .kind
                .version,
            version
        );
    }
    declaration
}

fn running(home: &polylogue_testkit::ClaudeHome) -> Running {
    let native = native::Native::start(
        home.dir.path(),
        vec![
            serde_json::to_value(declaration()).unwrap(),
            serde_json::to_value(polylogue_sinex::kinds::source_declaration()).unwrap(),
        ],
    );
    Running {
        config: Config {
            mode: Mode::Primary,
            nats: native.nats.url().into(),
            password_file: None,
            subject_root: "rebuild".into(),
            producer: "polylogue".into(),
            store: native.store.clone(),
            upload_root: None,
        },
        socket: native.socket.clone(),
        native,
    }
}

/// The old producer's immutable message envelope, validated by its exact schema
/// at native admission. Expected local answers come from the standalone parser.
fn old_message(input: &InputKey, message: Message, offset: u64, length: u64) -> Product {
    Product::Message {
        reading: Reading {
            source_length: Some(length),
            delivery: None,
            input: input.clone(),
            chain: 1,
            reading_version: "source7-parser".into(),
            start: 0,
            parsed_upto: length,
            progress: polylogue_model::batch::StepProgress::AtEnd,
        },
        facts: Box::new(MessageRow {
            offset,
            end: length - 1,
            id_kind: Default::default(),
            page_text: false,
            message,
        }),
        order: None,
        projection: None,
        native_fields: None,
        declared_injections: None,
        declared_attachment_bindings: None,
    }
}

fn admit_originals(running: &Running, parts: &[(&str, Option<Message>)]) -> (InputKey, u64) {
    let declaration = declaration();
    let source = declaration.source.clone();
    let input = InputKey {
        id: "source7-original".into(),
        format: "claude-code".into(),
        route: polylogue_model::Route::Live,
    };
    let root = tempfile::tempdir().unwrap();
    let mut length = 0;
    let mut previous = None;
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let client = async_nats::connect(running.native.nats.url()).await.unwrap();
        let subjects = sinex_protocol::Subjects::new("rebuild");
        let mut acks = client.subscribe(subjects.ack("polylogue")).await.unwrap();
        client.flush().await.unwrap();
        for (index, (bytes, message)) in parts.iter().enumerate() {
            let offset = length;
            length += bytes.len() as u64;
            let material = sinex_model::MaterialId::new();
            let hash = BlobHash::of(bytes.as_bytes());
            let blob = format!("blake3:{hash}");
            let raw = Ingress::Material { frame:format!("source7-material-{index}"),
                material:Box::new(sinex_protocol::MaterialFrame {
                    source:source.clone(), id:material,
                    input:sinex_model::InputRef {kind:"claude-code".into(),locator:"/other-host/session.jsonl".into(),host:"fixture-host".into()},
                    blob:blob.clone(),length:bytes.len() as u64,chunks:vec![],parser:"polylogue-claude-code".into(),
                    parser_version:3,part_index:index as u64,previous_part:previous,manifest:Default::default(),
                }) };
            sinex_transport::frame_bytes::write(root.path(),"polylogue",raw.frame(),&blob,bytes.len() as u64,bytes.as_bytes()).unwrap();
            sinex_transport::producer::transfer_frame(&client,&subjects,"polylogue",&mut acks,root.path(),&raw,Duration::from_secs(30)).await.unwrap();
            let receipt = sinex_transport::producer::submit_frame(&client,&subjects,"polylogue",&mut acks,&raw,Duration::from_secs(30)).await.unwrap();
            assert_eq!(receipt.refusal(),None,"{receipt:?}");
            let sinex_protocol::FrameResult::Admitted {material_extents,..} = receipt.result else { panic!("raw admission"); };
            let rows = serde_json::from_value(serde_json::json!({
                "input":input.id,"format":input.format,
                "revision":{"number":index+1,"produced":if index==0 {"first"} else {"appended"},"chain":1,"length":length},
                "extents":[{"chain":1,"offset":offset,"length":bytes.len(),"blob":hash,"blob_offset":0}],
                "observation":{"id":format!("source7-observation-{index}"),"machine":"other-machine","host":"other-host",
                    "path":"/other-host/session.jsonl","path_bytes":b"/other-host/session.jsonl",
                    "class":if index==0 {"first"} else {"appended"},"revision":index+1,"length":length,
                    "device":null,"inode":null,"size":null,"mtime_ns":null,"ctime_ns":null,
                    "polylogue_version":"synthetic-source7","seen_at":index+1,"details_json":null}
            })).unwrap();
            let product = Product::Input { rows:Box::new(rows),material:Some(Material {
                id:material.to_string(),hash,length:bytes.len() as u64,extents:material_extents,
            }) };
            let event = |kind: &str, record: Product, output_index| ProducerEvent {
                kind:declaration.kinds.iter().find(|decl| decl.kind.name==kind).unwrap().kind.clone(),
                payload:serde_json::json!({"record_key":format!("source7-{kind}-{index}"),"record":record}),
                location:Some(sinex_model::RecordLocation {material,anchor:0}),output_index,parents:vec![],reading_order:None,class:sinex_model::Class::Guarded,
            };
            let mut events = vec![event("polylogue.input_observation",product,0)];
            if let Some(message) = message {
                events.push(event("polylogue.message",old_message(&input,message.clone(),offset,length),1));
            }
            let interpreted = Ingress::Events {frame:format!("source7-facts-{index}"),source:Some(source.clone()),coverage:None,events,failures:vec![]};
            let receipt = sinex_transport::producer::submit_frame(&client,&subjects,"polylogue",&mut acks,&interpreted,Duration::from_secs(30)).await.unwrap();
            assert_eq!(receipt.refusal(),None,"{receipt:?}");
            previous=Some(material);
        }
    });
    (input, length)
}

fn expected(home: &polylogue_testkit::ClaudeHome, text: &str) -> Vec<Message> {
    home.append("session.jsonl", text);
    let (mut engine, lock) = home.engine(1 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    drop(engine);
    drop(lock);
    home.reads()
        .reader()
        .unwrap()
        .messages(
            &SessionId::parse("claude-code:source7-session").unwrap(),
            false,
        )
        .unwrap()
}

fn adoption(partial: bool) {
    let home = polylogue_testkit::ClaudeHome::new();
    let ordinary = polylogue_testkit::ClaudeHome::new();
    let bytes = polylogue_testkit::claude_user(
        "source7-message",
        "source7-session",
        "retained original words",
    );
    let wanted = expected(&ordinary, &bytes);
    assert_eq!(wanted.len(), 1);
    let running = running(&home);
    let mut parts = vec![(bytes.as_str(), Some(wanted[0].clone()))];
    if partial {
        parts.push(("{\"type\":\"assistant\"", None));
    }
    let (input, length) = admit_originals(&running, &parts);
    assert_eq!(
        sinex_rows(
            &running.socket,
            "events where event_type = polylogue.message"
        ),
        1
    );
    let sdk = connect(&running.config);
    let (archive, _lock) = Archive::open_primary(&home.root, &running.config.store).unwrap();
    let mut config = home.config(1 << 20);
    config.reads_sources = false;
    let mut engine = Engine::on_sinex(
        config,
        archive.clone(),
        sdk.clone(),
        polylogue_testkit::stopped_attempt(),
    )
    .unwrap();
    let commits = engine.writer().unwrap().completion_commits();
    confirm_capture(
        &mut engine,
        &commits,
        Instant::now(),
        (&archive, &sdk, &running),
        |pass| {
            let pool = ReadPool::open(&archive)?;
            let reader = pool.reader()?;
            Ok(!pass.capture_pending
                && reader.messages(
                    &SessionId::parse("claude-code:source7-session").unwrap(),
                    false,
                )? == wanted)
        },
        String::new,
    );
    let pool = ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let progress = reader.input_progress(&input.id).unwrap().unwrap();
    assert_eq!(progress.length, length);
    assert_eq!(progress.parsed_upto, bytes.len() as u64);
    assert_eq!(progress.waiting, partial);
    assert_eq!(reader.confirmed_refusals().unwrap().0, 0);
    assert_eq!(
        sinex_rows(
            &running.socket,
            "events where event_type = polylogue.message"
        ),
        1
    );
    assert_eq!(running.native.sql("SELECT count(*) FROM core.events WHERE event_type='polylogue.message' AND kind_version=8 AND source_version=8").trim(),"1");
    assert_eq!(running.native.sql("SELECT count(*) FROM core.events WHERE event_type='polylogue.message' AND kind_version=6").trim(),"0");
    assert_eq!(running.native.sql("SELECT count(*) FROM core.ingress_frames WHERE frame LIKE 'source7-material-%' AND source_name='polylogue' AND source_version=7 AND material_id IS NOT NULL").trim(),parts.len().to_string());
    assert_eq!(running.native.sql("SELECT fingerprint FROM registry.source_declarations WHERE name='polylogue' AND version=7").trim(),declaration().fingerprint());
}

#[test]
#[ignore = "needs the matching Sinex daemon; set SINEXD"]
fn source7_complete_original_adopts_the_current_native_output_set() {
    adoption(false);
}

#[test]
#[ignore = "needs the matching Sinex daemon; set SINEXD"]
fn source7_waiting_tail_adopts_only_when_previous_outputs_fit_the_parsed_prefix() {
    adoption(true);
}

#[test]
#[ignore = "needs the matching Sinex daemon; set SINEXD"]
fn source7_facts_beyond_a_refused_reading_frontier_remain_live() {
    let home = polylogue_testkit::ClaudeHome::new();
    let ordinary = polylogue_testkit::ClaudeHome::new();
    let first = polylogue_testkit::claude_user("first", "source7-session", "first original");
    let second = polylogue_testkit::claude_user("second", "source7-session", "later original");
    let wanted = expected(&ordinary, &format!("{first}{second}"));
    assert_eq!(wanted.len(), 2);
    let running = running(&home);
    let (input, length) = admit_originals(
        &running,
        &[
            (&first, Some(wanted[0].clone())),
            (&second, Some(wanted[1].clone())),
        ],
    );
    let sdk = connect(&running.config);
    let (archive, _lock) = Archive::open_primary(&home.root, &running.config.store).unwrap();
    let engine = Engine::on_sinex(
        home.config(1 << 20),
        archive.clone(),
        sdk.clone(),
        polylogue_testkit::stopped_attempt(),
    )
    .unwrap();
    let commits = engine.writer().unwrap().completion_commits();
    let started = Instant::now();
    loop {
        let ready =
            ReadPool::open(&archive).and_then(|pool| pool.reader()?.input_progress(&input.id));
        if ready.as_ref().is_ok_and(|progress| {
            progress
                .as_ref()
                .is_some_and(|progress| progress.length == length)
        }) {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "source7 snapshot: {ready:?}"
        );
        let _ = commits.recv_timeout(Duration::from_millis(20));
    }
    let before = native_rows(
        &running,
        "events where event_type = polylogue.message | fields id",
    );
    assert_eq!(before.len(), 2);
    let mut step = ParseStep::new(
        Default::default(),
        Default::default(),
        0..first.len() as u64,
        false,
    );
    step.reading_version = "refused-new-parser".into();
    step.progress = polylogue_model::batch::StepProgress::Refused {
        reason: "synthetic reader stops before the later original".into(),
    };
    let mut reading = polylogue_store::writer::ChainReading::new().unwrap();
    reading.push(&mut step).unwrap();
    let error = engine
        .writer()
        .unwrap()
        .replace_chain(input.clone(), 1, length, reading)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("previous facts require source bytes"),
        "{error}"
    );
    assert_eq!(
        native_rows(
            &running,
            "events where event_type = polylogue.message | fields id"
        ),
        before
    );
    assert!(
        ReadPool::open(&archive)
            .unwrap()
            .reader()
            .unwrap()
            .parser_checkpoint(&input.id, 1, "refused-new-parser")
            .unwrap()
            .is_none()
    );
    assert_eq!(
        running
            .native
            .sql("SELECT count(*) FROM core.replayed_sequences")
            .trim(),
        "0"
    );
}
