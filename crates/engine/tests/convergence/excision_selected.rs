use super::*;
use polylogue_store::{Delivery, UserRequest, UserWrite};

#[test]
fn input_excision_owns_text_shared_only_by_its_removed_sessions() {
    for (survivor, title_only) in [(false, false), (true, false), (false, true), (true, true)] {
        let shared = "shared synthetic evidence held by removed sessions";
        let message = |id| {
            if title_only {
                format!("separate synthetic message {id}")
            } else {
                shared.to_owned()
            }
        };
        let donor = ClaudeHome::new();
        for (n, id) in [(1, "first"), (2, "second"), (3, "third")] {
            donor.append(
                &format!("{id}.jsonl"),
                &user(&format!("u{n}"), id, &message(id)),
            );
        }
        let (mut source, _source_lock) = donor.engine(64 << 20);
        assert!(source.converge().unwrap().errors.is_empty());
        let export = |id| {
            let mut doc = donor.export(&format!("claude-code:{id}"));
            if title_only {
                doc.session
                    .offer_title(shared, polylogue_model::TitleSource::Custom);
            }
            doc
        };
        let selected = donor.dir.path().join("selected.json");
        fs::write(
            &selected,
            polylogue_model::export::Document::new(vec![export("first"), export("second")])
                .to_json()
                .unwrap(),
        )
        .unwrap();
        let w = ClaudeHome::new();
        let (mut engine, _lock) = w.engine(64 << 20);
        engine.import(std::slice::from_ref(&selected)).unwrap();
        if survivor {
            let kept = donor.dir.path().join("kept.json");
            fs::write(
                &kept,
                polylogue_model::export::Document::new(vec![export("third")])
                    .to_json()
                    .unwrap(),
            )
            .unwrap();
            engine.import(&[kept]).unwrap();
        }
        let input = w
            .reads()
            .reader()
            .unwrap()
            .known_inputs(&engine.config.machine)
            .unwrap()
            .into_iter()
            .find(|input| Path::new(&input.path) == selected)
            .unwrap()
            .input_id;
        engine
            .writer()
            .unwrap()
            .user(UserRequest::now(
                "shared-summary".into(),
                UserWrite::Deliver(Delivery {
                    recipient_kind: "operator".into(),
                    recipient_id: None,
                    sender: "cli".into(),
                    event: "recall".into(),
                    cwd: None,
                    text: shared.into(),
                    refs: vec!["session:claude-code:first".into()],
                    left_out: vec![],
                }),
            ))
            .unwrap();
        for dry_run in [true, false] {
            let done = engine
                .excise(ExciseRequest {
                    id: "remove-selected".into(),
                    scope: ExciseScope::Input(input.clone()),
                    reason: None,
                    dry_run,
                })
                .unwrap();
            assert_eq!(done.sessions.len(), 2);
            assert_eq!(done.messages, 2);
            assert_eq!(
                done.deliveries_redacted,
                u64::from(!survivor),
                "survivor={survivor}, title={title_only}, preview={dry_run}"
            );
            let expected = if dry_run || survivor {
                shared.into()
            } else {
                " ".repeat(shared.len())
            };
            let canonical = rusqlite::Connection::open(Archive::at(&w.root).archive_db()).unwrap();
            let hash: Vec<u8> = canonical
                .query_row(
                    "SELECT content_hash FROM delivery WHERE id='shared-summary'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(hash, blake3::hash(expected.as_bytes()).as_bytes());
        }
        assert!(w.texts("first").is_empty());
        assert!(w.texts("second").is_empty());
        if survivor {
            assert_eq!(w.texts("third"), [message("third")]);
        }
    }
}
