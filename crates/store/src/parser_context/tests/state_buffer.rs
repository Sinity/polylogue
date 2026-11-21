use super::*;
use polylogue_model::Json;
use serde_json::json;

fn key(namespace: &str, name: &str) -> Key {
    Key::State {
        namespace: namespace.into(),
        key: name.into(),
    }
}
fn state(context: &mut Overlay, key: &Key) -> Option<Json> {
    context.get(key).unwrap().map(|value| match value {
        Value::State { value } => value,
        _ => panic!("state changed kind"),
    })
}
fn put(context: &mut Overlay, key: &Key, value: Json) {
    context.put(key.clone(), Value::State { value }).unwrap();
}
fn usage(context: &Overlay) -> usize {
    let attempt = context.0.lock().unwrap();
    let size: i64 = attempt
        .scratch
        .pragma_query_value(None, "cache_size", |row| row.get(0))
        .unwrap();
    assert_eq!(size, -128);
    let used = attempt.states.usage();
    assert!(used + 128 * 1024 <= 256 * 1024);
    used
}

#[test]
fn repeated_state_updates_write_only_the_checkpoint_and_remove_actual_vm_work() {
    let run = |eager: bool| {
        let mut context = Overlay::temporary().unwrap();
        let key = key("sessions", "main");
        super::super::scratch::take_work();
        for revision in 0..1024 {
            assert_eq!(
                state(&mut context, &key),
                (revision > 0).then(|| json!({"revision":revision-1,"title":"synthetic"}))
            );
            let value = json!({"revision":revision,"title":"synthetic"});
            if eager {
                let attempt = context.0.lock().unwrap();
                super::super::scratch::write_value(&attempt.scratch, &key, &Value::State { value })
                    .unwrap();
            } else {
                put(&mut context, &key, value);
                usage(&context);
            }
        }
        let before = context.0.lock().unwrap().scratch.total_changes();
        assert_eq!(before, if eager { 1024 } else { 0 });
        let delta = context.checkpoint_delta(1024).unwrap();
        let work = super::super::scratch::take_work();
        (serde_json::to_value(delta).unwrap(), work)
    };
    let (expected, eager) = run(true);
    let (actual, buffered) = run(false);
    assert_eq!(actual, expected);
    assert_eq!((eager.state_encodes, buffered.state_encodes), (1024, 1));
    assert_eq!((eager.scratch_writes, buffered.scratch_writes), (1024, 1));
    assert_eq!((eager.state_decodes, buffered.state_decodes), (1024, 1));
    assert_eq!((eager.scratch_reads, buffered.scratch_reads), (1024, 1));
    assert!(buffered.read_vm_steps * 100 < eager.read_vm_steps);
    assert!(buffered.write_vm_steps * 100 < eager.write_vm_steps);
    println!(
        "STATE_BUFFER_WORK eager_reads={} buffered_reads={} eager_writes={} buffered_writes={} read_vm={}→{} write_vm={}→{}",
        eager.scratch_reads,
        buffered.scratch_reads,
        eager.scratch_writes,
        buffered.scratch_writes,
        eager.read_vm_steps,
        buffered.read_vm_steps,
        eager.write_vm_steps,
        buffered.write_vm_steps
    );
}

#[test]
fn eviction_oversize_clear_and_reinsert_match_independent_memory_state() {
    use polylogue_model::parser_context::Memory;
    let mut context = Overlay::temporary().unwrap();
    let mut expected = Memory::default();
    let keys: Vec<_> = (0..48)
        .map(|i| key("calls", &format!("call-{i:02}")))
        .collect();
    for (i, key) in keys.iter().enumerate() {
        let value = json!({"content":format!("{i}:{}","state ".repeat(1024))});
        expected
            .put(
                key.clone(),
                Value::State {
                    value: value.clone(),
                },
            )
            .unwrap();
        put(&mut context, key, value);
        usage(&context);
    }
    assert!(
        context.0.lock().unwrap().scratch.total_changes() > 0,
        "real eviction must spill"
    );
    let large = Json::String("oversized ".repeat(super::super::state_buffer::LIMIT));
    expected
        .put(
            keys[0].clone(),
            Value::State {
                value: large.clone(),
            },
        )
        .unwrap();
    put(&mut context, &keys[0], large);
    let other = key("names", "keep");
    put(&mut context, &other, json!("retained"));
    for key in &keys {
        assert_eq!(
            serde_json::to_value(context.get(key).unwrap()).unwrap(),
            serde_json::to_value(expected.get(key).unwrap()).unwrap()
        );
    }
    context.clear_state("calls").unwrap();
    expected.clear_state("calls").unwrap();
    assert_eq!(state(&mut context, &keys[0]), None);
    put(
        &mut context,
        &keys[0],
        json!([1.25, true, null, "reinserted"]),
    );
    expected
        .put(
            keys[0].clone(),
            Value::State {
                value: json!([1.25, true, null, "reinserted"]),
            },
        )
        .unwrap();
    let removed = key("calls", "removed");
    put(&mut context, &removed, json!(1));
    put(&mut context, &removed, Json::Null);
    assert_eq!(state(&mut context, &removed), None);
    let delta = context.checkpoint_delta(50).unwrap();
    assert_eq!(delta.cleared_states, ["calls"]);
    assert_eq!(usage(&context), 0);
    assert_eq!(state(&mut context, &other), Some(json!("retained")));
    for key in &keys {
        assert_eq!(
            serde_json::to_value(context.get(key).unwrap()).unwrap(),
            serde_json::to_value(expected.get(key).unwrap()).unwrap()
        );
    }
    assert!(context.checkpoint_delta(51).unwrap().entries.is_empty());
}

#[test]
fn failed_spill_or_checkpoint_retains_the_last_successful_private_state() {
    let mut context = Overlay::temporary().unwrap();
    let key = key("session", "last");
    put(&mut context, &key, json!(7));
    context
        .0
        .lock()
        .unwrap()
        .scratch
        .pragma_update(None, "query_only", true)
        .unwrap();
    assert!(
        context
            .put(
                key.clone(),
                Value::State {
                    value: Json::String("x".repeat(2 * super::super::state_buffer::LIMIT))
                }
            )
            .is_err()
    );
    assert_eq!(state(&mut context, &key), Some(json!(7)));
    assert!(context.checkpoint_delta(1).is_err());
    context.check_source("", None, "", 0).unwrap();
    assert_eq!(state(&mut context, &key), Some(json!(7)));
    context
        .0
        .lock()
        .unwrap()
        .scratch
        .pragma_update(None, "query_only", false)
        .unwrap();
    let delta = context.checkpoint_delta(1).unwrap();
    assert_eq!(delta.entries.len(), 1);
    assert!(matches!(&delta.entries[0].value, Reference::State { value } if *value == json!(7)));
}

#[test]
fn resumed_committed_lookup_keeps_snapshot_and_pending_state_does_not_publish() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    crate::Writer::start(&archive).unwrap().derive().unwrap();
    let writer = open_write(&archive.index_db(), cache::SETUP).unwrap();
    writer.execute_batch("INSERT INTO input_ref(pk,id,format) VALUES(1,'buffer-resume','claude-code'); INSERT INTO parse_state(input_pk,chain,status,parsed_upto,reading_version) VALUES(1,1,'parsing',1,'synthetic=1'); INSERT INTO parser_context_state VALUES(1,1,'synthetic=1',1);").unwrap();
    let key = key("sessions", "main");
    let save = |number: i64| {
        writer
            .execute(
                "INSERT OR REPLACE INTO parser_context_value VALUES(1,1,'sessions','main',?1)",
                [crate::encoding::pack_json(&json!(number)).unwrap()],
            )
            .unwrap();
    };
    save(10);
    let mut context = Overlay::open(&archive, "buffer-resume", Some(1), "synthetic=1", 1).unwrap();
    assert_eq!(state(&mut context, &key), Some(json!(10)));
    save(20);
    assert_eq!(state(&mut context, &key), Some(json!(10)));
    put(&mut context, &key, json!(30));
    context.clear_state("sessions").unwrap();
    assert_eq!(state(&mut context, &key), None);
    put(&mut context, &key, json!(40));
    let delta = context.checkpoint_delta(2).unwrap();
    assert_eq!((delta.base, delta.through, delta.entries.len()), (1, 2, 1));
    drop(context);
    let mut reopened = Overlay::open(&archive, "buffer-resume", Some(1), "synthetic=1", 1).unwrap();
    assert_eq!(state(&mut reopened, &key), Some(json!(20)));
}

#[test]
fn buffered_state_spill_keeps_the_existing_lookup_encoding() {
    for value in [
        Json::Null,
        json!(true),
        json!(1.50),
        json!({"nested":["Łódź",null,2]}),
        json!([]),
    ] {
        let value = Value::State { value };
        assert_eq!(
            super::super::scratch::encode(&value).unwrap(),
            crate::encoding::pack_json(&value).unwrap()
        );
    }
}

#[test]
fn retained_capacity_deep_values_and_large_keys_bypass_the_bounded_buffer() {
    let mut context = Overlay::temporary().unwrap();
    let mut spare = String::with_capacity(2 * super::super::state_buffer::LIMIT);
    spare.push_str("small value with large retained capacity");
    put(&mut context, &key("memory", "spare"), Json::String(spare));
    assert_eq!(usage(&context), 0);
    let mut deep = json!(1);
    for _ in 0..70 {
        deep = json!([deep]);
    }
    put(&mut context, &key("memory", "deep"), deep.clone());
    assert_eq!(usage(&context), 0);
    assert_eq!(state(&mut context, &key("memory", "deep")), Some(deep));
    let mut long = String::with_capacity(2 * super::super::state_buffer::LIMIT);
    long.push_str("key");
    // Preserve the caller's allocation. The ordinary borrowed-key helper
    // clones strings and therefore cannot exercise retained key capacity.
    context
        .put(
            Key::State {
                namespace: "memory".into(),
                key: long,
            },
            Value::State { value: json!(3) },
        )
        .unwrap();
    assert_eq!(usage(&context), 0);
    assert_eq!(context.checkpoint_delta(1).unwrap().entries.len(), 3);
}
