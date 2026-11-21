use super::*;

fn shared_responses() -> Vec<Message> {
    let thinking = "shared reasoning text ".repeat(4096);
    (0..8)
        .map(|i| {
            let mut message = text_message(
                &format!("response-{i}"),
                "s",
                Role::Assistant,
                &format!("unique response {i}"),
                Some("end_turn"),
            );
            message.blocks.insert(
                0,
                Block::Thinking {
                    text: Some(thinking.clone()),
                    signature: None,
                    redacted: false,
                },
            );
            message
        })
        .collect()
}

fn prepare(conn: &Connection, b: &mut Batch) {
    resolve::prepare_index(conn, b, 1, &resolve::IndexUpkeep::PerTransaction).unwrap();
}

#[test]
fn shared_response_text_in_one_transaction_avoids_materialization_until_evicted() {
    for evict in [false, true] {
        let mut conn = database();
        let messages = shared_responses();
        let entries = messages
            .iter()
            .enumerate()
            .map(|(i, message)| reference(message, &format!("response-{i}")))
            .collect();
        let mut b = reading("shared", messages.clone(), entries);
        prepare(&conn, &mut b);
        // One mark cannot retain a shared text beside each response's unique
        // text. Eviction is a positive control for full stored validation.
        let budget = if evict {
            1
        } else {
            crate::archive::MemoryBudget::default().keys() / 3
        };
        let texts = crate::text_rows::TextIds::new(&conn, false, budget).unwrap();
        let (result, work) =
            apply_with_texts(&mut conn, b, &resolve::IndexUpkeep::PerTransaction, &texts);
        assert_eq!(result.unwrap(), messages.len() as u64);
        let reads = if evict { messages.len() as u64 - 1 } else { 0 };
        assert_eq!(
            (work.message_reads, work.text_queries, work.text_rows),
            (reads, reads, 2 * reads),
            "evict={evict}: {work:?}"
        );
        for (i, message) in messages.iter().enumerate() {
            let hash = message.content_hash().unwrap();
            let (stored, layout) = crate::versions::read_message(&conn, &hash.0)
                .unwrap()
                .unwrap();
            assert_eq!(&stored, message);
            assert_eq!(layout.source_version, hash);
            assert_eq!(layout.block_ordinals, [0, 1]);
            let (version, offset, end): (Vec<u8>, i64, i64) = conn
                .query_row(
                    "SELECT v.version_hash,c.source_start,c.source_end FROM parser_context_response c
                     JOIN message_version v ON v.pk=c.version_pk WHERE c.key=?1",
                    [format!("response-{i}")],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
                .unwrap();
            assert_eq!((version, offset, end), (hash.0.to_vec(), 0, 100));
        }
        assert_eq!(
            conn.scalar::<i64>("SELECT count(*) FROM text", []).unwrap(),
            messages.len() as i64 + 1
        );
        assert_eq!(
            conn.scalar::<i64>("SELECT count(*) FROM pragma_foreign_key_check", [])
                .unwrap(),
            0
        );
        println!("transaction response materialization: evict={evict} work={work:?}");
    }
}

#[test]
fn failed_shared_response_context_retries_with_the_same_text_id_owner() {
    let mut conn = database();
    let texts = crate::text_rows::TextIds::new(
        &conn,
        false,
        crate::archive::MemoryBudget::default().keys() / 3,
    )
    .unwrap();
    let messages = shared_responses();
    let entries: Vec<_> = messages
        .iter()
        .enumerate()
        .map(|(i, message)| reference(message, &format!("response-{i}")))
        .collect();
    let mut wrong = entries.clone();
    if let Reference::Response { end, .. } = &mut wrong.last_mut().unwrap().value {
        *end = 101;
    }
    let mut bad = reading("retry", messages.clone(), wrong);
    prepare(&conn, &mut bad);
    assert!(
        apply_with_texts(
            &mut conn,
            bad,
            &resolve::IndexUpkeep::PerTransaction,
            &texts,
        )
        .0
        .is_err()
    );
    for table in ["message", "message_version", "text", "parser_context_state"] {
        assert_eq!(
            conn.scalar::<i64>(&format!("SELECT count(*) FROM {table}"), [])
                .unwrap(),
            0,
            "rolled back {table}"
        );
    }
    let mut retry = reading("retry", messages.clone(), entries);
    prepare(&conn, &mut retry);
    let (result, work) = apply_with_texts(
        &mut conn,
        retry,
        &resolve::IndexUpkeep::PerTransaction,
        &texts,
    );
    assert_eq!(result.unwrap(), messages.len() as u64);
    assert_eq!((work.message_reads, work.text_queries), (0, 0));
    for message in messages {
        assert_eq!(
            crate::versions::read_message(&conn, &message.content_hash().unwrap().0)
                .unwrap()
                .unwrap()
                .0,
            message
        );
    }
}
