-- A default page reads its unit's rows in the order of its first terms
-- (`time IS NULL, time, ...`), so these indexes carry those exact
-- expressions: the page stops after its rows instead of sorting the table.
CREATE INDEX IF NOT EXISTS message_order ON message(at IS NULL, at, order_input_id, seq, record_position, id);
CREATE INDEX IF NOT EXISTS tool_call_order ON tool_call(at IS NULL, at, call_block_pk);
CREATE INDEX IF NOT EXISTS event_order ON event(at IS NULL, at, id);
CREATE INDEX IF NOT EXISTS message_at ON message(at, id);
CREATE INDEX IF NOT EXISTS message_session_at ON message(session_pk, at, id);
CREATE INDEX IF NOT EXISTS message_model_at ON message(model, at, id) WHERE model IS NOT NULL;
CREATE INDEX IF NOT EXISTS message_session_model_at ON message(session_pk, model, at, id);
CREATE INDEX IF NOT EXISTS event_at ON event(at, id);
CREATE INDEX IF NOT EXISTS block_original_ordinal ON block(message_pk,coalesce(ordinal,cast(idx AS TEXT)));

-- The older archive row parser emits ordinary source events. These views
-- derive their original notes without changing the canonical assertions.
CREATE VIEW IF NOT EXISTS imported_note_raw AS
SELECT e.pk, e.input_pk, e.source_chain, i.id AS input,
       json_extract(pl_block(e.data),'$.row.id') AS id,
       json_extract(pl_block(e.data),'$.row.first_note') AS first_note,
       json_extract(pl_block(e.data),'$.row.kind') AS kind,
       json_extract(pl_block(e.data),'$.row.key') AS key,
       json_extract(pl_block(e.data),'$.row.value') AS value,
       json_extract(pl_block(e.data),'$.row.author') AS author,
       json_extract(pl_block(e.data),'$.row.created_at') AS created_at,
       json_extract(pl_block(e.data),'$.row.commit_seq') AS commit_seq
FROM event e JOIN input_ref i ON i.pk=e.input_pk
WHERE e.kind='imported_note' AND i.format='polylogue-archive-rows';
CREATE VIEW IF NOT EXISTS imported_note_target AS
SELECT e.input_pk,e.source_chain,
       json_extract(pl_block(e.data),'$.row.note_id') AS note_id,
       json_extract(pl_block(e.data),'$.row.ref') AS ref
FROM event e JOIN input_ref i ON i.pk=e.input_pk
WHERE e.kind='imported_note_target' AND i.format='polylogue-archive-rows';
CREATE VIEW IF NOT EXISTS imported_note_relation AS
SELECT e.input_pk,e.source_chain,
       json_extract(pl_block(e.data),'$.row.note_id') AS note_id,
       json_extract(pl_block(e.data),'$.row.kind') AS kind,
       json_extract(pl_block(e.data),'$.row.other') AS other
FROM event e JOIN input_ref i ON i.pk=e.input_pk
WHERE e.kind='imported_note_relation' AND i.format='polylogue-archive-rows';
CREATE VIEW IF NOT EXISTS imported_note AS
SELECT n.*,
       (SELECT group_concat(ref,' ') FROM
           (SELECT t.ref FROM imported_note_target t WHERE t.input_pk=n.input_pk
            AND t.source_chain IS n.source_chain AND t.note_id=n.id ORDER BY t.ref)) AS target,
       NOT EXISTS(SELECT 1 FROM imported_note_raw later WHERE later.input_pk=n.input_pk
           AND later.source_chain IS n.source_chain AND later.first_note=n.first_note
           AND later.commit_seq>n.commit_seq) AS current,
       EXISTS(SELECT 1 FROM imported_note_relation w WHERE w.input_pk=n.input_pk
           AND w.source_chain IS n.source_chain AND w.kind='withdraws'
           AND w.other='note:'||n.first_note) AS withdrawn
FROM imported_note_raw n;
