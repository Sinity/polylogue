ALTER TABLE observation ADD COLUMN commit_seq INTEGER NOT NULL DEFAULT 0;
ALTER TABLE delivery_event ADD COLUMN commit_seq INTEGER NOT NULL DEFAULT 0;
-- Index the per-row backfill lookup; observation rowid is its local insertion order.
CREATE TEMP TABLE append_fact_order(kind TEXT NOT NULL,key TEXT NOT NULL,seq INTEGER NOT NULL,
 PRIMARY KEY(kind,key)) WITHOUT ROWID;
INSERT INTO append_fact_order
SELECT kind,key,ROW_NUMBER() OVER (ORDER BY kind,ordinal,key) +
 (SELECT MAX(n) FROM (
 SELECT COALESCE(CAST(value AS INTEGER),0) n FROM meta WHERE key='write_counter'
 UNION ALL SELECT COALESCE(MAX(commit_seq),0) FROM revision
 UNION ALL SELECT COALESCE(MAX(commit_seq),0) FROM note
 UNION ALL SELECT COALESCE(MAX(commit_seq),0) FROM excision
 UNION ALL SELECT COALESCE(MAX(commit_seq),0) FROM setting
 UNION ALL SELECT COALESCE(MAX(commit_seq),0) FROM delivery))
FROM (SELECT 'observation' kind,id key,rowid ordinal FROM observation
 UNION ALL SELECT 'delivery_event',json_array(delivery,what,at),0 FROM delivery_event);
UPDATE observation SET commit_seq=(SELECT seq FROM append_fact_order WHERE kind='observation' AND key=observation.id);
UPDATE delivery_event SET commit_seq=(SELECT seq FROM append_fact_order WHERE kind='delivery_event' AND key=json_array(delivery,what,at));
INSERT INTO meta(key,value) SELECT 'write_counter',CAST(MAX(seq) AS TEXT) FROM append_fact_order HAVING MAX(seq) IS NOT NULL
 ON CONFLICT(key) DO UPDATE SET value=excluded.value;
DROP TABLE append_fact_order;
CREATE INDEX observation_commit_seq ON observation(commit_seq);
CREATE INDEX delivery_event_commit_seq ON delivery_event(commit_seq);
