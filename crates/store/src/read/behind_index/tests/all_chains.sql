WITH revisions AS NOT MATERIALIZED (SELECT * FROM source_revision),
observations AS NOT MATERIALIZED (SELECT rowid,* FROM source_observation),
inputs AS MATERIALIZED (
                SELECT input, max(number) AS number, max(produced = 'imported') AS imported
                FROM revisions GROUP BY input
            ), chains AS (
                SELECT input, chain, max(length) AS length
                FROM revisions GROUP BY input, chain
            ), first_paths AS MATERIALIZED (
                SELECT input, min(rowid) AS observation
                FROM observations GROUP BY input
            ), formats AS (
                SELECT o.input, o.path, json_extract(o.details_json, '$.import_format') AS format,
                    row_number() OVER (PARTITION BY o.input
                        ORDER BY (json_extract(o.details_json, '$.import_format') = 'export-file'),
                                 o.rowid DESC) AS choice
                FROM observations o WHERE o.class = 'imported' AND json_valid(o.details_json)
                    AND json_type(o.details_json, '$.import_format') = 'text'
            )
            SELECT c.input, c.chain, c.length, l.chain = c.chain,
                s.imported,
                COALESCE(f.format, i.format),
                COALESCE(f.path, o.path),
                COALESCE(p.parsed_upto, 0), p.input_pk IS NOT NULL, COALESCE(p.status = 'waiting' AND p.at_length = c.length, 0), p.reading_version, p.status, p.at_length, p.reason, p.stopped_attempt
            FROM chains c JOIN source_input i ON i.id = c.input
            JOIN inputs s ON s.input = c.input
            JOIN revisions l ON l.input = c.input AND l.number = s.number
            LEFT JOIN formats f ON f.input = c.input AND f.choice = 1
            LEFT JOIN first_paths fp ON fp.input = c.input
            LEFT JOIN observations o ON o.rowid = fp.observation
            LEFT JOIN input_ref ir ON ir.id = c.input
            LEFT JOIN parse_state p ON p.input_pk = ir.pk AND p.chain = c.chain
            WHERE (c.length > 0
              AND (p.input_pk IS NULL
                OR (NOT COALESCE(p.at_length = c.length AND (p.status = 'waiting' OR NOT ?2), 0)
                    AND p.parsed_upto < c.length)
                OR (?2 AND p.status IN ('refused','failed') AND p.at_length=c.length)
                OR EXISTS(SELECT 1 FROM json_each(?1) expected
                    WHERE json_extract(expected.value, '$[0]') = COALESCE(f.format, i.format)
                      AND json_extract(expected.value, '$[1]') IS NOT p.reading_version)))
            ORDER BY 5,(SELECT max(o.seen_at) FROM observations o WHERE o.input=c.input) DESC,c.input,c.chain DESC;
