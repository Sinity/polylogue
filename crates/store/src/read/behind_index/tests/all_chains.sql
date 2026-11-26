WITH revisions AS NOT MATERIALIZED (SELECT * FROM source_revision),
observations AS NOT MATERIALIZED (SELECT * FROM source_observation),
inputs AS MATERIALIZED (
                SELECT input, max(number) AS number, max(produced = 'imported') AS imported
                FROM revisions GROUP BY input
            ), chains AS (
                SELECT input, chain, max(length) AS length
                FROM revisions GROUP BY input, chain
            ), first_paths AS MATERIALIZED (
                SELECT input, min(id) AS observation
                FROM observations GROUP BY input
            ), formats AS (
                SELECT o.input, o.path_bytes, json_extract(o.details_json, '$.import_format') AS format,
                    row_number() OVER (PARTITION BY o.input
                        ORDER BY (json_extract(o.details_json, '$.import_format') = 'export-file'),
                                 o.commit_seq DESC, o.id DESC) AS choice
                FROM observations o WHERE o.class = 'imported' AND json_valid(o.details_json)
                    AND json_type(o.details_json, '$.import_format') = 'text'
            )
            SELECT c.input, c.chain, c.length, l.chain = c.chain,
                s.imported,
                COALESCE(f.format, i.origin),
                COALESCE(f.path_bytes, o.path_bytes),
                p.input_pk, p.chain, p.state, p.parser_version, p.parsed_to,
                p.waiting, p.used, p.skipped, p.unreadable, p.resume IS NOT NULL, p.failure_kind, p.failure_detail
            FROM chains c JOIN source_input i ON i.id = c.input
            JOIN inputs s ON s.input = c.input
            JOIN revisions l ON l.input = c.input AND l.number = s.number
            LEFT JOIN formats f ON f.input = c.input AND f.choice = 1
            LEFT JOIN first_paths fp ON fp.input = c.input
            LEFT JOIN observations o ON o.id = fp.observation
            LEFT JOIN input_ref ir ON ir.id = c.input
            LEFT JOIN parse_state p ON p.input_pk = ir.pk AND p.chain = c.chain
            WHERE c.length > 0
            ORDER BY 5,(SELECT max(o.seen_at) FROM observations o WHERE o.input=c.input) DESC,c.input,c.chain DESC;
