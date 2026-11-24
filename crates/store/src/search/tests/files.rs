//! Search over several full-text files ranks as search over one
//! (docs/search.md: Index maintenance).
use super::*;
use proptest::prelude::*;
use rusqlite::Connection;

const WORDS: &[&str] = &[
    "amber", "birch", "cedar", "delta", "ember", "fjord", "grove", "harbor", "Łódź", "straße",
    "ambient", "cedars",
];

/// An index in memory whose texts' entries are placed in the files `place`
/// names, with the entries of `removed` written after them and removed
/// again; nothing else is in it.
fn index(texts: &[String], removed: &[String], place: impl Fn(usize) -> usize) -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    let write = |n: usize, text: Option<&String>| {
        crate::text_rows::write_entry(
            &conn,
            &crate::search::files::schema(place(n)),
            "text_fts",
            n as i64 + 1,
            text.map(|text| polylogue_model::text::fold(text))
                .as_deref(),
        )
        .unwrap();
    };
    for (n, text) in texts.iter().chain(removed).enumerate() {
        write(n, Some(text));
    }
    for n in texts.len()..texts.len() + removed.len() {
        write(n, None);
    }
    conn
}

fn ranked(conn: &Connection, expression: &str, limit: Option<i64>) -> Vec<(i64, f64)> {
    let limit = limit.map(|limit| limit.to_string());
    conn.prepare(&format!(
        "SELECT text_id, rank FROM ({}) ORDER BY rank, text_id",
        ranked_texts(SnippetIndex::Prose, "?1", None, limit.as_deref())
    ))
    .unwrap()
    .query_map([expression], |row| Ok((row.get(0)?, row.get(1)?)))
    .unwrap()
    .collect::<rusqlite::Result<_>>()
    .unwrap()
}

/// FTS5's own ranking over one plain table of the same texts: the
/// independent oracle.
fn one_table(texts: &[String], expression: &str) -> Vec<(i64, f64)> {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE t USING fts5(body, content='', contentless_delete=1,
                                           tokenize='unicode61 remove_diacritics 2')",
    )
    .unwrap();
    for (n, text) in texts.iter().enumerate() {
        conn.execute(
            "INSERT INTO t(rowid, body) VALUES (?1, ?2)",
            rusqlite::params![n as i64 + 1, polylogue_model::text::fold(text)],
        )
        .unwrap();
    }
    conn.prepare("SELECT rowid, bm25(t) FROM t WHERE t MATCH ?1 ORDER BY bm25(t), rowid")
        .unwrap()
        .query_map([expression], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

fn texts() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(
        prop::collection::vec(prop::sample::select(WORDS), 0..12).prop_map(|words| words.join(" ")),
        1..40,
    )
}

/// An FTS5 expression as the query compiler writes one: phrases, prefixes,
/// and terms joined by AND, OR or NOT.
fn expression() -> impl Strategy<Value = String> {
    let term = (
        prop::collection::vec(prop::sample::select(WORDS), 1..3),
        any::<bool>(),
    )
        .prop_map(|(words, prefix)| {
            let mut text = words.join(" ");
            if prefix {
                text.truncate(text.char_indices().nth(3).map_or(text.len(), |(at, _)| at));
                text.push('*');
            }
            polylogue_model::text::fts_phrase(&text).expect("a word")
        });
    (
        prop::collection::vec(term, 1..4),
        prop::sample::select(&[" AND ", " OR ", " NOT "][..]),
    )
        .prop_map(|(terms, join)| terms.join(join))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    #[test]
    fn ranked_list_over_n_files_equals_one_file(
        texts in texts(),
        removed in prop::collection::vec(
            prop::collection::vec(prop::sample::select(WORDS), 0..12).prop_map(|words| words.join(" ")),
            0..8,
        ),
        expression in expression(),
        spread in prop::collection::vec(0..crate::search::files::FILES, 48),
        limit in 1i64..8,
    ) {
        let one = index(&texts, &removed, |_| 0);
        let by_key = index(&texts, &removed, |n| crate::search::files::file_of(n as i64 + 1));
        let spread = index(&texts, &removed, |n| spread[n]);
        let expected = ranked(&one, &expression, None);
        prop_assert_eq!(&ranked(&by_key, &expression, None), &expected);
        prop_assert_eq!(&ranked(&spread, &expression, None), &expected);
        let best = ranked(&one, &expression, Some(limit));
        prop_assert_eq!(&ranked(&spread, &expression, Some(limit)), &best);
        prop_assert_eq!(&best[..], &expected[..best.len()]);
        // Equal to FTS5's own ranking over one table of the texts that
        // remain, written once.
        let oracle = one_table(&texts, &expression);
        prop_assert_eq!(
            expected.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            oracle.iter().map(|(id, _)| *id).collect::<Vec<_>>()
        );
        for ((_, rank), (_, own)) in expected.iter().zip(&oracle) {
            prop_assert!((rank - own).abs() <= 1e-12 * own.abs().max(1.0), "{} vs {}", rank, own);
        }
    }
}
