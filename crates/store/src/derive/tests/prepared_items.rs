use super::*;
use crate::derive::{Derived, Output, SessionWrite, Token, Write};
use rusqlite::{Connection, params};

fn duplicate_write(
    conn: &Connection,
    session: i64,
    message: i64,
    duplicate: Option<&str>,
) -> Write {
    let token = Token::read(conn, session).unwrap().unwrap().0;
    let items = if duplicate.is_none() {
        vec![(
            message,
            crate::items::prepare_retained_message(conn, message).unwrap(),
        )]
    } else {
        Vec::new()
    };
    Write {
        sessions: vec![SessionWrite {
            pk: session,
            token,
            items,
            outputs: vec![Derived {
                name: "duplicates",
                version: "synthetic".into(),
                output: Output::Duplicates(vec![(message, duplicate.map(str::to_owned))]),
            }],
        }],
        finish: false,
    }
}

fn hidden() -> (WriterArchive, Connection, i64, i64) {
    let mut world = WriterArchive::new();
    world.write(
        "session",
        prose("session", 1, 1000, "retained original content"),
    );
    let conn = crate::archive::open_write(&world.archive.index_db(), crate::archive::cache::SETUP)
        .unwrap();
    let (message, session) = conn
        .query_row("SELECT pk,session_pk FROM message", [], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
        })
        .unwrap();
    conn.execute(
        "UPDATE session SET changed=changed+1 WHERE pk=?1",
        [session],
    )
    .unwrap();
    let hide = duplicate_write(&conn, session, message, Some("synthetic-repeat"));
    assert_eq!(crate::derive::apply(&conn, &hide).unwrap(), vec![true]);
    assert_eq!(
        conn.query_row("SELECT count(*) FROM session_item", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    (world, conn, session, message)
}

#[test]
fn revealed_message_items_are_prepared_before_the_derive_commit() {
    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
    let (_world, conn, session, message) = hidden();
    let reveal = duplicate_write(&conn, session, message, None);
    conn.flush_prepared_statement_cache();
    conn.authorizer(Some(|context: AuthContext<'_>| match context.action {
        AuthAction::Read {
            table_name: "block" | "text" | "message_version_body",
            column_name: "body",
        } => Authorization::Deny,
        _ => Authorization::Allow,
    }))
    .unwrap();
    assert_eq!(crate::derive::apply(&conn, &reveal).unwrap(), vec![true]);
    conn.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    let expected = crate::items::computed_entries(&conn, session).unwrap();
    let actual = conn
        .prepare("SELECT entry FROM session_item ORDER BY seq")
        .unwrap()
        .query_map([], |row| row.get::<_, Vec<u8>>(0))
        .unwrap()
        .map(|row| {
            serde_json::from_slice::<polylogue_model::read::IndexEntry>(&row.unwrap()).unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert_eq!(actual[1].body_bytes, 25);
}

#[test]
fn stale_reveal_cannot_restore_an_old_version_or_retracted_message() {
    for retract in [false, true] {
        let (mut world, conn, session, message) = hidden();
        let reveal = duplicate_write(&conn, session, message, None);
        world.write_snapshot(
            "session",
            if retract {
                Vec::new()
            } else {
                prose("session", 1, 2000, "newer selected content")
            },
        );
        let before: Vec<Vec<u8>> = conn
            .prepare("SELECT entry FROM session_item ORDER BY seq")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(crate::derive::apply(&conn, &reveal).unwrap(), vec![false]);
        let after: Vec<Vec<u8>> = conn
            .prepare("SELECT entry FROM session_item ORDER BY seq")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(after, before);
        if retract {
            assert_eq!(
                conn.query_row(
                    "SELECT count(*) FROM message WHERE pk=?1",
                    [message],
                    |row| row.get::<_, i64>(0)
                )
                .unwrap(),
                0
            );
        } else {
            let selected = crate::items::computed_entries(&conn, session).unwrap();
            assert_eq!(selected[1].body_bytes, 22);
        }
        let hidden: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM message WHERE pk=?1 AND duplicate_of IS NOT NULL)",
                params![message],
                |row| row.get(0),
            )
            .unwrap();
        assert!(!hidden);
    }
}
