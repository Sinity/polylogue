//! The Morrow archive built through the real loop and import path holds
//! the sessions `FACTS.md` lists, with their kinds and own message counts.

use polylogue_model::SessionKind;
use polylogue_store::{Reader, SqlValue};
use polylogue_testkit::{Morrow, Set, session};

/// `FACTS.md`, "Sessions": own messages and human prompts per session.
/// #7 counts every node of its tree (6 on the active path, 2 on the
/// abandoned branch). Codex is stored one message per item, so #5 and #6
/// are checked by prompts only (`examples/morrow/README.md`).
/// Human prose prompts on the session's active path (a branched chat's
/// abandoned branch is off it).
fn prompts_on_active_path(r: &Reader, id: &str) -> usize {
    let rows = r
        .query(
            "SELECT count(*) AS n FROM message m JOIN session s ON s.pk = m.session_pk
             WHERE s.id = ?1 AND m.written_by = 'human' AND m.prompt_kind = 'prose'
               AND COALESCE(m.on_active_path, 1) = 1",
            &[SqlValue::Text(id.to_owned())],
        )
        .unwrap();
    rows[0]["n"].as_u64().unwrap() as usize
}

const FACTS: [(u8, Option<usize>, usize); 8] = [
    (1, Some(12), 3),
    (2, Some(18), 3),
    (3, Some(6), 0),
    (4, Some(10), 3),
    (5, None, 2),
    (6, None, 1),
    (7, Some(8), 3),
    (8, Some(4), 2),
];

#[test]
fn every_session_in_facts_is_built_with_its_own_messages() {
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let r = pool.reader().unwrap();
    let mut wrong = Vec::new();
    for (n, messages, prompts) in FACTS {
        let id = session(n);
        let Some(s) = r.session(&id).unwrap() else {
            wrong.push(format!("#{n} {id} missing"));
            continue;
        };
        let want_kind = if n == 3 {
            SessionKind::Subagent
        } else {
            SessionKind::Main
        };
        if s.kind != want_kind {
            wrong.push(format!("#{n} kind {:?}, want {want_kind:?}", s.kind));
        }
        let own = r.messages(&id, false).unwrap();
        if let Some(want) = messages
            && own.len() != want
        {
            wrong.push(format!("#{n} {} own messages, want {want}", own.len()));
        }
        let human = prompts_on_active_path(&r, &id.to_string());
        if human != prompts {
            wrong.push(format!("#{n} {human} human prompts, want {prompts}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_continuation_reads_its_inherited_prefix_logically() {
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let r = pool.reader().unwrap();
    // #4 resumes #2: 18 inherited messages, then its 10 own.
    assert_eq!(r.messages(&session(4), true).unwrap().len(), 28);
}

#[test]
#[ignore = "diagnostic"]
fn dump_links() {
    let w = Morrow::build(Set::Main);
    let pool = w.reads();
    let r = pool.reader().unwrap();
    for q in [
        "SELECT parent_id, child_id, kind, how FROM link",
        "SELECT id, repo FROM session",
        "SELECT tool, action, count(*) n FROM tool_call GROUP BY tool, action",
    ] {
        for row in r.query(q, &[]).unwrap() {
            eprintln!("{row}");
        }
    }
}
