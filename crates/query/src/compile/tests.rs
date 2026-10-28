//! Compiler tests: the SQL a query produces, checked by its shape here and by running it in the daemon tests.

use super::*;

fn opts(saved: &[(&str, &str)]) -> Options {
    Options {
        clock: crate::Clock::new(
            1_790_000_000_000,
            polylogue_model::LogicalDay::named("Europe/Warsaw", 240).unwrap(),
        ),
        origins: vec!["claude-code".into(), "codex".into()],
        command_words: vec!["list".into(), "brief".into()],
        page_size: None,
        random_seed: None,
        embeddings: false,
        self_session: None,
        saved: saved
            .iter()
            .map(|(name, text)| {
                (
                    (*name).to_owned(),
                    vec![SavedQuery {
                        text: (*text).to_owned(),
                        revision: format!("{name}-revision"),
                        first_note: format!("{name}-first"),
                    }],
                )
            })
            .collect(),
        kept: Default::default(),
        fields: vec![],
        chat_pages: vec![ChatPage {
            origin: "chatgpt".into(),
            hosts: vec!["chatgpt.com".into(), "chat.openai.com".into()],
            path: "c".into(),
        }],
    }
}

#[test]
fn a_conversation_url_names_its_session_by_the_chat_pages() {
    for url in [
        "https://chatgpt.com/c/abc",
        "https://www.chat.openai.com/c/abc?model=x",
    ] {
        let p = c(url);
        assert!(
            p.params.contains(&Param::Text("chatgpt:abc".into())),
            "{url}: {:?}",
            p.params
        );
    }
    assert!(compile_err("https://claude.ai/chat/abc").contains("not a conversation URL"));
}

fn c(text: &str) -> Program {
    compile(text, &opts(&[])).unwrap_or_else(|e| panic!("{text}: {e}"))
}

fn compile_err(text: &str) -> String {
    compile(text, &opts(&[])).unwrap_err().to_string()
}

fn same(a: &str, b: &str, o: &Options) {
    let (pa, pb) = (compile(a, o).unwrap(), compile(b, o).unwrap());
    assert_eq!(pa.unit, pb.unit, "{a} / {b}");
    // The saved condition keeps its own parentheses; the meaning is equal.
    let flat = |s: &str| s.replace(['(', ')'], "");
    assert_eq!(flat(&pa.sql), flat(&pb.sql), "{a} / {b}");
    assert_eq!(pa.params, pb.params, "{a} / {b}");
}

/// A saved query selects the same rows used inline or by name.
#[test]
fn a_saved_query_selects_what_it_says_inline_or_by_name() {
    let o = opts(&[
        ("failing", "tool_calls where outcome:error"),
        ("morrow", "tag:project:morrow"),
        ("by-action", "tool_calls | group by action | count"),
    ]);
    // As the source, narrowed by the rest.
    same(
        "query:failing session.repo:morrow",
        "tool_calls where (outcome:error) session.repo:morrow",
        &o,
    );
    // As a condition over the same unit.
    same(
        "tool_calls where query:failing action:shell",
        "tool_calls where (outcome:error) action:shell",
        &o,
    );
    same("query:morrow", "sessions where (tag:project:morrow)", &o);
    // A shaped result is read, then further staged.
    same(
        "query:by-action | limit 2",
        "tool_calls | group by action | count | limit 2",
        &o,
    );
}

#[test]
fn a_saved_query_that_cannot_stand_there_is_refused() {
    let o = opts(&[
        ("failing", "tool_calls where outcome:error"),
        ("by-action", "tool_calls | group by action | count"),
        ("loop", "sessions where query:loop"),
    ]);
    for bad in [
        "tool_calls where query:by-action",
        "messages where query:failing",
        "query:by-action outcome:error",
        "query:nope",
        "query:loop",
    ] {
        assert!(compile(bad, &o).is_err(), "{bad}");
    }
}

#[test]
fn sort_takes_tie_breaking_keys() {
    let p = c("sessions | sort by live_rank, last_event_at desc");
    let directions: Vec<bool> = p.order.iter().map(|k| k.descending).collect();
    assert_eq!(directions, [false, false, false, true, false]);
    let term = |n: usize| {
        let end = p.sql.find(&format!(" AS _o{n},")).expect("order column");
        &p.sql[..end]
    };
    assert!(term(1).ends_with("END))"), "live_rank first: {}", p.sql);
    assert!(
        term(3).ends_with("live_last_event_at)"),
        "then last_event_at: {}",
        p.sql
    );
    assert!(compile_err("sessions | group by repo | sort by count, repo").contains("one key"));
}

#[test]
fn measure_ends_a_session_query() {
    let p = c("repo:polylogue | measure usage");
    let m = p.measure.expect("measure");
    assert_eq!(m.what, "usage");
    assert!(!m.everything);
    assert!(
        c("sessions | measure daily:tokens")
            .measure
            .unwrap()
            .everything
    );
    assert!(compile_err("messages | measure usage").contains("session query"));
    assert!(compile_err("sessions | measure usage | sort by cost").contains("last stage"));
}

#[test]
fn fields_choose_the_columns() {
    let p = compile(
        "sessions",
        &Options {
            fields: vec!["id".into(), "cost".into(), "repo".into()],
            ..opts(&[])
        },
    )
    .unwrap();
    let names: Vec<&str> = p.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["id", "cost", "repo"]);
}

#[test]
fn changes_continue_from_a_cursor() {
    let p = c("changes where after:generation:12:3");
    assert!(p.sql.contains("> 12") && p.sql.contains("> 3"), "{}", p.sql);
    assert!(compile_err("changes where after:12").contains("GEN:COUNTER:ARCHIVE"));
}

#[test]
fn deliveries_take_their_outcome_from_the_latest_event() {
    let p = c("deliveries where sender:signal | sort by at desc");
    assert!(p.sql.contains("delivery_event"), "{}", p.sql);
    assert!(!p.sql.contains(".outcome"), "{}", p.sql);
}

#[test]
fn a_query_starting_with_a_command_is_refused() {
    assert!(
        compile_err("list repo:x").contains("is a command"),
        "{}",
        compile_err("list repo:x")
    );
    let near = compile_err("brife now");
    assert!(near.contains("\"brief\""), "{near}");
    assert!(compile(r#""list" repo:x"#, &opts(&[])).is_ok());
}

/// Every value the model stores in a closed-set column can be queried, and
/// the field offers no value the model never stores.
#[test]
fn every_stored_value_can_be_queried() {
    use polylogue_model::{
        BlockKind, LINK_HOW_NAMES, LINK_KIND_NAMES, LinkKind, PromptKind, Role, SessionKind,
        WorkNodeKind,
    };
    let closed: &[(&str, &[&str])] = &[
        ("sessions where kind:", SessionKind::NAMES),
        ("messages where role:", Role::NAMES),
        ("messages where prompt_kind:", PromptKind::NAMES),
        ("blocks where kind:", BlockKind::NAMES),
        ("links where kind:", LINK_KIND_NAMES),
        ("links where how:", LINK_HOW_NAMES),
        ("work where kind:", WorkNodeKind::NAMES),
    ];
    for (query, names) in closed {
        for name in *names {
            c(&format!("{query}{name}"));
        }
        let (unit, field) = query.trim_end_matches(':').split_once(" where ").unwrap();
        let unit = crate::ast::Unit::from_name(unit).unwrap();
        let offered = crate::fields::lookup(unit, field).unwrap().values.unwrap();
        assert_eq!(offered, *names, "{query}");
    }
    for k in LinkKind::ALL {
        let (kind, _) = k.columns();
        c(&format!("sessions where link:{kind}"));
    }
    c("sessions where link:fork");
    c("sessions where link:resume");
    assert!(compile_err("sessions where kind:chat").contains("kind on sessions is one of"));
    assert!(compile_err("sessions where link:reference").contains("link is one of"));
}

#[test]
fn change_selections_are_output_arrays_not_scalar_operands() {
    for text in [
        "changes where sessions:x",
        "changes where messages > 0",
        "changes | sort by sessions",
        "changes | group by messages | count",
        "changes | agg sum:messages",
    ] {
        assert!(compile_err(text).contains("selection array"), "{text}");
    }
    let generations = c("changes where after:generation:1:2 | limit 1").change_generations;
    assert!(!generations.is_empty());
    assert!(
        generations
            .iter()
            .all(|generation| generation == "generation")
    );
}

mod documents;

#[test]
fn column_references_follow_fields_and_projection_owners() {
    let kinds = |program: Program| {
        program
            .columns
            .into_iter()
            .map(|c| (c.name, c.reference))
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    assert_eq!(kinds(c("sessions"))["id"], Some(ColumnReference::Session));
    assert_eq!(kinds(c("documents"))["id"], Some(ColumnReference::Document));
    assert_eq!(kinds(c("messages"))["id"], Some(ColumnReference::Message));
    assert_eq!(
        kinds(c("messages"))["session"],
        Some(ColumnReference::Session)
    );
    assert_eq!(kinds(c("notes"))["id"], Some(ColumnReference::Note));
    let mut options = opts(&[]);
    options.fields = vec!["id".into(), "session.id".into(), "session.title".into()];
    let projected = kinds(compile("messages", &options).unwrap());
    assert_eq!(projected["id"], Some(ColumnReference::Message));
    assert_eq!(projected["session_id"], Some(ColumnReference::Session));
    assert_eq!(projected["session_title"], None);
    for query in [
        "sessions | group by id",
        "documents | group by id",
        "sessions | agg min:id",
    ] {
        assert!(
            c(query).columns.iter().all(|c| c.reference.is_none()),
            "{query}"
        );
    }
}

#[test]
fn kept_rows_preserve_only_recorded_reference_metadata() {
    let mut options = opts(&[]);
    options.kept.insert(
        "docs".into(),
        Kept {
            unit: "documents".into(),
            columns: c("documents").columns,
            match_columns: vec![],
        },
    );
    assert_eq!(
        compile("result:docs | sort by title", &options)
            .unwrap()
            .columns[0]
            .reference,
        Some(ColumnReference::Document)
    );
    assert!(
        compile("result:docs | group by id", &options)
            .unwrap()
            .columns
            .iter()
            .all(|c| c.reference.is_none())
    );
    let fields = [("name", "id"), ("kind", "text")].into_iter();
    let without_reference: Column =
        serde::Deserialize::deserialize(serde::de::value::MapDeserializer::<
            _,
            serde::de::value::Error,
        >::new(fields))
        .unwrap();
    options.kept.insert(
        "plain".into(),
        Kept {
            unit: "sessions".into(),
            columns: vec![without_reference],
            match_columns: vec![],
        },
    );
    assert_eq!(
        compile("result:plain", &options).unwrap().columns[0].reference,
        None
    );
    options.kept.insert(
        "group".into(),
        Kept {
            unit: "sessions".into(),
            columns: c("sessions | group by id").columns,
            match_columns: vec![],
        },
    );
    assert!(
        compile("result:group", &options)
            .unwrap()
            .columns
            .iter()
            .all(|c| c.reference.is_none())
    );
}

#[test]
fn near_builds_only_when_an_answer_needs_ranking() {
    for embeddings in [false, true] {
        let mut options = opts(&[]);
        options.embeddings = embeddings;
        for text in [
            "near:dates | count",
            "near:dates | group by origin",
            "near:dates | agg sum:cost",
            "near:dates | measure activity",
            "near:dates | messages",
            "near:dates | sort by started | limit 1 | count",
        ] {
            let program = compile(text, &options).unwrap();
            assert!(program.near.is_none(), "{text}");
            assert!(program.incomplete.is_empty(), "{text}");
            assert!(!program.sql.contains("pl_similarity"), "{text}");
        }
        for text in [
            "near:dates",
            "near:dates | limit 1 | count",
            "near:dates | limit 1 | measure sessions",
            "near:dates | union (origin:codex)",
        ] {
            let program = compile(text, &options).unwrap();
            assert_eq!(program.near.is_some(), embeddings, "{text}");
            assert_eq!(program.incomplete.is_empty(), embeddings, "{text}");
        }
        for text in [
            "sessions | where near:dates",
            "sessions | messages where near:dates",
            "near:dates near:times",
            "sessions where NOT near:dates",
        ] {
            assert!(compile(text, &options).is_err(), "{text}");
        }
        if embeddings {
            assert!(compile("near:dates | union (near:times)", &options).is_err());
        } else {
            let counted = compile("near:dates | union (origin:codex) | count", &options).unwrap();
            assert!(counted.incomplete.is_empty());
        }
    }
    let text = "sessions | messages where near:dates";
    let QueryError::Invalid { position, .. } = compile(text, &opts(&[])).unwrap_err() else {
        panic!("expected invalid query")
    };
    assert_eq!(position, text.find("near:").unwrap());
}

#[test]
fn explicit_identity_equality_keeps_case_and_does_not_expand_a_prefix() {
    let exact = c("notes where first_note = \"Root:Case\" current:true");
    assert!(exact.params.contains(&Param::Text("Root:Case".into())));
    assert!(!exact.sql.contains("first_note LIKE"));
    let prefix = c("notes where first_note:Root");
    assert!(prefix.sql.contains("first_note LIKE"));
}

#[test]
fn kept_references_use_query_literals_without_splitting_their_ids() {
    let mut options = opts(&[]);
    let id = "source:λ | \"reader\".matches";
    let columns = c("sessions").columns;
    options.kept.insert(
        id.into(),
        Kept {
            unit: "sessions".into(),
            columns: columns.clone(),
            match_columns: vec![],
        },
    );
    let reference = r#"result:"source:λ | \"reader\".matches""#;
    let program = compile(&format!("{reference} | sort by id"), &options).unwrap();
    assert!(
        program
            .params
            .iter()
            .any(|p| matches!(p,Param::Text(value) if value==id))
    );
    assert!(program.sql.contains("$.rows"));
    assert!(!program.sql.contains("$.matches"));
    assert!(matches!(
        compile(&format!("{reference} title:other"), &options),
        Err(QueryError::Invalid { .. })
    ));
    options.kept.insert(
        "simple".into(),
        Kept {
            unit: "sessions".into(),
            columns: columns.clone(),
            match_columns: columns,
        },
    );
    assert!(
        compile("result:simple.matches | count", &options)
            .unwrap()
            .sql
            .contains("$.matches")
    );
}
