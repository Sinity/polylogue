//! Refusals reach every surface: the command line, the socket and MCP
//! give the same answer for the same query text.

use super::*;
use crate::commands::{Action, Caller, Surface};
use crate::daemon::{Request, Response};
use polylogue_store::Archive;

/// The answer to `text` on one surface: the response data, or the error
/// kind and its text.
fn answer(archive: &Archive, surface: &str, text: &str) -> Result<Value, (String, String)> {
    let response = match surface {
        "cli" => {
            let (command, values) = crate::cli::parse(&["polylogue", "--json", text]).unwrap();
            if command.name != "query" {
                // The command line runs a command named by the whole
                // argument; it is never searched for.
                return Err(("command".into(), command.name.into()));
            }
            let Action::Request(build) = command.action else {
                panic!("{text}: the query runs locally");
            };
            let caller = Caller {
                surface: Surface::Cli,
                cwd: Some("/".into()),
                json: true,
            };
            let request = build(command, &values, &caller).unwrap();
            crate::cli::send(&archive.socket(), &request).unwrap()
        }
        "socket" => crate::cli::send(
            &archive.socket(),
            &Request::Query {
                text: text.to_owned(),
                cursor: None,
                page_size: None,
                why: false,
                session: None,
                fields: vec![],
            },
        )
        .unwrap(),
        _ => {
            let arguments = json!({ "query": text }).as_object().unwrap().clone();
            return match crate::mcp::call(archive, "query", arguments) {
                Ok(data) => Ok(serde_json::from_str(&data).unwrap()),
                Err(error) => {
                    let kind = error
                        .strip_prefix("error (")
                        .and_then(|e| e.split_once(')'))
                        .map_or_else(String::new, |(kind, _)| kind.to_owned());
                    Err((kind, error))
                }
            };
        }
    };
    match response {
        Response::Ok { data } => Ok(data),
        Response::Error {
            kind,
            message,
            detail,
        } => {
            let suggestion = detail
                .as_ref()
                .and_then(|d| d["suggestion"].as_str())
                .unwrap_or_default();
            Err((kind, format!("{message}\ntry: {suggestion}")))
        }
    }
}

/// A command word or a near miss of a unit or command at the start of a
/// query, and a value outside a field's closed set, are refused on every
/// surface with what to write instead, never answered with an empty
/// search; the same words quoted, a field comparison, a unit, short words
/// and correct values run as queries.
#[test]
fn every_surface_refuses_command_words_and_values_outside_a_set() {
    let w = crate::test_daemon::start("synthetic walrus prompt");
    let archive = Archive::at(&w.daemon.archive.root);
    for surface in ["cli", "socket", "mcp"] {
        for (text, kind, says) in [
            ("brief yesterday", "not_a_query", "\"brief\" yesterday"),
            ("status", "not_a_query", "\"status\""),
            ("messges cargo", "not_a_query", "messages cargo"),
            (
                "sessin where repo:morrow",
                "invalid_query",
                "session where repo:morrow",
            ),
            ("statsu", "not_a_query", "status"),
            ("sessions where outcome:error", "bad_value", "tool_calls"),
            ("tool_calls where outcome:solved", "bad_value", "sessions"),
            ("messages where role:asistant", "bad_value", "assistant"),
            ("kind:continuation", "bad_value", "links"),
            ("kind:mian", "bad_value", "subagent"),
        ] {
            match answer(&archive, surface, text) {
                Ok(v) => panic!("{surface}: {text} was answered: {v}"),
                Err((got, name)) if got == "command" => {
                    assert_eq!(surface, "cli");
                    assert_eq!(text, name, "{text} ran {name}");
                }
                Err((got, message)) => {
                    assert_eq!(got, kind, "{surface}: {text}: {message}");
                    assert!(
                        message.contains(says),
                        "{surface}: {text}: {message} does not say {says}"
                    );
                }
            }
        }
        for text in [
            "\"brief\" yesterday",
            "\"status\"",
            "sessions where messages > 0",
            "sessions | group by origin | count",
            "date parsing",
            "bug fix",
            "test output",
            "walrus",
            "tool_calls where outcome:unknown",
            "messages where role:assistant",
            "kind:subagent",
            "kind:*",
        ] {
            if let Err((kind, message)) = answer(&archive, surface, text) {
                panic!("{surface}: {text} refused: {kind} {message}");
            }
        }
    }
}
