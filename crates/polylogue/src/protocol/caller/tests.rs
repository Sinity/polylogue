use super::*;

fn environment(values: &[(&str, &str)]) -> BTreeMap<String, OsString> {
    values
        .iter()
        .map(|(key, value)| (key.to_string(), OsString::from(value)))
        .collect()
}

#[test]
fn nearest_harness_ancestor_selects_the_identity_that_actually_launched_the_client() {
    let env = environment(&[
        ("CLAUDE_CODE_SESSION_ID", "outer"),
        ("CODEX_THREAD_ID", "inner"),
        ("CODEX_TOOL_CALL_ID", "call-one"),
    ]);
    let ancestors = [
        (17, Origin::new("codex").unwrap()),
        (12, Origin::new("claude-code").unwrap()),
    ];
    let caller = select(&env, &ancestors, true, Some("shell.scope".into()));
    assert_eq!(caller.session.unwrap().to_string(), "codex:inner");
    assert_eq!(caller.harness_pid, Some(17));
    assert_eq!(caller.tool_call_id.as_deref(), Some("call-one"));
}

#[test]
fn known_harness_without_session_remains_an_agent_identity() {
    let caller = select(
        &BTreeMap::new(),
        &[(17, Origin::new("gemini-cli").unwrap())],
        true,
        None,
    );
    let authorship = polylogue_model::authorship::compute_authorship(
        polylogue_model::authorship::Channel::Cli,
        &caller,
        None,
    );
    assert_eq!(authorship.author.to_string(), "agent:gemini-cli");
}

#[test]
fn cgroup_and_parent_process_parsing_keep_native_identity() {
    assert_eq!(
        parent("17 (a command (with) parentheses) S 12 1 2 3"),
        Some(12)
    );
    assert_eq!(
        unit("0::/user.slice/user-1000.slice/user@1000.service/app.slice/task.scope\n").as_deref(),
        Some("task.scope")
    );
    assert_eq!(
        harness(
            Some(Path::new("/tools/node")),
            Some(b"node\0/tools/@anthropic-ai/claude-code/cli.js\0")
        )
        .unwrap()
        .as_str(),
        "claude-code"
    );
}

#[test]
fn a_query_word_named_mcp_does_not_change_the_observed_channel() {
    assert_eq!(channel(&["polylogue", "find", "mcp"]), Channel::Cli);
    assert_eq!(channel(&["polylogue", "--json", "mcp"]), Channel::Mcp);
    assert_eq!(
        channel(&["polylogue", "--deadline", "100", "mcp"]),
        Channel::Mcp
    );
}

#[test]
fn conflicting_environment_identities_without_a_harness_ancestor_stay_unattributed() {
    let env = environment(&[
        ("CLAUDE_CODE_SESSION_ID", "one"),
        ("CODEX_THREAD_ID", "two"),
    ]);
    let caller = select(&env, &[], true, Some("shell.scope".into()));
    assert!(caller.controlling_terminal);
    assert!(caller.ambiguous_harness);
    assert_eq!(
        polylogue_model::authorship::compute_authorship(Channel::Cli, &caller, None).author,
        polylogue_model::Author::Unattributed
    );
}
