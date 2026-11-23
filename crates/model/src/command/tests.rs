use super::*;

fn n(line: &str) -> String {
    normalize(line).unwrap()
}

#[test]
fn prefixes_do_not_change_the_key() {
    for line in [
        "cargo test -p store",
        "cd /src/tide && cargo test -p store",
        "RUST_LOG=debug cargo test -p store",
        "cd /x && FOO=1 BAR=2 cargo test",
        "env RUST_LOG=1 cargo test",
        "timeout 600 nice -n 19 ionice -c3 cargo test",
        "sudo -u build /usr/bin/cargo +nightly test",
    ] {
        assert_eq!(n(line), "cargo test", "{line}");
    }
    assert_eq!(n("git add -A && git commit -m x"), "git commit");
    assert_eq!(n("cargo build 2>&1 | tail -5"), "tail");
    let Line::Stages(s) = parse("cargo test 2>&1 >/dev/null") else {
        panic!()
    };
    assert_eq!(s.len(), 1, "a redirection is not a stage separator");
    assert_eq!(n("git -C /repo status"), "git status");
    assert_eq!(n("bd --actor me update proj-1 --claim"), "bd update");
}

#[test]
fn quotes_keep_separators_inside_one_word() {
    assert_eq!(n("jq '.a | .b' file.json"), "jq");
    assert_eq!(n("python3 -c \"import x; print(1)\""), "python3");
    assert_eq!(n("echo $(date; uptime)"), "echo");
    assert_eq!(
        effective_stages("echo $(date; uptime) && ls")
            .iter()
            .map(Stage::normalized)
            .collect::<Vec<_>>(),
        ["echo", "ls"]
    );
}

#[test]
fn writing_flags_make_a_different_command() {
    assert_eq!(n("sed -n 1,5p f"), "sed");
    assert_eq!(n("sed -i s/a/b/ f"), "sed -i");
    assert_eq!(n("sed -Ei s/a/b/ f"), "sed -i");
    assert_eq!(n("sed -ni s/a/b/p f"), "sed -i");
    assert_eq!(n("sed -i.bak s/a/b/ f"), "sed -i");
    assert_eq!(n("sed -E s/a/b/ f"), "sed");
    assert_eq!(n("find . -name x -delete"), "find -delete");
    assert_eq!(n("find . -exec rm {} ;"), "find -exec");
}

#[test]
fn scripts_and_lone_cd() {
    assert_eq!(n("for f in *; do echo $f; done"), "shell script");
    assert_eq!(n("if true; then ls; fi"), "shell script");
    assert_eq!(n("cat <<EOF > x\nhi\nEOF"), "shell script");
    assert_eq!(n("cat <<< 'here string'"), "cat");
    assert_eq!(n("cd /tmp"), "cd");
    assert_eq!(n("cd /a && cd /b"), "cd");
}

#[test]
fn inline_functions_keep_body_commands() {
    for (line, expected) in [
        ("R=x; g(){ cd a; git -C b status; }; g", "git status"),
        ("R=x; g() { cd a; git -C b status; }; g", "git status"),
        ("R=x; g(){ cd a; git -C b status; }; $G", "git status"),
        ("g(){ git -C b status; }; g", "git status"),
        ("g(){ git status; }; h(){ cargo test; }; g; h", "cargo test"),
    ] {
        assert_eq!(n(line), expected, "{line}");
    }
    let Line::Stages(stages) = parse("R=x; g(){ cd a; git -C b status; }; g") else {
        panic!()
    };
    assert_eq!(
        stages.iter().map(Stage::normalized).collect::<Vec<_>>(),
        ["cd", "git status"]
    );
}

#[test]
fn runners_expose_the_command_they_run() {
    let progs = |line: &str| -> Vec<String> {
        effective_stages(line)
            .iter()
            .map(Stage::normalized)
            .collect()
    };
    assert_eq!(
        progs("cd /r && nix develop --command cargo nextest run -p x"),
        ["cd", "cargo nextest"]
    );
    assert_eq!(
        progs("timeout 900 nix develop .#ci -c bash -c 'cargo test && bd close a-1'"),
        ["cargo test", "bd close"]
    );
    assert_eq!(progs("flock /tmp/l.lock cargo build"), ["cargo build"]);
    // The normalized command keeps the runner.
    assert_eq!(n("nix develop --command cargo test"), "nix develop");
}

#[test]
fn redirections_are_not_arguments() {
    assert_eq!(n("bd 2>/dev/null update x-1 --claim"), "bd update");
    assert_eq!(n("git > out.txt log"), "git log");
    assert_eq!(n("2> /dev/null bd update x-1 --claim"), "bd update");
    assert_eq!(n("FOO=1 2> /dev/null git log"), "git log");
    let heads: Vec<String> = effective_stages("cd /r && git commit -q -F - <<'EOF'\nmsg\nEOF")
        .iter()
        .map(Stage::normalized)
        .collect();
    assert_eq!(heads, ["cd", "git commit"]);
}

#[test]
fn stages_expose_operands() {
    let Line::Stages(s) = parse("bd --actor x update proj-abc.7 --status in_progress") else {
        panic!()
    };
    assert_eq!(s[0].subcommand(), Some("update"));
    assert_eq!(s[0].operands(), ["proj-abc.7", "--status", "in_progress"]);
}

#[test]
fn an_option_value_spelled_like_the_subcommand_is_not_the_subcommand() {
    let Line::Stages(s) = parse("git -c log log x") else {
        panic!()
    };
    assert_eq!(s[0].subcommand(), Some("log"));
    assert_eq!(s[0].operands(), ["x"]);
}
