//! Shell command lines as structure: split into stages outside quotes, each
//! stage's program and words with assignments and wrapper programs skipped.
//! Nothing here reads meaning from a command's text; it only finds its shape.

/// Version of command normalization used while parsing.
pub const VERSION: u32 = 1;

/// One pipeline or list element: the program run and its arguments, with
/// leading `NAME=value` assignments and wrapper programs removed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    /// The program's base name (`/usr/bin/git` → `git`).
    pub program: String,
    /// The arguments after the program, unquoted.
    pub args: Vec<String>,
}

/// A command line split into stages, or a script: a loop, conditional,
/// subshell or heredoc, which has no single program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Line {
    Stages(Vec<Stage>),
    Script,
}

/// Programs whose first non-option argument is a subcommand.
const MULTI: &[&str] = &[
    "cargo",
    "git",
    "nix",
    "npm",
    "pnpm",
    "yarn",
    "uv",
    "go",
    "docker",
    "systemctl",
    "bd",
    "gh",
    "kubectl",
    "just",
];

/// Wrapper programs and the options of theirs that take a value.
const WRAPPERS: &[(&str, &[&str])] = &[
    (
        "env",
        &["-u", "--unset", "-C", "--chdir", "-S", "--split-string"],
    ),
    (
        "sudo",
        &[
            "-u", "--user", "-g", "--group", "-C", "-D", "-h", "-p", "-r", "-t",
        ],
    ),
    ("timeout", &["-k", "--kill-after", "-s", "--signal"]),
    ("nice", &["-n", "--adjustment"]),
    (
        "ionice",
        &["-c", "--class", "-n", "--classdata", "-p", "--pid"],
    ),
    ("nohup", &[]),
    ("time", &["-f", "--format", "-o", "--output"]),
    ("exec", &["-a"]),
];

/// Global options of multi-command programs that take a value, skipped
/// before the subcommand (`git -C dir status` → `git status`).
const GLOBAL_VALUE_OPTS: &[(&str, &[&str])] = &[
    (
        "git",
        &["-C", "-c", "--git-dir", "--work-tree", "--namespace"],
    ),
    ("cargo", &["-C", "--config", "-Z", "--color"]),
    (
        "nix",
        &["--option", "--extra-experimental-features", "--log-format"],
    ),
    ("bd", &["--actor", "--db", "--dir", "-C"]),
    ("gh", &["-R", "--repo"]),
    ("docker", &["-H", "--host", "--context", "-c", "--config"]),
    (
        "kubectl",
        &["-n", "--namespace", "--context", "--kubeconfig"],
    ),
    ("systemctl", &["-H", "--host", "-M", "--machine"]),
    ("npm", &["-C", "--prefix"]),
    ("pnpm", &["-C", "--dir", "--filter"]),
    ("yarn", &["--cwd"]),
    ("go", &["-C"]),
    ("uv", &["--directory", "--project"]),
    ("just", &["-f", "--justfile", "-d", "--working-directory"]),
];

const SCRIPT_WORDS: &[&str] = &[
    "for", "while", "until", "if", "case", "select", "function", "{", "[[",
];

/// Splits `line` into stages at `&&`, `||`, `;`, `|`, `&` and newlines,
/// outside quotes and command substitutions.
pub fn parse(line: &str) -> Line {
    let mut stages = Vec::new();
    let mut words: Vec<String> = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut chars = line.chars().peekable();
    let flush_word = |words: &mut Vec<String>, word: &mut String, in_word: &mut bool| {
        if *in_word {
            words.push(std::mem::take(word));
            *in_word = false;
        }
    };
    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                in_word = true;
                for q in chars.by_ref() {
                    if q == '\'' {
                        break;
                    }
                    word.push(q);
                }
            }
            '"' => {
                in_word = true;
                while let Some(q) = chars.next() {
                    match q {
                        '"' => break,
                        '\\' => {
                            if let Some(n) = chars.next() {
                                if !matches!(n, '"' | '\\' | '$' | '`') {
                                    word.push('\\');
                                }
                                word.push(n);
                            }
                        }
                        _ => word.push(q),
                    }
                }
            }
            '\\' => {
                in_word = true;
                match chars.next() {
                    Some('\n') | None => {}
                    Some(n) => word.push(n),
                }
            }
            '$' if chars.peek() == Some(&'(') => {
                // A command substitution stays inside its word.
                in_word = true;
                word.push('$');
                let mut depth = 0;
                for q in chars.by_ref() {
                    word.push(q);
                    match q {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                }
            }
            '`' => {
                in_word = true;
                word.push('`');
                for q in chars.by_ref() {
                    word.push(q);
                    if q == '`' {
                        break;
                    }
                }
            }
            '<' if chars.peek() == Some(&'<') => {
                // A heredoc (`<<`), not a here-string (`<<<`).
                chars.next();
                if chars.peek() != Some(&'<') {
                    return Line::Script;
                }
                chars.next();
                flush_word(&mut words, &mut word, &mut in_word);
            }
            '(' if !in_word && words.is_empty() => return Line::Script,
            '&' if in_word && (word.ends_with('>') || word.ends_with('<')) => {
                // `2>&1`, `<&3`: a redirection, not a separator.
                word.push('&');
            }
            '&' | '|' | ';' | '\n' => {
                flush_word(&mut words, &mut word, &mut in_word);
                if c == '&' && chars.peek() == Some(&'>') {
                    // `&>file` redirection, not a separator.
                    words.push("&>".into());
                    chars.next();
                    continue;
                }
                if (c == '&' || c == '|') && chars.peek() == Some(&c) {
                    chars.next();
                }
                if !words.is_empty() {
                    stages.push(std::mem::take(&mut words));
                }
            }
            c if c.is_whitespace() => flush_word(&mut words, &mut word, &mut in_word),
            _ => {
                in_word = true;
                word.push(c);
            }
        }
    }
    flush_word(&mut words, &mut word, &mut in_word);
    if !words.is_empty() {
        stages.push(words);
    }
    let mut out = Vec::new();
    for words in stages {
        match stage(words) {
            Some(Ok(s)) => out.push(s),
            Some(Err(Script)) => return Line::Script,
            None => {}
        }
    }
    Line::Stages(out)
}

struct Script;

/// One stage's program and arguments; `None` for a stage of assignments
/// or redirections only.
fn stage(words: Vec<String>) -> Option<Result<Stage, Script>> {
    let mut rest = words.into_iter().peekable();
    // Leading assignments and redirections.
    while let Some(w) = rest.peek() {
        if is_assignment(w) || w.starts_with('>') || w.starts_with('<') || w.starts_with("2>") {
            rest.next();
        } else {
            break;
        }
    }
    let words: Vec<String> = rest.collect();
    let mut rest = without_redirections(words).into_iter().peekable();
    loop {
        let first = rest.next()?;
        if SCRIPT_WORDS.contains(&first.as_str()) {
            return Some(Err(Script));
        }
        let base = base_name(&first);
        if let Some((_, value_opts)) = WRAPPERS.iter().find(|(w, _)| *w == base) {
            // Skip the wrapper's options, and `timeout`'s duration and
            // `env`'s assignments.
            while let Some(w) = rest.peek() {
                if w.starts_with('-') {
                    let takes_value = !w.contains('=') && value_opts.contains(&w.as_str());
                    rest.next();
                    if takes_value {
                        rest.next();
                    }
                } else if is_assignment(w) {
                    rest.next();
                } else {
                    break;
                }
            }
            if base == "timeout"
                && rest
                    .peek()
                    .is_some_and(|w| w.chars().next().is_some_and(|c| c.is_ascii_digit()))
            {
                rest.next();
            }
            continue;
        }
        return Some(Ok(Stage {
            program: base.to_owned(),
            args: rest.collect(),
        }));
    }
}

/// A word that redirects (`2>/dev/null`, `>out`, `2>&1`, `<in`), with
/// whether its target is the next word (`>` alone).
fn redirection(w: &str) -> Option<bool> {
    let body = w.trim_start_matches(|c: char| c.is_ascii_digit());
    let body = body.strip_prefix('&').unwrap_or(body);
    let op = ["<<<", ">>", "<>", ">&", "<&", ">", "<"]
        .into_iter()
        .find(|op| body.starts_with(op))?;
    Some(body.len() == op.len())
}

fn without_redirections(words: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    let mut skip_next = false;
    for w in words {
        if skip_next {
            skip_next = false;
            continue;
        }
        match redirection(&w) {
            Some(alone) => skip_next = alone,
            None => out.push(w),
        }
    }
    out
}

fn is_assignment(w: &str) -> bool {
    let Some((name, _)) = w.split_once('=') else {
        return false;
    };
    !name.is_empty()
        && name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn base_name(program: &str) -> &str {
    program.rsplit('/').next().unwrap_or(program)
}

impl Stage {
    /// The subcommand of a multi-command program, after its global
    /// options.
    pub fn subcommand(&self) -> Option<&str> {
        if !MULTI.contains(&self.program.as_str()) {
            return None;
        }
        let value_opts = GLOBAL_VALUE_OPTS
            .iter()
            .find(|(p, _)| *p == self.program)
            .map_or(&[][..], |(_, o)| *o);
        let mut args = self.args.iter();
        while let Some(a) = args.next() {
            if self.program == "cargo" && a.starts_with('+') {
                continue;
            }
            if a.starts_with('-') {
                if !a.contains('=') && value_opts.contains(&a.as_str()) {
                    args.next();
                }
                continue;
            }
            return Some(a);
        }
        None
    }

    /// The arguments after the subcommand (or after the program, for a
    /// single-command program).
    pub fn operands(&self) -> &[String] {
        match self.subcommand() {
            Some(sub) => {
                let at = self.args.iter().position(|a| a == sub).unwrap_or(0);
                &self.args[at + 1..]
            }
            None => &self.args,
        }
    }

    /// The stage's grouping key: program, subcommand, and the flag that
    /// makes a reading program write.
    pub fn normalized(&self) -> String {
        let mut key = self.program.clone();
        if let Some(sub) = self.subcommand() {
            key.push(' ');
            key.push_str(sub);
        }
        let writes = match self.program.as_str() {
            "sed" => self
                .args
                .iter()
                .find(|a| a.starts_with("-i") || a.starts_with("--in-place"))
                .map(|_| "-i"),
            "find" => self
                .args
                .iter()
                .find(|a| *a == "-delete" || a.starts_with("-exec"))
                .map(|a| if a == "-delete" { "-delete" } else { "-exec" }),
            _ => None,
        };
        if let Some(flag) = writes {
            key.push(' ');
            key.push_str(flag);
        }
        key
    }
}

impl Stage {
    /// The command a runner stage runs, when it is one: `nix develop
    /// --command …`, `nix shell … -c …`, `flock FILE …`, `bash -c '…'`.
    /// The normalized command keeps the runner; what a
    /// command does is read from the command it runs.
    pub fn runs(&self) -> Option<Line> {
        let after = |flags: &[&str]| {
            let at = self.args.iter().position(|a| flags.contains(&a.as_str()))?;
            Some(&self.args[at + 1..])
        };
        let words = match (self.program.as_str(), self.subcommand()) {
            ("nix", Some("develop" | "shell")) => after(&["--command", "-c"])?.to_vec(),
            ("flock", _) => {
                // `flock [options] FILE COMMAND…`: the first operand is the
                // lock file.
                let mut it = self.args.iter().skip_while(|a| a.starts_with('-'));
                it.next()?;
                it.cloned().collect()
            }
            ("bash" | "sh" | "zsh", _) => return Some(parse(after(&["-c"])?.first()?)),
            ("direnv", Some("exec")) => self.operands().get(1..)?.to_vec(),
            _ => return None,
        };
        if words.is_empty() {
            return None;
        }
        Some(parse(&shell_join(&words)))
    }
}

/// Words joined back into one command line, quoting what needs it.
fn shell_join(words: &[String]) -> String {
    words
        .iter()
        .map(|w| {
            if !w.is_empty()
                && w.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_./=:+,@%".contains(c))
            {
                w.clone()
            } else {
                format!("'{}'", w.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Every stage of a line, with the commands runner stages run in their
/// place, in order.
pub fn effective_stages(line: &str) -> Vec<Stage> {
    fn walk(line: Line, out: &mut Vec<Stage>, depth: u8) {
        let Line::Stages(stages) = line else {
            return;
        };
        for s in stages {
            match s.runs() {
                Some(inner) if depth < 4 => walk(inner, out, depth + 1),
                _ => out.push(s),
            }
        }
    }
    let mut out = Vec::new();
    let parsed = match parse(line) {
        // A heredoc's own command line still has a shape; its body is data.
        Line::Script if line.contains("<<") => {
            let head = line.lines().next().unwrap_or("");
            let head = head.split("<<").next().unwrap_or("");
            parse(head)
        }
        other => other,
    };
    walk(parsed, &mut out, 0);
    out
}

/// The call's normalized command: its last stage's, with `cd` and `pushd`
/// counted only when nothing follows them.
pub fn normalize(line: &str) -> Option<String> {
    match parse(line) {
        Line::Script => Some("shell script".into()),
        Line::Stages(stages) => {
            let last = stages
                .iter()
                .rev()
                .find(|s| !matches!(s.program.as_str(), "cd" | "pushd"))
                .or(stages.last())?;
            Some(last.normalized())
        }
    }
}

#[cfg(test)]
mod tests;
