//! `@N`: row N of the last list printed in this terminal (spec
//! `surfaces/cli.md`, Row numbers). Lists printed for a person number
//! their rows; the row IDs are kept per terminal in the archive's runtime
//! directory, and `@N` in a later command is replaced by the ID before the
//! command runs. Output for programs (`--json`) is never remembered.

use crate::web::access;
use std::fs;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

/// The terminal this process talks to (`pts-3`), from the first standard
/// stream that is one.
fn terminal() -> Option<String> {
    for fd in [2, 0, 1] {
        let Ok(target) = fs::read_link(format!("/proc/self/fd/{fd}")) else {
            continue;
        };
        let t = target.to_string_lossy();
        if let Some(rest) = t.strip_prefix("/dev/")
            && (rest.starts_with("pts/") || rest.starts_with("tty"))
        {
            return Some(rest.replace('/', "-"));
        }
    }
    None
}

fn rows_file(root: &Path) -> Option<PathBuf> {
    let tty = terminal()?;
    Some(access::runtime_dir(&access::archive_key(root)).join(format!("rows-{tty}")))
}

/// Keeps the IDs of a list just printed in this terminal. Failing to
/// write only means `@N` will not resolve, so errors are ignored.
pub fn remember(root: &Path, ids: &[String]) {
    let Some(path) = rows_file(root) else {
        return;
    };
    if let Some(dir) = path.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let tmp = path.with_extension("tmp");
    let written = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&tmp)
        .and_then(|mut f| f.write_all(ids.join("\n").as_bytes()));
    if written.is_ok() {
        let _ = fs::rename(&tmp, &path);
    }
}

/// Replaces each `@N` word in `text` by row N's ID. A word is `@` and
/// digits, standing alone, inside parentheses or as a field's value
/// (`parent:@1`); `@` inside a word
/// (`note:a@b`, `file:x@HEAD`) is left alone.
pub fn resolve(
    text: &str,
    rows: &dyn Fn() -> Result<Vec<String>, String>,
) -> Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut loaded: Option<Vec<String>> = None;
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let boundary = i == 0 || chars[i - 1].is_whitespace() || matches!(chars[i - 1], '(' | ':');
        if chars[i] == '@' && boundary {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            let digits: String = chars[i + 1..j].iter().collect();
            let mut k = j;
            if k < chars.len() && chars[k] == '.' {
                k += 1;
                while k < chars.len() && chars[k].is_ascii_digit() {
                    k += 1;
                }
            }
            let ends = k == chars.len() || chars[k].is_whitespace() || chars[k] == ')';
            if !digits.is_empty() && ends {
                if k > j {
                    return Err(format!(
                        "`{}` names a hit within a row, which is not kept yet; use `@{digits}`",
                        chars[i..k].iter().collect::<String>()
                    ));
                }
                let n: usize = digits
                    .parse()
                    .map_err(|_| format!("`@{digits}` is too large"))?;
                if loaded.is_none() {
                    loaded = Some(rows()?);
                }
                let ids = loaded.as_ref().expect("loaded above");
                let id = n.checked_sub(1).and_then(|i| ids.get(i)).ok_or_else(|| {
                    format!(
                        "`@{n}`: the last list printed in this terminal has {} rows",
                        ids.len()
                    )
                })?;
                out.push_str(id);
                i = j;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    Ok(out)
}

/// The IDs of the last list printed in this terminal.
pub fn last_rows(root: &Path) -> Result<Vec<String>, String> {
    let path = rows_file(root)
        .ok_or("`@N` refers to a list printed in a terminal, and this is not one")?;
    let text = fs::read_to_string(&path)
        .map_err(|_| "no list has been printed in this terminal yet".to_owned())?;
    Ok(text.lines().map(str::to_owned).collect())
}

/// The command line with `@N` replaced in every argument. `polylogue @2
/// read` is also accepted for `polylogue read @2`, and the same for the
/// other commands that take one session.
pub fn rewrite_args(args: Vec<String>, root: &Path) -> Result<Vec<String>, String> {
    let rows = || last_rows(root);
    let mut args = args;
    const VERBS: [&str; 4] = ["read", "export", "takeover", "continue"];
    if args.len() > 2 && is_position(&args[1]) && VERBS.contains(&args[2].as_str()) {
        args.swap(1, 2);
    }
    args.into_iter().map(|a| resolve(&a, &rows)).collect()
}

fn is_position(word: &str) -> bool {
    word.strip_prefix('@')
        .is_some_and(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
}

/// The ID each row of a printed answer is known by: its own `id`, else
/// its session's or message's.
pub fn row_ids(rows: &serde_json::Value) -> Vec<String> {
    rows.as_array()
        .into_iter()
        .flatten()
        .map(|r| {
            ["id", "session", "message"]
                .iter()
                .find_map(|k| r[*k].as_str())
                .unwrap_or("")
                .to_owned()
        })
        .collect()
}

#[cfg(test)]
mod tests;
