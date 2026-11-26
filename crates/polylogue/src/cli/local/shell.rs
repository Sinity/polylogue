//! Shell completion and picker registration, installed through the file owner.
use crate::installation::File;
use std::io;

pub(super) fn install(
    startup: &crate::config::Startup,
    arguments: &super::arguments::Install,
    json: bool,
    output: &mut dyn io::Write,
) -> io::Result<()> {
    let shell = arguments.target.as_str();
    let program = arguments.program.as_deref().unwrap_or("polylogue");
    if program.is_empty()
        || matches!(program, "." | "..")
        || program.contains('/')
        || program.chars().any(char::is_control)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--program needs one command name, without a directory or control characters",
        ));
    }
    let script = super::completion_script(shell, program).map_err(io::Error::other)?;
    if arguments.dry_run && !arguments.remove {
        return output.write_all(script.as_bytes());
    }
    let paths = &startup.config.paths;
    let path = match shell {
        "zsh" => paths
            .data
            .join("zsh/site-functions")
            .join(format!("_{program}")),
        "bash" => paths.data.join("bash-completion/completions").join(program),
        "fish" => paths
            .config
            .join("fish/completions")
            .join(format!("{program}.fish")),
        _ => unreachable!("only shell targets reach shell installation"),
    };
    let read = |path: &std::path::Path| {
        File::read(path).map_err(|error| io::Error::new(error.kind(), format!("{error}; print the managed script with `polylogue install {shell} --dry-run --program {program}`")))
    };
    let completion = read(&path)?;
    let startup_path = match shell {
        "zsh" => Some(startup.client.zsh_directory.join(".zshrc")),
        "fish" => Some(
            paths
                .config
                .join("fish/conf.d")
                .join(format!("{program}-polylogue.fish")),
        ),
        _ => None,
    };
    let startup_file = startup_path.as_ref().map(|path| read(path)).transpose()?;
    let marker = format!("# Polylogue {program}");
    let source = match shell {
        "zsh" => format!(
            "source {}",
            polylogue_render::shell_quote(&path.to_string_lossy())
        ),
        "fish" => format!("source {}", fish_quote(&path.to_string_lossy())),
        _ => String::new(),
    };
    // Validate every file before changing either of them.
    let startup_bytes = startup_file
        .as_ref()
        .map(|file| {
            edit_startup_file(
                file.current().unwrap_or_default(),
                &marker,
                &source,
                arguments.remove,
            )
        })
        .transpose()?;
    if arguments.remove
        && completion.current().is_some()
        && completion.current() != Some(script.as_bytes())
        && completion.backup().is_some()
        && completion.current() != completion.backup()
    {
        return Err(io::Error::other(format!(
            "{} changed after installation; its original bytes remain in {}",
            path.display(),
            crate::installation::backup_path(&path).display()
        )));
    }
    let mut files = vec![path.clone()];
    files.extend(startup_path.iter().cloned());
    if !arguments.dry_run {
        for path in &files {
            eprintln!(
                "{} {}",
                if arguments.remove {
                    "uninstalling"
                } else {
                    "installing"
                },
                path.display()
            );
        }
        if arguments.remove {
            if completion.current() == Some(script.as_bytes()) || completion.backup().is_some() {
                completion.restore()?;
            }
        } else {
            completion.write(script.as_bytes())?;
        }
        if let Some((file, bytes)) = startup_file.as_ref().zip(startup_bytes.as_ref()) {
            if arguments.remove
                && (file.current().is_none()
                    || file.backup().is_some_and(|original| {
                        bytes.as_slice() == original
                            // Install added a newline after a last line that lacked one.
                            || (!original.ends_with(b"\n")
                                && bytes.strip_suffix(b"\n") == Some(original))
                    })
                    || (bytes.is_empty() && file.backup().is_none()))
            {
                file.restore()?;
            } else if file.current() != Some(bytes.as_slice()) {
                file.write(bytes)?;
            }
        }
    }
    if json {
        writeln!(
            output,
            "{}",
            serde_json::json!({"removed":arguments.remove,"dry_run":arguments.dry_run,"files":files})
        )?;
    } else {
        for path in files {
            writeln!(
                output,
                "{} {}",
                match (arguments.remove, arguments.dry_run) {
                    (true, true) => "would uninstall",
                    (true, false) => "uninstalled",
                    (false, _) => "installed",
                },
                path.display()
            )?;
        }
    }
    Ok(())
}

fn edit_startup_file(
    current: &[u8],
    marker: &str,
    source: &str,
    remove: bool,
) -> io::Result<Vec<u8>> {
    let text = std::str::from_utf8(current)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let start = format!("{marker}\n");
    let end = format!("{marker} end\n");
    let starts: Vec<_> = text
        .match_indices(&start)
        .filter(|(at, _)| *at == 0 || text.as_bytes()[at - 1] == b'\n')
        .collect();
    let ends: Vec<_> = text
        .match_indices(&end)
        .filter(|(at, _)| *at == 0 || text.as_bytes()[at - 1] == b'\n')
        .collect();
    let block = format!("{start}{source}\n{end}");
    let result = match (starts.as_slice(), ends.as_slice()) {
        ([], []) if remove => return Ok(current.to_vec()),
        ([], []) => format!(
            "{text}{}{block}",
            if text.is_empty() || text.ends_with('\n') {
                ""
            } else {
                "\n"
            }
        ),
        ([(from, _)], [(until, _)]) if from < until => {
            let after = until + end.len();
            format!(
                "{}{}{}",
                &text[..*from],
                if remove { "" } else { &block },
                &text[after..]
            )
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "the Polylogue shell startup block has incomplete or repeated markers",
            ));
        }
    };
    Ok(result.into_bytes())
}

pub(super) fn picker_script(shell: &str, program: &str) -> String {
    let name: String = program
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect();
    match shell {
        "zsh" => format!(
            r#"
function _polylogue_pick_{name}() {{
    local references
    references="$({program} recent pick </dev/tty)" || return
    [[ -z "$references" ]] && return
    local reference
    for reference in "${{(@f)references}}"; do
        LBUFFER+="${{(q)reference}} "
    done
    zle redisplay
}}
zle -N _polylogue_pick_{name}
bindkey '^[p' _polylogue_pick_{name}
"#,
            program = polylogue_render::shell_quote(program)
        ),
        "fish" => format!(
            r#"
function _polylogue_pick_{name}
    set -l references ({program} recent pick </dev/tty)
    or return
    for reference in $references
        commandline --insert -- (string escape -- $reference)' '
    end
    commandline --function repaint
end
bind \ep _polylogue_pick_{name}
"#,
            program = fish_quote(program)
        ),
        _ => String::new(),
    }
}

fn fish_quote(text: &str) -> String {
    format!("'{}'", text.replace('\\', "\\\\").replace('\'', "\\'"))
}

#[cfg(test)]
mod tests;
