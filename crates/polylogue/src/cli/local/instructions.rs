//! The MCP card in each supported harness's global instructions file.
use crate::{config::Startup, hook::Harness, installation::File};
use std::{io, path::PathBuf};

const OPEN: &str = "<!-- polylogue -->";
const CLOSE: &str = "<!-- /polylogue -->";

pub(super) fn block() -> String {
    let card = crate::manual::card();
    format!(
        "{OPEN}\n{card}{}{CLOSE}\n",
        if card.ends_with('\n') { "" } else { "\n" }
    )
}

pub(super) fn install(
    startup: &Startup,
    arguments: &super::arguments::Install,
    json: bool,
    output: &mut dyn io::Write,
) -> io::Result<()> {
    let harness = Harness::parse(&arguments.target).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "--instructions supports claude-code, codex and gemini-cli",
        )
    })?;
    let filename = match harness {
        Harness::ClaudeCode => "CLAUDE.md",
        Harness::Codex => "AGENTS.md",
        Harness::GeminiCli => "GEMINI.md",
        Harness::Hermes => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Hermes reads its hooks from config.yaml; --instructions has no supported global instructions file for Hermes",
            ));
        }
    };
    // The hook owner already resolves the harness's configured home. A settings
    // override changes only the settings file, never its global instructions.
    let default_settings = crate::hook::settings_path(harness, &startup.engine)
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
    let path = default_settings.with_file_name(filename);
    if arguments.dry_run && !arguments.remove {
        return output.write_all(block().as_bytes());
    }
    let settings = arguments
        .settings
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or(default_settings);
    // A settings override must not make the two edits (or their backups)
    // write into the same file.
    for settings_target in [&settings, &crate::installation::backup_path(&settings)] {
        for instructions_target in [&path, &crate::installation::backup_path(&path)] {
            let same = std::path::absolute(settings_target)?
                == std::path::absolute(instructions_target)?
                || settings_target
                    .canonicalize()
                    .ok()
                    .zip(instructions_target.canonicalize().ok())
                    .is_some_and(|(left, right)| left == right);
            if same {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "hook settings and global instructions need distinct files and backups",
                ));
            }
        }
    }
    // Refuse malformed blocks or managed instruction/backup files before the
    // existing hook installer can change the settings file.
    let file = File::read(&path).map_err(|error| io::Error::new(error.kind(),format!(
        "{error}; print the managed block with `polylogue install {} --instructions --dry-run`", arguments.target,
    )))?;
    let bytes = edit(file.current().unwrap_or_default(), arguments.remove)?;
    let restore = arguments.remove
        && match file.backup() {
            Some(original) => {
                bytes.as_slice() == original
                    || (!original.ends_with(b"\n") && bytes.strip_suffix(b"\n") == Some(original))
            }
            None => {
                bytes.is_empty()
                    && file
                        .current()
                        .is_some_and(|current| current != bytes.as_slice())
            }
        };
    let files = [&settings, &path];
    if !arguments.dry_run {
        for target in files {
            eprintln!(
                "{} {}",
                if arguments.remove {
                    "removing Polylogue from"
                } else {
                    "installing Polylogue in"
                },
                target.display()
            );
        }
        if !arguments.remove {
            eprint!("{}", block());
        }
        crate::hook::install(
            &settings,
            harness,
            || crate::hook::probe_version(harness),
            !arguments.remove,
        )?;
        if restore {
            file.restore()?;
        } else if file.current().unwrap_or_default() != bytes.as_slice() {
            file.write(&bytes)?;
        }
    }
    if json {
        writeln!(
            output,
            "{}",
            serde_json::json!({"removed":arguments.remove,"dry_run":arguments.dry_run,"files":files})
        )
    } else {
        for target in files {
            writeln!(
                output,
                "{} {}",
                match (arguments.remove, arguments.dry_run) {
                    (true, true) => "would remove Polylogue from",
                    (true, false) => "removed Polylogue from",
                    (false, _) => "installed Polylogue in",
                },
                target.display()
            )?;
        }
        Ok(())
    }
}

fn edit(current: &[u8], remove: bool) -> io::Result<Vec<u8>> {
    let text = std::str::from_utf8(current)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let (mut start, mut end) = (None, None);
    let mut at = 0;
    for line in text.split_inclusive('\n') {
        let marker = line.strip_suffix('\n').unwrap_or(line);
        let marker = marker.strip_suffix('\r').unwrap_or(marker);
        let repeated = match marker {
            OPEN => start.replace(at).is_some(),
            CLOSE => end.replace(at + line.len()).is_some(),
            _ => false,
        };
        if repeated {
            return Err(markers());
        }
        at += line.len();
    }
    match (start, end) {
        (None, None) if remove => Ok(current.to_vec()),
        (None, None) => Ok(format!(
            "{text}{}{}",
            if text.is_empty() || text.ends_with('\n') {
                ""
            } else {
                "\n"
            },
            block()
        )
        .into_bytes()),
        (Some(from), Some(until)) if from < until => Ok(format!(
            "{}{}{}",
            &text[..from],
            if remove { String::new() } else { block() },
            &text[until..]
        )
        .into_bytes()),
        _ => Err(markers()),
    }
}
fn markers() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "the Polylogue instructions block has incomplete, reversed or repeated markers",
    )
}
