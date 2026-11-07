//! `polylogue verify`: the store's check of every stored byte, run at idle
//! priority. It reads only; a manifest, when asked for, is written to a new
//! file and appears complete or not at all (not when a revision is unreadable).
use super::service::{Daemon, error, store_error};
use command_surface::Answer;
use serde_json::Value;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

pub(super) fn run(d: &Daemon, manifest: Option<&Path>) -> Answer<Value> {
    let _lease = match d.idle_admission.background() {
        Ok(lease) => lease,
        Err(e) => return store_error(&e),
    };
    let blobs = match d.archive.blobs() {
        Ok(blobs) => Arc::new(blobs),
        Err(e) => return store_error(&e),
    };
    let mut pace = || d.idle_admission.idle();
    let found = match manifest {
        None => polylogue_store::verify::verify(&d.reads, &blobs, None, &mut pace),
        Some(path) => {
            let mut file = match new_manifest(path) {
                Ok(file) => file,
                Err(message) => return error("bad_request", message),
            };
            let found = {
                let mut out = std::io::BufWriter::new(file.as_file_mut());
                polylogue_store::verify::verify(&d.reads, &blobs, Some(&mut out), &mut pace)
                    .and_then(|found| {
                        out.flush()?;
                        Ok(found)
                    })
            };
            found.and_then(|found| {
                // A manifest missing a revision is not written: the
                // temporary file is removed when it is dropped.
                let complete = !found.problems.iter().any(|problem| {
                    problem.kind == polylogue_store::verify::ProblemKind::RevisionUnreadable
                });
                if complete {
                    file.as_file().sync_all()?;
                    file.persist_noclobber(path).map_err(|e| e.error)?;
                }
                Ok(found)
            })
        }
    };
    match found {
        Ok(found) if found.problems.is_empty() => super::service::ok(found),
        Ok(found) => {
            let note = format!(
                "{} problem{} found; recovery is a restore from backup",
                found.problems.len(),
                if found.problems.len() == 1 { "" } else { "s" }
            );
            match serde_json::to_value(found) {
                Ok(value) => command_surface::Answer {
                    outcome: command_surface::Outcome::Degraded {
                        value,
                        missing: vec![note],
                    },
                    envelope: command_surface::Envelope {
                        next: vec![],
                        ..Default::default()
                    },
                },
                Err(cause) => error("internal", cause),
            }
        }
        Err(e) => store_error(&e),
    }
}

/// A temporary file beside `path`, which must name a new file in an
/// existing directory.
fn new_manifest(path: &Path) -> Result<tempfile::NamedTempFile, String> {
    if !path.is_absolute() {
        return Err("the manifest path must be absolute".into());
    }
    if path.symlink_metadata().is_ok() {
        return Err(format!("{} already exists", path.display()));
    }
    let directory = path
        .parent()
        .ok_or_else(|| format!("{} has no directory", path.display()))?;
    tempfile::Builder::new()
        .prefix(".polylogue-manifest-")
        .tempfile_in(directory)
        .map_err(|e| format!("cannot write in {}: {e}", directory.display()))
}

#[cfg(test)]
mod tests;
