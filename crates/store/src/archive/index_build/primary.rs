//! A deliberate primary rebuild keeps its WAL and every pipeline connection
//! on one physical file. Only the name serving readers changes at publication.
use super::*;
use crate::ConnExt as _;
use crate::primary::ReplayState;
use std::os::unix::fs::symlink;

#[derive(Clone)]
pub(super) struct Build {
    previous: Option<Previous>,
    observed: bool,
}

#[derive(Clone)]
struct Previous {
    scope: String,
    stream: String,
    position: i64,
}

fn stream(state: &ReplayState) -> Option<&str> {
    state
        .completed_delivery
        .as_ref()
        .map(|marker| marker.stream_instance.as_str())
        .or_else(|| {
            state
                .snapshot
                .as_ref()
                .map(|snapshot| snapshot.stream_instance.as_str())
        })
        .or_else(|| {
            state
                .staged_delivery
                .as_ref()
                .map(|marker| marker.stream_instance.as_str())
        })
        .or_else(|| {
            state
                .snapshot_progress
                .as_ref()
                .map(|snapshot| snapshot.stream_instance.as_str())
        })
}

fn previous(path: &Path) -> Result<Option<Previous>> {
    let index = crate::archive::open_read(path)?;
    if schema::index_identity(&index)? != schema::build_index_identity(true)? {
        return Ok(None);
    }
    match crate::primary::scope::require_readable(&index) {
        Ok(()) => {}
        Err(StoreError::ScopeChanged) => return Ok(None),
        Err(error) => return Err(error),
    }
    let state = crate::primary::confirmed::replay_state(&index)?;
    Ok(
        match (
            state.scope.as_deref(),
            stream(&state),
            state.applied_position,
        ) {
            (Some(scope), Some(stream), Some(position)) => Some(Previous {
                scope: scope.into(),
                stream: stream.into(),
                position,
            }),
            _ => None,
        },
    )
}

pub(super) fn published(paths: &Paths) -> bool {
    fs::read_link(&paths.index).is_ok_and(|target| target == Path::new(DIR).join("index.db"))
}

/// No archive handles have been handed out. Close and checkpoint a WAL before
/// moving its main file to a different directory.
fn checkpoint_closed(path: &Path) -> Result<()> {
    let main = File::open(path)?;
    let conn = crate::archive::open_runtime_writer(path, crate::archive::cache::SETUP)?;
    let busy: i64 = conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| row.get(0))?;
    if busy != 0 {
        return Err(StoreError::invalid(
            "the primary index still has an active WAL reader",
        ));
    }
    drop(conn);
    main.sync_all()?;
    match fs::metadata(with_suffix(path, "-wal")) {
        Ok(meta) if meta.len() > 0 => {
            return Err(StoreError::invalid(
                "the closed primary index still has WAL frames",
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

/// The one-shot request is consumed after the previous index is durably set aside.
pub(crate) fn prepare(archive: &Archive) -> Result<bool> {
    let paths = Paths::of(&archive.root);
    let request = archive.root.join("rebuild-index");
    if !request.is_file() && !paths.dir.exists() {
        return Ok(false);
    }

    if published(&paths) {
        checkpoint_closed(&paths.build)?;
        fs::rename(&paths.build, &paths.index)?;
        sync_directory(&archive.root)?;
        sync_directory(&paths.dir)?;
        crate::fault::check("primary.build.after_fold", &archive.root.to_string_lossy())?;
    }
    if paths.index.exists() && paths.previous.exists() && !paths.build.exists() {
        remove_build_files(&paths)?;
        archive.index_build.set(None, false);
    }
    if !paths.previous.exists() {
        if !request.is_file() {
            return Ok(false);
        }
        if paths.index.is_symlink() {
            return Err(StoreError::invalid(
                "the primary index points outside its rebuild",
            ));
        }
        if !paths.index.exists() {
            // An empty archive already starts at position zero.
            remove(&request)?;
            sync_directory(&archive.root)?;
            return Ok(false);
        }
        checkpoint_closed(&paths.index)?;
        set_aside(&paths)?;
        crate::fault::check(
            "primary.build.after_set_aside",
            &archive.root.to_string_lossy(),
        )?;
    }
    if request.exists() {
        remove(&request)?;
        sync_directory(&archive.root)?;
    }
    if paths.index.exists() {
        return Err(StoreError::invalid(
            "the primary rebuild has both a previous and an installed index",
        ));
    }
    let index = crate::archive::open_runtime_writer(&paths.build, crate::archive::cache::SETUP)?;
    archive.ensure_index(&index, false)?;
    let previous = match crate::primary::scope::require_readable(&index) {
        Ok(()) => previous(&paths.previous)?,
        Err(StoreError::ScopeChanged) => None,
        Err(error) => return Err(error),
    };
    crate::archive::generation::open(archive, &index)?;
    sync_directory(&paths.dir)?;
    let mut inner = archive.index_build.inner();
    inner.file = Some(BuildFile::Writing);
    inner.serving_previous = false;
    inner.primary = Some(Build {
        previous,
        observed: false,
    });
    Ok(true)
}

/// Withdraw old readers and detached results before a different grant, stream
/// or invalidated snapshot can resume. Also record the fence in the old
/// generation: after a crash a later grant cannot revive a previously denied cache.
pub(crate) fn withdraw(
    archive: &Archive,
    scope: Option<&str>,
    instance: Option<&str>,
    invalid: bool,
) -> Result<bool> {
    {
        let mut inner = archive.index_build.inner();
        let Some(build) = inner.primary.as_mut() else {
            return Ok(false);
        };
        build.observed = false;
        let Some(previous) = &build.previous else {
            return Ok(false);
        };
        if !invalid
            && scope == Some(previous.scope.as_str())
            && instance.is_none_or(|s| s == previous.stream)
        {
            return Ok(false);
        }
        inner.serving_previous = false;
    }
    let index = Paths::of(&archive.root).previous;
    let mut conn = match crate::archive::open_runtime_writer(&index, crate::archive::cache::SETUP) {
        Ok(conn) => conn,
        Err(error) => {
            archive.read_scope.begin_reset()?.invalidate()?;
            return Err(error);
        }
    };
    crate::primary::scope::deny(&mut conn, archive)?;
    if let Some(build) = archive.index_build.inner().primary.as_mut() {
        build.previous = None;
    }
    Ok(true)
}

/// Only a successful follower page can establish that the current read grant
/// and stream still cover the generation the daemon is serving.
pub(crate) fn observe(archive: &Archive, index: &Connection) -> Result<()> {
    if archive.index_build.inner().primary.is_none() {
        return Ok(());
    }
    let state = crate::primary::confirmed::replay_state(index)?;
    let (Some(scope), Some(instance)) = (state.scope.as_deref(), stream(&state)) else {
        return Ok(());
    };
    if withdraw(archive, Some(scope), Some(instance), false)? {
        // The new scope is already committed in the private index.
        crate::primary::scope::require_readable(index)?;
        archive.read_scope.finish_reset();
    }
    let mut inner = archive.index_build.inner();
    if let Some(build) = inner.primary.as_mut() {
        build.observed = true;
        inner.serving_previous = build.previous.is_some();
    }
    Ok(())
}

/// The normal flush checks a primary rebuild after parsing. Keep the one
/// follower on this inode through publication; a later exclusive startup
/// moves the closed file into place and retires the previous SQLite files.
pub(crate) fn finish(archive: &Archive, index: &Connection) -> Result<bool> {
    let previous_position = {
        let inner = archive.index_build.inner();
        let Some(build) = &inner.primary else {
            return Ok(false);
        };
        if !build.observed {
            return Ok(false);
        }
        build.previous.as_ref().map(|previous| previous.position)
    };
    crate::primary::scope::require_readable(index)?;
    let state = crate::primary::confirmed::replay_state(index)?;
    if state.snapshot_progress.is_some()
        || state.scope.is_none()
        || stream(&state).is_none()
        || !matches!((state.applied_position, state.latest_position), (Some(applied), Some(latest)) if applied >= latest && previous_position.is_none_or(|previous| applied >= previous))
        || !crate::derive::caught_up(index)?
        || index.exists(
            "SELECT EXISTS(SELECT 1 FROM primary_pending) OR EXISTS(SELECT 1 FROM primary_complete) OR EXISTS(SELECT 1 FROM search_pending)", []
        )?
        || !crate::archive::index_rebuild::built(index)?
    {
        return Ok(false);
    }
    // Publication changes a name. SQLite keeps both the database and its WAL
    // on their original paths, so readers need not release older WAL snapshots.
    crate::archive::generation::publish(index)?;
    let paths = Paths::of(&archive.root);
    sync_directory(&paths.dir)?;
    crate::fault::check("primary.build.after_sync", &archive.root.to_string_lossy())?;
    let link = paths.dir.join("install");
    remove(&link)?;
    symlink(Path::new(DIR).join("index.db"), &link)?;
    {
        let mut inner = archive.index_build.inner();
        fs::rename(&link, &paths.index)?;
        sync_directory(&archive.root)?;
        sync_directory(&paths.dir)?;
        inner.file = None;
        inner.primary = None;
        inner.serving_previous = false;
    }
    crate::fault::check(
        "primary.build.after_install",
        &archive.root.to_string_lossy(),
    )?;
    Ok(true)
}
