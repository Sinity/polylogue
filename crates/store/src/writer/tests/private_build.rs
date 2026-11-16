//! The private index build at its durability boundaries
//! (docs/architecture.md, Durability; docs/operations.md, Replacing an index
//! from another schema).
use super::*;
use crate::archive::index_build::Paths;

fn one_message(input: &str, message: &str, text: &str) -> Batch {
    batch(
        input,
        Route::Live,
        0,
        vec![MessageRow {
            page_text: false,
            id_kind: polylogue_model::IdKind::Native,
            offset: 0,
            end: 100,
            message: text_message(message, input, Role::User, text, None),
        }],
        &[input],
    )
}

fn store_message(archive: &Archive, writer: &Writer, input: &str, text: &str) {
    let (blob, file) = archive.prepare_blob(&SOURCE_RECORD).unwrap();
    let mut reading = one_message(input, &format!("{input}-m"), text);
    reading.new_blob = Some(blob);
    let stored = writer
        .store_group(
            vec![crate::staging::PreparedBatch {
                batch: reading,
                files: file.into_iter().collect(),
            }],
            std::sync::Arc::new(archive.blobs().unwrap()),
        )
        .unwrap();
    for result in stored {
        result.unwrap();
    }
    let report = writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
}

fn serving(archive: &Archive) -> ReadPool {
    ReadPool::serving(archive, 1, 8 << 20).unwrap()
}

fn messages(pool: &ReadPool) -> u64 {
    pool.reader().unwrap().status().unwrap().messages
}

fn journal_mode(path: &std::path::Path) -> String {
    crate::archive::open_read(path)
        .unwrap()
        .query_row("PRAGMA main.journal_mode", [], |r| r.get(0))
        .unwrap()
}

#[test]
fn a_build_is_private_until_installed_and_reads_answer_its_progress() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let paths = Paths::of(dir.path());
    assert!(paths.build.exists());
    assert!(!paths.index.exists());
    let writer = Writer::start(&archive).unwrap();
    assert!(paths.unsynced.exists());
    store_message(&archive, &writer, "first-build", "private build text");
    let pool = serving(&archive);
    match pool.reader() {
        Err(StoreError::IndexBuilding(build)) => assert_eq!(
            *build,
            crate::archive::index_build::IndexBuild {
                serving_previous: false,
                chains: 1,
                chains_read: 1,
                bytes: 0,
                bytes_read: 0,
            }
        ),
        other => panic!("expected an index build, got {:?}", other.err()),
    }
    // The build pipeline reads its private rows before any install or sync.
    let pipeline = ReadPool::open(&archive).unwrap();
    assert_eq!(messages(&pipeline), 1);
    assert_eq!(journal_mode(&paths.build), "delete");
    writer.derive().unwrap();
    assert!(!paths.dir.exists());
    assert_eq!(journal_mode(&paths.index), "wal");
    assert_eq!(archive.index_build(), None);
    assert_eq!(messages(&pool), 1);
    assert_eq!(messages(&pipeline), 1);
    // The installed index is the writer's from now on.
    store_message(&archive, &writer, "after-install", "ordinary write");
    assert_eq!(messages(&pool), 2);
}

#[test]
fn a_clean_stop_during_a_build_keeps_its_progress_for_the_next_start() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::of(dir.path());
    {
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        store_message(&archive, &writer, "kept", "kept across a stop");
        drop(writer);
        assert!(!paths.unsynced.exists());
        assert!(paths.build.exists());
    }
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert!(archive.index_build().is_some());
    assert_eq!(messages(&ReadPool::open(&archive).unwrap()), 1);
    let writer = Writer::start(&archive).unwrap();
    writer.derive().unwrap();
    assert_eq!(messages(&serving(&archive)), 1);
}

#[test]
fn a_rebuild_keeps_serving_a_same_schema_index_until_it_is_installed() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::of(dir.path());
    {
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        store_message(&archive, &writer, "served", "served while rebuilt");
        writer.derive().unwrap();
    }
    // An extra index changes the schema but not its identity: this build's
    // code can still read the file.
    crate::archive::open_write(&paths.index, crate::archive::cache::SETUP)
        .unwrap()
        .execute_batch("CREATE INDEX synthetic_extra ON message(pk)")
        .unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert!(paths.previous.exists());
    assert_eq!(
        archive.index_build(),
        Some(crate::archive::index_build::IndexBuild {
            serving_previous: true,
            chains: 0,
            chains_read: 0,
            bytes: 0,
            bytes_read: 0,
        })
    );
    let pool = serving(&archive);
    assert_eq!(messages(&pool), 1);
    let writer = Writer::start(&archive).unwrap();
    assert_eq!(messages(&pool), 1);
    writer.derive().unwrap();
    // The installed index replaced the previous one; it holds what the
    // build wrote, which here is nothing parsed yet.
    assert!(!paths.dir.exists());
    assert_eq!(messages(&pool), 0);
}

#[test]
fn a_rebuild_of_an_index_in_another_format_answers_index_building() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::of(dir.path());
    {
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        Writer::start(&archive).unwrap().derive().unwrap();
    }
    crate::archive::open_write(&paths.index, crate::archive::cache::SETUP)
        .unwrap()
        .execute_batch("CREATE TABLE synthetic_older(value)")
        .unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert!(matches!(
        serving(&archive).reader().err(),
        Some(StoreError::IndexBuilding(build)) if !build.serving_previous
    ));
    Writer::start(&archive).unwrap().derive().unwrap();
    let tables: i64 = crate::archive::open_read(&paths.index)
        .unwrap()
        .query_row(
            "SELECT count(*) FROM sqlite_master WHERE name='synthetic_older'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(tables, 0);
    assert_eq!(messages(&serving(&archive)), 0);
}

/// Runs in a child process stopped at `point`, which the parent then kills
/// as a crash or power loss would end it.
#[cfg(feature = "faults")]
#[test]
fn private_build_crash_child() {
    let Some(root) = std::env::var_os("POLYLOGUE_PRIVATE_BUILD_ROOT") else {
        return;
    };
    let point = std::env::var("POLYLOGUE_PRIVATE_BUILD_POINT").unwrap();
    let root = std::path::Path::new(&root);
    if point == "index.after_creation" {
        crate::fault::arm(&point, "", crate::fault::Action::StopProcess, 0, 1);
    }
    let (archive, _lock) = Archive::open_or_create(root).unwrap();
    let writer = Writer::start(&archive).unwrap();
    store_message(&archive, &writer, "crashed", "written before the crash");
    match point.as_str() {
        "mid-build" => {
            crate::fault::arm(&point, "", crate::fault::Action::StopProcess, 0, 1);
            crate::fault::check(&point, "").unwrap();
        }
        _ => {
            crate::fault::arm(&point, "", crate::fault::Action::StopProcess, 0, 1);
            writer.derive().unwrap();
        }
    }
    panic!("the configured durability boundary was not reached");
}

#[cfg(feature = "faults")]
fn crash_at(point: &str) -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "writer::tests::private_build::private_build_crash_child",
            "--nocapture",
        ])
        .env("POLYLOGUE_PRIVATE_BUILD_ROOT", directory.path())
        .env("POLYLOGUE_PRIVATE_BUILD_POINT", point)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let pid = child.id() as libc::pid_t;
    let status = wait_stopped(pid);
    assert!(
        libc::WIFSTOPPED(status),
        "child exited before {point}: {status}"
    );
    child.kill().unwrap();
    assert!(!child.wait().unwrap().success());
    directory
}

#[cfg(feature = "faults")]
#[allow(unsafe_code)]
fn wait_stopped(pid: libc::pid_t) -> i32 {
    let mut status = 0;
    // This test owns pid; status is a live, uniquely borrowed output cell.
    loop {
        let result = unsafe { libc::waitpid(pid, &mut status, libc::WUNTRACED) };
        if result == pid {
            return status;
        }
        let error = std::io::Error::last_os_error();
        assert_eq!(
            error.kind(),
            std::io::ErrorKind::Interrupted,
            "waitpid: {error}"
        );
    }
}

#[cfg(feature = "faults")]
#[test]
fn a_crash_before_the_build_marker_resumes_the_synced_build_file() {
    let directory = crash_at("index.after_creation");
    let paths = Paths::of(directory.path());
    assert!(paths.build.exists());
    assert!(!paths.unsynced.exists());
    let generation = |path: &std::path::Path| -> String {
        crate::archive::open_read(path)
            .unwrap()
            .query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
                r.get(0)
            })
            .unwrap()
    };
    let before = generation(&paths.build);
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    // Same boot, same file: the build was resumed, not discarded.
    assert_eq!(generation(&paths.build), before);
    let writer = Writer::start(&archive).unwrap();
    store_message(&archive, &writer, "after", "after the crash");
    writer.derive().unwrap();
    assert_eq!(messages(&serving(&archive)), 1);
}

#[cfg(feature = "faults")]
#[test]
fn a_crash_during_a_build_discards_it_and_restarts_from_the_stored_bytes() {
    let directory = crash_at("mid-build");
    let paths = Paths::of(directory.path());
    assert!(paths.unsynced.exists());
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    assert!(!paths.unsynced.exists());
    let pipeline = ReadPool::open(&archive).unwrap();
    let reader = pipeline.reader().unwrap();
    // The stored bytes survive in the archive; the build starts over and
    // finds the chain unparsed.
    assert_eq!(reader.status().unwrap().messages, 0);
    assert_eq!(reader.status().unwrap().inputs, 1);
    let progress = reader.input_progress("crashed").unwrap().unwrap();
    assert!(!progress.parsed);
    assert_eq!(progress.length, 100);
}

#[cfg(feature = "faults")]
#[test]
fn a_crash_between_the_finish_sync_and_the_rename_installs_at_the_next_open() {
    let directory = crash_at("index.build.before_rename");
    let paths = Paths::of(directory.path());
    assert!(!paths.unsynced.exists());
    assert!(!paths.index.exists());
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    assert!(!paths.dir.exists());
    assert_eq!(archive.index_build(), None);
    assert_eq!(messages(&serving(&archive)), 1);
}

#[cfg(feature = "faults")]
#[test]
fn a_crash_after_the_finish_sync_before_the_marker_is_removed_rebuilds() {
    let directory = crash_at("index.build.after_finish_sync");
    let paths = Paths::of(directory.path());
    assert!(paths.unsynced.exists());
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    assert!(archive.index_build().is_some());
    assert_eq!(messages(&ReadPool::open(&archive).unwrap()), 0);
}

#[cfg(feature = "faults")]
#[test]
fn pipeline_readers_open_the_ready_build_and_refuse_it_after_install() {
    crate::fault::clear();
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let paths = Paths::of(dir.path());
    let writer = Writer::start(&archive).unwrap();
    store_message(
        &archive,
        &writer,
        "ready-build",
        "read from the installed build",
    );
    let pipeline = ReadPool::open(&archive).unwrap();
    crate::fault::arm(
        "index.build.before_rename",
        &archive.root.to_string_lossy(),
        crate::fault::Action::Hold,
        0,
        1,
    );
    let sender = writer.sender();
    let installer = std::thread::spawn(move || sender.derive());
    while !crate::fault::held() && !installer.is_finished() {
        std::thread::yield_now();
    }
    let held = crate::fault::held();
    let selected_path = archive.index_db();
    let checkout = pipeline.reader();
    crate::fault::release();
    let installed = installer.join();
    crate::fault::clear();

    assert!(held, "the install reached its rename boundary");
    installed.unwrap().unwrap();
    let reader = checkout.expect("a pipeline reader opens the ready build before install");
    assert_eq!(selected_path, paths.built);
    assert!(matches!(
        reader.status(),
        Err(StoreError::ReadIndexReplaced)
    ));
    drop(reader);
    assert_eq!(archive.index_db(), paths.index);
    assert_eq!(messages(&pipeline), 1);
}

#[cfg(feature = "faults")]
#[test]
fn a_started_ready_build_reader_keeps_its_snapshot_after_installed_writes() {
    crate::fault::clear();
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    store_message(&archive, &writer, "ready-first", "the original snapshot");
    let pipeline = ReadPool::open(&archive).unwrap();
    crate::fault::arm(
        "index.build.before_rename",
        &archive.root.to_string_lossy(),
        crate::fault::Action::Hold,
        0,
        1,
    );
    let sender = writer.sender();
    let installer = std::thread::spawn(move || sender.derive());
    while !crate::fault::held() && !installer.is_finished() {
        std::thread::yield_now();
    }
    let held = crate::fault::held();
    let checkout = pipeline.reader();
    let started = checkout.as_ref().ok().map(|reader| reader.status());
    crate::fault::release();
    let installed = installer.join();
    crate::fault::clear();

    assert!(held, "the install reached its rename boundary");
    installed.unwrap().unwrap();
    let reader = checkout.expect("open the ready build before install");
    assert_eq!(started.unwrap().unwrap().messages, 1);
    store_message(&archive, &writer, "ready-second", "written after install");
    drop(writer);
    assert_eq!(reader.status().unwrap().messages, 1);
    assert_eq!(messages(&pipeline), 2);
}
