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
            response: None,
            page_text: false,
            id_kind: polylogue_model::IdKind::Native,
            offset: 0,
            end: 100,
            message: text_message(message, input, Role::User, text, None),
        }],
        &[input],
    )
}

fn store_message(archive: &Archive, writer: &Writer, input: &str, text: &str) -> i64 {
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
    let chain = stored.into_iter().next().unwrap().unwrap().chain;
    let report = writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    chain
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

/// An archive at `root` holding one stored message per input, opened again
/// without its index, so a build is due. Returns each input's chain.
pub(in crate::writer) fn due_a_build(root: &std::path::Path, inputs: &[&str]) -> Vec<i64> {
    let chains = {
        let (archive, _lock) = Archive::open_or_create(root).unwrap();
        assert!(archive.index_build().is_none());
        let writer = Writer::start(&archive).unwrap();
        inputs
            .iter()
            .map(|input| store_message(&archive, &writer, input, &format!("{input} text")))
            .collect()
    };
    let paths = Paths::of(root);
    for suffix in ["", "-wal", "-shm"] {
        let mut name = paths.index.clone().into_os_string();
        name.push(suffix);
        let _ = std::fs::remove_file(name);
    }
    chains
}

/// One run holding each input's chain read from the record.
pub(in crate::writer) fn runs(
    archive: &Archive,
    writer: &Writer,
    inputs: &[&str],
    chains: &[i64],
) -> Vec<std::path::PathBuf> {
    let mut run = crate::writer::build::BuildWorker::create(archive, 0, 32 << 20).unwrap();
    for (input, &chain) in inputs.iter().zip(chains) {
        let mut batch = one_message(input, &format!("{input}-m"), &format!("{input} text"));
        let sender = writer.sender();
        let mut open = run
            .begin(&sender, &batch.input, chain, batch.length)
            .unwrap();
        for step in &mut batch.parse {
            crate::writer::ChainSink::step(&mut open, step).unwrap();
        }
        open.finish().unwrap();
    }
    run.finish().unwrap().into_iter().collect()
}

#[test]
fn a_build_is_private_until_installed_and_reads_answer_its_progress() {
    let dir = tempfile::tempdir().unwrap();
    let chains = due_a_build(dir.path(), &["first-build"]);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let paths = Paths::of(dir.path());
    assert!(paths.build.exists());
    assert!(!paths.index.exists());
    let writer = Writer::start(&archive).unwrap();
    archive.record_build_progress(1, 0);
    let pool = serving(&archive);
    match pool.reader() {
        Err(StoreError::IndexBuilding(build)) => assert_eq!(
            *build,
            crate::archive::index_build::IndexBuild {
                serving_previous: false,
                chains: 1,
                chains_read: 0,
                bytes: 0,
                bytes_read: 0,
            }
        ),
        other => panic!("expected an index build, got {:?}", other.err()),
    }
    // A write during the build goes into the build file as upkeep writes
    // it; the build then loads the chains it read.
    store_message(&archive, &writer, "during-build", "stored while building");
    let pipeline = ReadPool::open(&archive).unwrap();
    assert_eq!(messages(&pipeline), 1);
    assert_eq!(journal_mode(&paths.build), "delete");
    let runs = runs(&archive, &writer, &["first-build"], &chains);
    writer.build(runs).unwrap();
    assert!(!paths.dir.exists());
    assert_eq!(journal_mode(&paths.index), "wal");
    assert_eq!(archive.index_build(), None);
    assert_eq!(messages(&pool), 2);
    assert_eq!(messages(&pipeline), 2);
    // The installed index is the writer's from now on.
    store_message(&archive, &writer, "after-install", "ordinary write");
    assert_eq!(messages(&pool), 3);
}

#[test]
fn a_stopped_build_starts_again_from_the_stored_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::of(dir.path());
    let chains = due_a_build(dir.path(), &["stopped"]);
    {
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        let runs = runs(&archive, &writer, &["stopped"], &chains);
        assert!(runs[0].exists());
        drop(writer);
    }
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert!(!paths.runs.exists());
    assert!(archive.index_build().is_some());
    assert_eq!(messages(&ReadPool::open(&archive).unwrap()), 0);
    let writer = Writer::start(&archive).unwrap();
    let runs = runs(&archive, &writer, &["stopped"], &chains);
    writer.build(runs).unwrap();
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
    writer.build(Vec::new()).unwrap();
    // The installed index replaced the previous one; it holds what the
    // build loaded, which here is nothing.
    assert!(!paths.dir.exists());
    assert_eq!(messages(&pool), 0);
}

#[test]
fn a_rebuild_of_an_index_in_another_format_answers_index_building() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::of(dir.path());
    {
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        drop(archive);
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
    Writer::start(&archive).unwrap().build(Vec::new()).unwrap();
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
    let inputs = ["crashed", "second"];
    let chains = due_a_build(root, &inputs);
    let (archive, _lock) = Archive::open_or_create(root).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let mut runs = runs(&archive, &writer, &inputs[..1], &chains[..1]);
    let mut second = crate::writer::build::BuildWorker::create(&archive, 1, 32 << 20).unwrap();
    let mut batch = one_message("second", "second-m", "second text");
    let sender = writer.sender();
    let mut open = second
        .begin(&sender, &batch.input, chains[1], batch.length)
        .unwrap();
    for step in &mut batch.parse {
        crate::writer::ChainSink::step(&mut open, step).unwrap();
    }
    open.finish().unwrap();
    runs.extend(second.finish().unwrap());
    crate::fault::arm(&point, "", crate::fault::Action::StopProcess, 0, 1);
    writer.build(runs).unwrap();
    panic!("the configured durability boundary was not reached");
}

#[cfg(feature = "faults")]
fn crash_at(point: &str) -> tempfile::TempDir {
    crash_test_at(
        "writer::tests::private_build::private_build_crash_child",
        point,
    )
}

#[cfg(feature = "faults")]
pub(in crate::writer) fn crash_test_at(test: &str, point: &str) -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env("POLYLOGUE_PRIVATE_BUILD_ROOT", directory.path())
        .env("POLYLOGUE_PRIVATE_BUILD_POINT", point)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::inherit())
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

/// A build killed in the middle of its load leaves no index
/// and no partial build: the next open starts it again from the stored
/// bytes (docs/architecture.md, Durability).
#[cfg(feature = "faults")]
#[test]
fn a_killed_build_restarts_from_stored_bytes_and_leaves_no_partial_index() {
    let directory = crash_at("index.build.loading");
    let paths = Paths::of(directory.path());
    assert!(paths.runs.exists());
    assert!(!paths.index.exists());
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    assert!(!paths.runs.exists());
    assert!(!paths.index.exists());
    assert!(matches!(
        serving(&archive).reader().err(),
        Some(StoreError::IndexBuilding(_))
    ));
    let pipeline = ReadPool::open(&archive).unwrap();
    let reader = pipeline.reader().unwrap();
    // The stored bytes survive in the archive; the build file holds nothing
    // of the loaded run.
    assert_eq!(reader.status().unwrap().messages, 0);
    assert_eq!(reader.status().unwrap().inputs, 2);
    drop(reader);
    let writer = Writer::start(&archive).unwrap();
    let chains = [1, 1];
    let runs = runs(&archive, &writer, &["crashed", "second"], &chains);
    writer.build(runs).unwrap();
    assert_eq!(messages(&serving(&archive)), 2);
}

#[cfg(feature = "faults")]
#[test]
fn a_crash_between_the_finish_sync_and_the_rename_installs_at_the_next_open() {
    let directory = crash_at("index.build.before_rename");
    let paths = Paths::of(directory.path());
    assert!(paths.built.exists());
    assert!(!paths.index.exists());
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    assert!(!paths.dir.exists());
    assert_eq!(archive.index_build(), None);
    assert_eq!(messages(&serving(&archive)), 2);
}

#[cfg(feature = "faults")]
#[test]
fn a_crash_after_the_finish_sync_before_the_rename_rebuilds() {
    let directory = crash_at("index.build.after_finish_sync");
    let paths = Paths::of(directory.path());
    assert!(paths.build.exists());
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    assert!(archive.index_build().is_some());
    assert_eq!(messages(&ReadPool::open(&archive).unwrap()), 0);
}

#[cfg(feature = "faults")]
#[test]
fn pipeline_readers_open_the_ready_build_and_refuse_it_after_install() {
    crate::fault::clear();
    let dir = tempfile::tempdir().unwrap();
    let chains = due_a_build(dir.path(), &["ready-build"]);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let paths = Paths::of(dir.path());
    let writer = Writer::start(&archive).unwrap();
    let runs = runs(&archive, &writer, &["ready-build"], &chains);
    let pipeline = ReadPool::open(&archive).unwrap();
    crate::fault::arm(
        "index.build.before_rename",
        &archive.root.to_string_lossy(),
        crate::fault::Action::Hold,
        0,
        1,
    );
    let sender = writer.sender();
    let installer = std::thread::spawn(move || sender.build(runs));
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
    let chains = due_a_build(dir.path(), &["ready-first"]);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let runs = runs(&archive, &writer, &["ready-first"], &chains);
    let pipeline = ReadPool::open(&archive).unwrap();
    crate::fault::arm(
        "index.build.before_rename",
        &archive.root.to_string_lossy(),
        crate::fault::Action::Hold,
        0,
        1,
    );
    let sender = writer.sender();
    let installer = std::thread::spawn(move || sender.build(runs));
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
