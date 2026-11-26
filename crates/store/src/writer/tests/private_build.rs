//! The private index build at its durability boundaries
//! (docs/architecture.md, Durability; docs/operations.md, Replacing an index
//! from another schema).
use super::*;
use crate::index::file::{IndexFiles, Installed, Paths, RebuildReason, Rebuilding};

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
        .store_group(vec![crate::staging::PreparedBatch {
            batch: reading,
            files: file.into_iter().collect(),
        }])
        .unwrap();
    let chain = stored.into_iter().next().unwrap().unwrap().chain;
    let report = writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    chain
}

fn serving(archive: &Archive) -> ReadPool {
    ReadPool::serving(archive, 1, 8 << 20).unwrap()
}

/// The messages a reader of `pool` sees.
fn messages(pool: &ReadPool) -> u64 {
    rows(&pool.reader().unwrap())
}

/// The messages `reader` sees.
fn rows(reader: &crate::read::Reader<'_>) -> u64 {
    reader
        .query("SELECT count(*) AS n FROM message", &[])
        .unwrap()[0]["n"]
        .as_u64()
        .unwrap()
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
        assert!(
            archive.index_build().is_some(),
            "a new archive's index is built"
        );
        let writer = Writer::start(&archive).unwrap();
        crate::fixture::install_empty_rebuild(&archive, &writer.sender()).unwrap();
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
    let mut run = crate::index::rebuild::RebuildWorker::create(archive, 0, 32 << 20).unwrap();
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

/// Two chains whose texts compress to the same group bytes, split into texts
/// at different places: the build stores the group once, and the texts only
/// the second chain holds are read from the first chain's row.
#[test]
fn a_group_two_chains_bring_is_one_row_and_each_text_reads_from_it() {
    let dir = tempfile::tempdir().unwrap();
    let inputs = ["joined", "split"];
    let chains = due_a_build(dir.path(), &inputs);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let mut run = crate::index::rebuild::RebuildWorker::create(&archive, 0, 32 << 20).unwrap();
    for (input, &chain) in inputs.iter().zip(&chains) {
        let mut batch = one_message(input, &format!("{input}-m"), "walrus narwhal");
        if *input == "split" {
            batch.parse[0].output.messages[0].message.blocks = vec![
                Block::Text {
                    text: "walrus ".into(),
                },
                Block::Text {
                    text: "narwhal".into(),
                },
            ];
        }
        let sender = writer.sender();
        let mut open = run
            .begin(&sender, &batch.input, chain, batch.length)
            .unwrap();
        for step in &mut batch.parse {
            crate::writer::ChainSink::step(&mut open, step).unwrap();
        }
        open.finish().unwrap();
    }
    crate::fixture::rebuild_from_runs(&archive, &writer.sender(), run.finish().unwrap()).unwrap();
    let index = crate::archive::open_read(&archive.index_db()).unwrap();
    let groups: i64 = index
        .query_row("SELECT count(DISTINCT group_id) FROM text", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(groups, 1);
    let mut texts: Vec<String> = index
        .prepare("SELECT id FROM text ORDER BY id")
        .unwrap()
        .query_map([], |r| r.get::<_, i64>(0))
        .unwrap()
        .map(|id| crate::text_rows::read(&index, id.unwrap()).unwrap())
        .collect();
    texts.sort();
    assert_eq!(texts, ["narwhal", "walrus ", "walrus narwhal"]);
    let found: i64 = index
        .query_row(
            &format!(
                "SELECT count(*) FROM ({})",
                crate::search::matching_texts(crate::search::SnippetIndex::Prose, "'narwhal'")
            ),
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(found, 2);
}

/// The sessions assembling counts and the total it counts towards are one
/// set: a session a message names without any chain stating its record is
/// assembled, so it is also in the total (docs/operations.md, Replacing an
/// index from another schema).
#[cfg(feature = "faults")]
#[test]
fn assembling_counts_up_to_its_total_when_a_message_names_an_unstated_session() {
    use crate::archive::build_progress::{CountUnit, StepCount};
    let dir = tempfile::tempdir().unwrap();
    let inputs = ["stated", "unstated"];
    let chains = due_a_build(dir.path(), &inputs);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let mut run = crate::index::rebuild::RebuildWorker::create(&archive, 0, 32 << 20).unwrap();
    for (input, &chain) in inputs.iter().zip(&chains) {
        let mut batch = one_message(input, &format!("{input}-m"), &format!("{input} text"));
        if *input == "unstated" {
            batch.parse[0].output.sessions.clear();
        }
        let sender = writer.sender();
        let mut open = run
            .begin(&sender, &batch.input, chain, batch.length)
            .unwrap();
        for step in &mut batch.parse {
            crate::writer::ChainSink::step(&mut open, step).unwrap();
        }
        open.finish().unwrap();
    }
    let runs: Vec<_> = run.finish().unwrap().into_iter().collect();
    // Held once assembly has ended, the rebuild's progress holds the
    // assembling stage's last count.
    crate::fault::arm("index.deferred", "", crate::fault::Action::Hold, 0, 1);
    let rebuilt = archive.clone();
    let sender = writer.sender();
    let built =
        std::thread::spawn(move || crate::fixture::rebuild_from_runs(&rebuilt, &sender, runs));
    crate::fault::wait_held();
    let (_, progress) = archive.build_progress().unwrap();
    crate::fault::release();
    built.join().unwrap().unwrap();
    crate::fault::clear();
    let counts: Vec<(String, Option<StepCount>)> = progress
        .finished
        .into_iter()
        .filter(|step| {
            step.count
                .as_ref()
                .is_some_and(|count| count.unit == CountUnit::Sessions)
        })
        .map(|step| (step.name, step.count))
        .collect();
    let sessions = |done, total| {
        Some(StepCount {
            done,
            total: Some(total),
            unit: CountUnit::Sessions,
        })
    };
    assert_eq!(counts, [("assembling".to_owned(), sessions(2, 2))]);
}

/// The index writer never writes the rebuild's file. With no installed
/// index, a write during the rebuild is stored in the archive and refused
/// for the index with `index_building`; the installed rebuild covers the
/// archive up to the position it started at, and the chain stored after it
/// is due (docs/architecture.md, Durability: the index file).
#[test]
fn a_rebuild_is_private_until_installed_and_reads_answer_its_progress() {
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
                reason: RebuildReason::NoIndex,
                chains: 1,
                chains_read: 0,
                bytes: 0,
                bytes_read: 0,
            }
        ),
        other => panic!("expected an index build, got {:?}", other.err()),
    }
    let rebuild = crate::index::rebuild::Rebuild::start(
        &archive,
        polylogue_model::WallClock::fixed(jiff::Timestamp::UNIX_EPOCH),
    )
    .unwrap();
    let started_at = rebuild.position();
    // A write while the rebuild runs: its bytes are stored, its parse is
    // refused for the index, which does not exist yet.
    let (blob, file) = archive.prepare_blob(&SOURCE_RECORD).unwrap();
    let mut reading = one_message("during-build", "during-build-m", "stored while building");
    reading.new_blob = Some(blob);
    let stored = writer
        .store_group(vec![crate::staging::PreparedBatch {
            batch: reading,
            files: file.into_iter().collect(),
        }])
        .unwrap();
    assert!(
        matches!(&stored[..], [Err(StoreError::IndexBuilding(build))] if build.reason == RebuildReason::NoIndex),
        "{stored:?}"
    );
    let pipeline = ReadPool::open(&archive).unwrap();
    assert_eq!(messages(&pipeline), 0);
    assert_eq!(journal_mode(&paths.build), "delete");
    let runs = runs(&archive, &writer, &["first-build"], &chains);
    let stages = rebuild.finish(runs, 0, 2, &writer.sender()).unwrap();
    assert_eq!(
        stages.iter().map(|stage| stage.stage).collect::<Vec<_>>(),
        crate::index::rebuild::RebuildStage::ALL
    );
    assert!(!paths.dir.exists());
    assert_eq!(journal_mode(&paths.index), "wal");
    assert_eq!(archive.index_build(), None);
    // The rebuild holds what it read; the write made meanwhile is stored
    // after the position it covers, and is due.
    assert_eq!(messages(&pool), 1);
    assert_eq!(messages(&pipeline), 1);
    let index = crate::archive::open_read(&paths.index).unwrap();
    assert_eq!(
        crate::meta_get(&index, crate::index::file::ARCHIVE_POSITION).unwrap(),
        Some(started_at.0.to_string())
    );
    let reader = pool.reader().unwrap();
    let due: Vec<String> = reader
        .chains_behind_index(
            &[("claude-code".to_owned(), "synthetic=1".to_owned())],
            crate::read::RevisionChains::All,
            None,
        )
        .unwrap()
        .into_iter()
        .map(|chain| chain.input)
        .collect();
    assert_eq!(due, ["during-build"]);
    drop(reader);
    // The installed index is the writer's from now on.
    store_message(&archive, &writer, "after-install", "ordinary write");
    assert_eq!(messages(&pool), 2);
}

/// The planner's total and the workers' count are sampled apart, so a
/// sample can hold a count the total has not caught up with.
#[test]
fn a_builds_files_read_stay_within_its_files() {
    let dir = tempfile::tempdir().unwrap();
    due_a_build(dir.path(), &["first-build"]);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    archive.record_build_progress(36_571, 37_383);
    let build = archive.index_build().unwrap();
    assert_eq!((build.chains_read, build.chains), (37_383, 37_383));
    archive.record_build_progress(40_000, 37_383);
    let build = archive.index_build().unwrap();
    assert_eq!((build.chains_read, build.chains), (37_383, 40_000));
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
    assert!(!paths.sorted.exists());
    assert!(archive.index_build().is_some());
    assert_eq!(messages(&ReadPool::open(&archive).unwrap()), 0);
    let writer = Writer::start(&archive).unwrap();
    let runs = runs(&archive, &writer, &["stopped"], &chains);
    crate::fixture::rebuild_from_runs(&archive, &writer.sender(), runs).unwrap();
    assert_eq!(messages(&serving(&archive)), 1);
}

/// A round that fails, before composing or after composition and the
/// search triggers, leaves the build file as the next open would after a
/// crash: empty, so a pool reading it finds no chain read. The next round
/// on the same writer reads the chains again and takes the whole batch
/// path, composing from its runs.
#[test]
fn a_failed_round_leaves_an_empty_build_file_for_the_next() {
    for point in ["index.build.composing", "index.deferred"] {
        let dir = tempfile::tempdir().unwrap();
        let chains = due_a_build(dir.path(), &["first-round"]);
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        let pipeline = ReadPool::open(&archive).unwrap();
        let first = runs(&archive, &writer, &["first-round"], &chains);
        crate::fault::arm(point, "", crate::fault::Action::Fail, 0, 1);
        let failed = crate::fixture::rebuild_from_runs(&archive, &writer.sender(), first);
        crate::fault::clear();
        assert!(failed.is_err(), "{point}");
        let status = pipeline
            .reader()
            .unwrap()
            .status(&crate::fixture::synthetic_contracts())
            .unwrap();
        assert_eq!(status.parse_states.get("done"), None, "{point}");
        assert_eq!(status.inputs, 1, "{point}");
        let again = runs(&archive, &writer, &["first-round"], &chains);
        let phases = crate::fixture::rebuild_from_runs(&archive, &writer.sender(), again).unwrap();
        assert_eq!(
            phases.iter().map(|p| p.stage.name()).collect::<Vec<_>>(),
            [
                "parsing",
                "keying texts",
                "loading parses",
                "assembling",
                "indexing",
                "full text",
                "installing"
            ],
            "{point}"
        );
        assert_eq!(messages(&serving(&archive)), 1, "{point}");
    }
}

#[test]
fn a_rebuild_keeps_serving_a_same_schema_index_until_it_is_installed() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::of(dir.path());
    {
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        crate::fixture::install_empty_rebuild(&archive, &writer.sender()).unwrap();
        store_message(&archive, &writer, "served", "served while rebuilt");
    }
    // An extra index changes the schema but not its identity: this build's
    // code can still read the file.
    crate::archive::open_write(&paths.index, crate::archive::cache::SETUP)
        .unwrap()
        .execute_batch("CREATE INDEX synthetic_extra ON message(pk)")
        .unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert!(paths.index.exists());
    assert_eq!(
        archive.index_build(),
        Some(crate::archive::index_build::IndexBuild {
            reason: RebuildReason::SchemaDiffers,
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
    crate::fixture::install_empty_rebuild(&archive, &writer.sender()).unwrap();
    // The rebuild replaced the index; it holds what the rebuild loaded,
    // which here is nothing.
    assert!(!paths.dir.exists());
    assert_eq!(messages(&pool), 0);
}

#[test]
fn a_rebuild_of_an_index_in_another_format_answers_index_building() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::of(dir.path());
    {
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        crate::fixture::install_empty_rebuild(&archive, &Writer::start(&archive).unwrap().sender())
            .unwrap();
    }
    crate::archive::open_write(&paths.index, crate::archive::cache::SETUP)
        .unwrap()
        .execute_batch("CREATE TABLE synthetic_older(value)")
        .unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert!(matches!(
        serving(&archive).reader().err(),
        Some(StoreError::IndexBuilding(build)) if build.reason == RebuildReason::SchemaDiffers
    ));
    crate::fixture::install_empty_rebuild(&archive, &Writer::start(&archive).unwrap().sender())
        .unwrap();
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
    // Preparing the second input stops the process with the first input's
    // run already written; the other points are reached by the load.
    let subject = if point == "index.build.prepare" {
        "second"
    } else {
        ""
    };
    crate::fault::arm(&point, subject, crate::fault::Action::StopProcess, 0, 1);
    let mut runs = runs(&archive, &writer, &inputs[..1], &chains[..1]);
    let mut second = crate::index::rebuild::RebuildWorker::create(&archive, 1, 32 << 20).unwrap();
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
    crate::fixture::rebuild_from_runs(&archive, &writer.sender(), runs).unwrap();
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

/// A build killed while preparing an input and a build killed in the middle
/// of its load each leave no index and no partial build: the next open starts
/// it again from the stored bytes (docs/architecture.md, Durability).
#[cfg(feature = "faults")]
#[test]
fn a_crash_during_a_build_discards_it_and_restarts_from_the_stored_bytes() {
    for point in ["index.build.prepare", "index.build.loading"] {
        let directory = crash_at(point);
        let paths = Paths::of(directory.path());
        assert!(paths.sorted.exists(), "{point}");
        assert!(!paths.index.exists(), "{point}");
        let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
        assert!(!paths.sorted.exists(), "{point}");
        assert!(!paths.index.exists(), "{point}");
        assert!(
            matches!(
                serving(&archive).reader().err(),
                Some(StoreError::IndexBuilding(_))
            ),
            "{point}"
        );
        let pipeline = ReadPool::open(&archive).unwrap();
        let reader = pipeline.reader().unwrap();
        // The stored bytes survive in the archive; the build file holds
        // nothing of the prepared runs.
        assert_eq!(rows(&reader), 0, "{point}");
        assert_eq!(reader.status(&[]).unwrap().inputs, 2, "{point}");
        drop(reader);
        let writer = Writer::start(&archive).unwrap();
        let chains = [1, 1];
        let runs = runs(&archive, &writer, &["crashed", "second"], &chains);
        crate::fixture::rebuild_from_runs(&archive, &writer.sender(), runs).unwrap();
        assert_eq!(messages(&serving(&archive)), 2, "{point}");
    }
}

#[cfg(feature = "faults")]
#[test]
fn a_crash_between_the_finish_sync_and_the_rename_installs_at_the_next_open() {
    let directory = crash_at("index.build.before_rename");
    let paths = Paths::of(directory.path());
    assert!(paths.ready.exists());
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
fn a_pipeline_reader_opened_before_an_install_reads_the_installed_index() {
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
    let rebuilt = archive.clone();
    let installer =
        std::thread::spawn(move || crate::fixture::rebuild_from_runs(&rebuilt, &sender, runs));
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
    assert_eq!(selected_path, paths.ready);
    // Its read had not started: it starts on the installed index.
    assert_eq!(rows(&reader), 1);
    assert_eq!(reader.index_identity().unwrap(), {
        use std::os::unix::fs::MetadataExt as _;
        let installed = std::fs::metadata(&paths.index).unwrap();
        (installed.dev(), installed.ino())
    });
    drop(reader);
    assert_eq!(archive.index_db(), paths.index);
    assert_eq!(messages(&pipeline), 1);
}

#[cfg(feature = "faults")]
#[test]
fn a_request_read_that_meets_the_install_reads_the_installed_index() {
    crate::fault::clear();
    let dir = tempfile::tempdir().unwrap();
    let chains = due_a_build(dir.path(), &["read-current"]);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let paths = Paths::of(dir.path());
    let writer = Writer::start(&archive).unwrap();
    let runs = runs(&archive, &writer, &["read-current"], &chains);
    let pipeline = ReadPool::open(&archive).unwrap();
    crate::fault::arm(
        "index.build.before_rename",
        &archive.root.to_string_lossy(),
        crate::fault::Action::Hold,
        0,
        1,
    );
    let sender = writer.sender();
    let rebuilt = archive.clone();
    let installer =
        std::thread::spawn(move || crate::fixture::rebuild_from_runs(&rebuilt, &sender, runs));
    while !crate::fault::held() && !installer.is_finished() {
        std::thread::yield_now();
    }
    assert!(
        crate::fault::held(),
        "the install reached its rename boundary"
    );
    // A request's reader is checked out on the ready file; the install
    // completes before its read starts, as when the writer installs while
    // a request is answered.
    let limit = crate::read::limit::ReadLimit::without_deadline(archive.stop().child());
    let reader = pipeline
        .checkout(crate::read::Lane::Interactive, &limit)
        .unwrap();
    let opened = archive.index_db();
    crate::fault::release();
    installer.join().unwrap().unwrap();
    crate::fault::clear();

    assert_eq!(opened, paths.ready);
    assert_eq!(rows(&reader), 1);
    assert_eq!(archive.index_db(), paths.index);
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
    let rebuilt = archive.clone();
    let installer =
        std::thread::spawn(move || crate::fixture::rebuild_from_runs(&rebuilt, &sender, runs));
    while !crate::fault::held() && !installer.is_finished() {
        std::thread::yield_now();
    }
    let held = crate::fault::held();
    let checkout = pipeline.reader();
    let started = checkout.as_ref().ok().map(rows);
    crate::fault::release();
    let installed = installer.join();
    crate::fault::clear();

    assert!(held, "the install reached its rename boundary");
    installed.unwrap().unwrap();
    let reader = checkout.expect("open the ready build before install");
    assert_eq!(started.unwrap(), 1);
    store_message(&archive, &writer, "ready-second", "written after install");
    drop(writer);
    assert_eq!(rows(&reader), 1);
    assert_eq!(messages(&pipeline), 2);
}

/// Opens an archive whose index names another archive, stopping the
/// process at `POLYLOGUE_PRIVATE_BUILD_POINT` in the open's start of the
/// rebuild.
#[cfg(feature = "faults")]
#[test]
fn index_reason_crash_child() {
    let Some(root) = std::env::var_os("POLYLOGUE_PRIVATE_BUILD_ROOT") else {
        return;
    };
    let point = std::env::var("POLYLOGUE_PRIVATE_BUILD_POINT").unwrap();
    let root = std::path::Path::new(&root);
    {
        let (archive, _lock) = Archive::open_or_create(root).unwrap();
        crate::fixture::install_empty_rebuild(&archive, &Writer::start(&archive).unwrap().sender())
            .unwrap();
    }
    let index =
        crate::archive::open_write(&root.join("index.db"), crate::archive::cache::SETUP).unwrap();
    crate::meta_set(&index, "archive_id", "0123456789abcdef0123456789abcdef").unwrap();
    drop(index);
    crate::fault::arm(&point, "", crate::fault::Action::StopProcess, 0, 1);
    let _ = Archive::open_or_create(root);
    panic!("the configured durability boundary was not reached");
}

/// A kill right after the rebuild's reason is written keeps the reason:
/// with the index that gave it gone, the next open rebuilds for it, and
/// reads and `status` name it.
#[cfg(feature = "faults")]
#[test]
fn the_reason_survives_a_kill_after_it_is_written() {
    let directory = crash_test_at(
        "writer::tests::private_build::index_reason_crash_child",
        "index.after_reason",
    );
    let paths = Paths::of(directory.path());
    assert!(paths.reason.exists());
    assert!(!paths.build.exists());
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(crate::index::file::with_suffix(&paths.index, suffix));
    }
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    assert_eq!(
        archive.index_files().unwrap(),
        IndexFiles {
            installed: Installed::None,
            rebuilding: Rebuilding::Writing {
                reason: RebuildReason::OtherArchive
            },
        }
    );
    assert_eq!(
        archive.index_build().unwrap().reason,
        RebuildReason::OtherArchive
    );
    assert!(matches!(
        serving(&archive).reader().err(),
        Some(StoreError::IndexBuilding(build)) if build.reason == RebuildReason::OtherArchive
    ));
}

/// A kill at each step of an install restarts to a complete installed
/// index or to an install of the finished rebuild, never to a half state.
/// The fault fires in the child, which `crash_test_at` sees stopped at it.
#[cfg(feature = "faults")]
#[test]
fn a_kill_at_each_step_of_the_install_ends_installed() {
    for point in ["index.install.after_reason", "index.install.after_rename"] {
        let directory = crash_at(point);
        let paths = Paths::of(directory.path());
        let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
        assert!(!paths.dir.exists(), "{point}");
        let files = archive.index_files().unwrap();
        assert_eq!(files.rebuilding, Rebuilding::No, "{point}");
        assert!(files.serves(), "{point}");
        assert_eq!(messages(&serving(&archive)), 2, "{point}");
    }
}

/// An install that fails at its rename in a running daemon leaves the
/// rebuild ready and the writer without an index connection: reads go on
/// from the index they had, `status` shows the failure, and the next commit
/// installs it.
#[cfg(feature = "faults")]
#[test]
fn a_failed_install_is_installed_at_the_next_commit() {
    crate::fault::clear();
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::of(dir.path());
    {
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        crate::fixture::install_empty_rebuild(&archive, &writer.sender()).unwrap();
        store_message(&archive, &writer, "served", "served while rebuilt");
    }
    crate::archive::open_write(&paths.index, crate::archive::cache::SETUP)
        .unwrap()
        .execute_batch("CREATE INDEX synthetic_extra ON message(pk)")
        .unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let pool = serving(&archive);
    let writer = Writer::start(&archive).unwrap();
    crate::fault::arm(
        "index.build.before_rename",
        &archive.root.to_string_lossy(),
        crate::fault::Action::Fail,
        0,
        1,
    );
    assert!(crate::fixture::install_empty_rebuild(&archive, &writer.sender()).is_err());
    assert_eq!(
        crate::fault::recorded("index.build.before_rename"),
        [(
            archive.root.to_string_lossy().into_owned(),
            crate::writer::WRITER_THREAD.to_owned()
        )]
    );
    crate::fault::clear();
    assert_eq!(archive.index_files().unwrap().rebuilding, Rebuilding::Ready);
    assert!(paths.ready.exists());
    let failure = archive.index_status().unwrap().install_failure.unwrap();
    assert_eq!(failure.subject, "installing the rebuilt index");
    assert_eq!(messages(&pool), 1);
    // The next commit event installs the rebuild before the write it
    // carries is applied.
    store_message(
        &archive,
        &writer,
        "after-failure",
        "written after the failure",
    );
    let status = archive.index_status().unwrap();
    assert_eq!(status.files.rebuilding, Rebuilding::No);
    assert!(status.files.serves());
    assert_eq!(status.install_failure, None);
    assert!(!paths.dir.exists());
    assert_eq!(messages(&pool), 1);
}

/// A read keeps the index it started on through an install; the read after
/// its restart is a new read and starts on the installed index.
#[test]
fn a_restarted_read_starts_on_the_installed_index() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::of(dir.path());
    {
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        crate::fixture::install_empty_rebuild(&archive, &writer.sender()).unwrap();
        store_message(&archive, &writer, "served", "served while rebuilt");
    }
    crate::archive::open_write(&paths.index, crate::archive::cache::SETUP)
        .unwrap()
        .execute_batch("CREATE INDEX synthetic_extra ON message(pk)")
        .unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let pool = serving(&archive);
    let reader = pool.reader().unwrap();
    assert_eq!(rows(&reader), 1);
    // The rebuild loads nothing, so the installed index holds no message.
    crate::fixture::install_empty_rebuild(&archive, &Writer::start(&archive).unwrap().sender())
        .unwrap();
    assert_eq!(rows(&reader), 1);
    reader.restart().unwrap();
    assert_eq!(rows(&reader), 0);
    assert_eq!(reader.index_identity().unwrap(), {
        use std::os::unix::fs::MetadataExt as _;
        let installed = std::fs::metadata(&paths.index).unwrap();
        (installed.dev(), installed.ino())
    });
}

/// Commits made while a rebuild runs (a chain that grows, a new input, and
/// an alternate's text) are in the index after the rebuild is installed and
/// one incremental round, whether a current index was installed meanwhile
/// or none was (docs/architecture.md, one path: a build reads the record as
/// it stood at one position).
#[test]
fn commits_during_a_rebuild_reach_the_index_after_its_install_and_one_round() {
    for current in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::of(dir.path());
        let chains = if current {
            let chains = {
                let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
                let writer = Writer::start(&archive).unwrap();
                crate::fixture::install_empty_rebuild(&archive, &writer.sender()).unwrap();
                vec![store_message(&archive, &writer, "grows", "grows text")]
            };
            // An extra index: this build reads the file, and rebuilds it.
            crate::archive::open_write(&paths.index, crate::archive::cache::SETUP)
                .unwrap()
                .execute_batch("CREATE INDEX synthetic_extra ON message(pk)")
                .unwrap();
            chains
        } else {
            due_a_build(dir.path(), &["grows"])
        };
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        assert_eq!(archive.index_files().unwrap().serves(), current);
        let writer = Writer::start(&archive).unwrap();
        let rebuild = crate::index::rebuild::Rebuild::start(
            &archive,
            polylogue_model::WallClock::fixed(jiff::Timestamp::UNIX_EPOCH),
        )
        .unwrap();
        let runs = runs(&archive, &writer, &["grows"], &chains);
        // While it runs: the chain grows, a new input arrives, and an
        // alternate is written. Their bytes and rows are committed to the
        // archive; their parses go to the installed index when there is one.
        let commit = |reading: Batch| -> Batch {
            let mut bytes_only = reading.clone();
            bytes_only.parse.clear();
            let prepared = archive.prepare_bytes(&SOURCE_RECORD[..]).unwrap();
            let stored = writer.store_bytes(prepared, bytes_only).unwrap();
            let mut reading = reading;
            reading.change = Change::Reparse {
                chain: stored.chain,
            };
            reading.observation = None;
            reading.continues = false;
            reading
        };
        let grown = commit(batch(
            "grows",
            Route::Live,
            100,
            vec![MessageRow {
                response: None,
                page_text: false,
                id_kind: polylogue_model::IdKind::Native,
                offset: 100,
                end: 200,
                message: text_message("grows-m2", "grows", Role::User, "grown meanwhile", None),
            }],
            &["grows"],
        ));
        let fresh = commit(one_message("fresh", "fresh-m", "a new input"));
        writer
            .archive()
            .commit(crate::archive::writer::Group::RestoredRequest(
                crate::archive::writer::tests::alternate(),
            ))
            .unwrap();
        for reading in [grown.clone(), fresh.clone()] {
            writer.send(reading).unwrap();
        }
        let report = writer.flush().unwrap();
        assert_eq!(report.errors.is_empty(), current, "{:?}", report.errors);
        rebuild.finish(runs, 0, 2, &writer.sender()).unwrap();
        // The installed rebuild holds what the archive had when it started;
        // what came after is due, and one round of incremental updates
        // brings it in.
        let pool = serving(&archive);
        let reader = pool.reader().unwrap();
        let mut due: Vec<String> = reader
            .chains_behind_index(
                &[("claude-code".to_owned(), "synthetic=1".to_owned())],
                crate::read::RevisionChains::All,
                None,
            )
            .unwrap()
            .into_iter()
            .map(|chain| chain.input)
            .collect();
        due.sort();
        assert_eq!(due, ["fresh", "grows"], "current: {current}");
        drop(reader);
        for reading in [grown, fresh] {
            writer.send(reading).unwrap();
        }
        let report = writer.flush().unwrap();
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(messages(&pool), 3, "current: {current}");
        let mut found = 0;
        pool.reader()
            .unwrap()
            .visit_alternate_search("pierwszy", Some("pl"), |_| {
                found += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(found, 1, "current: {current}");
    }
}
