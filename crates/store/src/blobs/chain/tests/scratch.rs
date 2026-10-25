use super::*;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

struct OwnedChild(std::process::Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

const CHILD: &str = "POLYLOGUE_EXTENT_LIST_SCRATCH_CHILD";

#[test]
fn a_killed_extent_list_owner_leaves_no_named_payload() {
    if let Some(dir) = std::env::var_os(CHILD) {
        child(Path::new(&dir));
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let alias_dir = tempfile::tempdir().unwrap();
    let alias = alias_dir.path().join("extents");
    std::os::unix::fs::symlink(dir.path(), &alias).unwrap();
    let mut child = OwnedChild(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "blobs::chain::tests::scratch::a_killed_extent_list_owner_leaves_no_named_payload",
                "--nocapture",
            ])
            .env(CHILD, &alias)
            .env("TMPDIR", &alias)
            .env("SQLITE_TMPDIR", &alias)
            .stdout(Stdio::null())
            .spawn()
            .unwrap(),
    );
    while !dir.path().join("ready").exists() {
        if let Some(status) = child.0.try_wait().unwrap() {
            panic!("scratch child exited before readiness: {status}");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    // Kernel descriptor links use the underlying path, including when the
    // temporary directory was reached through a proc-FD or symlink alias.
    let physical = alias.canonicalize().unwrap();
    let descriptors = std::fs::read_dir(format!("/proc/{}/fd", child.0.id()))
        .unwrap()
        .filter_map(|e| std::fs::read_link(e.unwrap().path()).ok())
        .filter(|p| p.starts_with(&physical))
        .collect::<Vec<_>>();
    assert!(
        descriptors
            .iter()
            .filter(|p| p.to_string_lossy().ends_with(" (deleted)"))
            .count()
            >= 1,
        "{descriptors:?}"
    );
    assert!(
        descriptors
            .iter()
            .all(|p| p.to_string_lossy().ends_with(" (deleted)")),
        "{descriptors:?}"
    );
    eprintln!(
        "scratch ownership: {}",
        std::fs::read_to_string(dir.path().join("ready")).unwrap()
    );
    eprintln!("scratch descriptors: {descriptors:?}");
    child.0.kill().unwrap();
    assert!(!child.0.wait().unwrap().success());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}

fn child(dir: &Path) {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&conn).unwrap();
    conn.execute(
        "INSERT INTO input(id,format) VALUES ('input','synthetic')",
        [],
    )
    .unwrap();
    conn.execute("INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES ('input',1,'first',1,8388608,1)", []).unwrap();
    // Unique incompressible inline new blobs force the actual extents beyond its cache.
    for n in 0u64..512 {
        let bytes = (0u64..512)
            .flat_map(|i| {
                blake3::hash(&(n * 512 + i).to_le_bytes())
                    .as_bytes()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        let hash = BlobHash::of(&bytes);
        let data = zstd::bulk::compress(&bytes, 1).unwrap();
        conn.execute("INSERT INTO blob(hash,size,stored_size,place,encoding,state,data) VALUES (?1,16384,?2,'inline','plain','present',?3)", params![hash.0.as_slice(),data.len() as i64,data]).unwrap();
        conn.execute(
            "INSERT INTO extent VALUES ('input',1,?1,16384,?2,0)",
            params![(n * 16384) as i64, hash.0.as_slice()],
        )
        .unwrap();
    }
    let chain = StoredChain::load(&conn, "", "input", 1, 8 << 20).unwrap();
    let locked = chain.extents.as_ref().unwrap().connection();
    assert_eq!(locked.path(), Some(""));
    let pages: i64 = locked
        .query_row("PRAGMA page_count", [], |r| r.get(0))
        .unwrap();
    assert!(pages > 2048);
    assert_eq!(
        locked
            .query_row("PRAGMA cache_size", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        -64
    );
    drop(locked);
    let start = Instant::now();
    let idle: Vec<_> = (0..32)
        .map(|_| StoredChain::load_range(&conn, "", "input", 1, 0, 1).unwrap())
        .collect();
    for chain in &idle {
        let db = chain.extents.as_ref().unwrap().connection();
        assert_eq!(
            db.query_row("PRAGMA cache_size", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            -64
        );
    }
    let fds = std::fs::read_dir("/proc/self/fd").unwrap().count();
    std::fs::write(dir.join("ready"), format!("33 extent lists (32 small idle); total process descriptors={fds}; small extents capture={:?}; 64KiB cache target each (not RSS)",start.elapsed())).unwrap();
    loop {
        std::thread::park();
    }
}
