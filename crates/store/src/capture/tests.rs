use super::*;
use std::sync::mpsc;

struct Small {
    remaining: usize,
    calls: usize,
}
impl Read for Small {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.calls += 1;
        let n = bytes.len().min(self.remaining).min(7);
        bytes[..n].fill(b'x');
        self.remaining -= n;
        Ok(n)
    }
}
fn proof(w: &Writer, bytes: &[u8]) -> PreparedBlob {
    w.blob_capture()
        .unwrap()
        .capture(bytes, &AtomicBool::new(false))
        .unwrap()
}
fn rows(w: &Writer) -> i64 {
    w.archive
        .query_row("SELECT count(*) FROM blob", [], |r| r.get(0))
        .unwrap()
}
#[test]
fn many_small_reads_are_captured_and_metadata_is_idempotent() {
    let root = tempfile::tempdir().unwrap();
    let mut w = Writer::open(root.path()).unwrap();
    let mut reader = Small {
        remaining: 65537,
        calls: 0,
    };
    let p = w
        .blob_capture()
        .unwrap()
        .capture(&mut reader, &AtomicBool::new(false))
        .unwrap();
    assert!(reader.calls > 9000);
    assert_eq!(p.length(), 65537);
    assert!(p.stored_size() > 0);
    assert_eq!(p.hash(), BlobHash::of(&vec![b'x'; 65537]));
    assert_eq!(rows(&w), 0);
    let h = w.store_captured_blob(p).unwrap();
    let counter = w.status().unwrap().write_counter;
    assert_eq!(
        w.store_captured_blob(proof(&w, &vec![b'x'; 65537]))
            .unwrap(),
        h
    );
    assert_eq!(w.status().unwrap().write_counter, counter);
    assert_eq!(rows(&w), 1);
}
#[test]
fn cancel_or_read_error_records_nothing() {
    struct Failing;
    impl Read for Failing {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "synthetic source denied",
            ))
        }
    }
    struct Cancels<'a>(&'a AtomicBool);
    impl Read for Cancels<'_> {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            self.0.store(true, Ordering::Release);
            b[0] = 1;
            Ok(1)
        }
    }
    let root = tempfile::tempdir().unwrap();
    let w = Writer::open(root.path()).unwrap();
    let c = w.blob_capture().unwrap();
    let cancelled = AtomicBool::new(false);
    assert!(matches!(
        c.capture(Cancels(&cancelled), &cancelled),
        Err(Error::Cancelled)
    ));
    assert!(matches!(
        c.capture(&b"ignored"[..], &cancelled),
        Err(Error::Cancelled)
    ));
    match c.capture(Failing, &AtomicBool::new(false)) {
        Err(Error::Blob(hashed_blobs::Error::Io(e))) => {
            assert_eq!(e.kind(), io::ErrorKind::PermissionDenied);
            assert_eq!(e.to_string(), "synthetic source denied");
        }
        _ => panic!("source read error was not passed through"),
    }
    assert_eq!(rows(&w), 0);
}
#[test]
fn captured_blob_is_bound_to_its_archive_and_holds_the_lock() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let wa = Writer::open(a.path()).unwrap();
    let mut wb = Writer::open(b.path()).unwrap();
    let p = proof(&wa, b"bound");
    assert!(matches!(
        wb.store_captured_blob(p),
        Err(Error::Invariant(_))
    ));
    assert_eq!(rows(&wb), 0);
    let capture = wa.blob_capture().unwrap();
    drop(wa);
    assert!(matches!(
        Writer::open(a.path()),
        Err(Error::WriterHeld { .. })
    ));
    let p = capture
        .capture(&b"lock"[..], &AtomicBool::new(false))
        .unwrap();
    drop(capture);
    assert!(matches!(
        Writer::open(a.path()),
        Err(Error::WriterHeld { .. })
    ));
    drop(p);
    assert!(Writer::open(a.path()).is_ok());
}
#[test]
fn slow_capture_does_not_block_the_writer() {
    struct Blocked {
        entered: mpsc::Sender<()>,
        resume: mpsc::Receiver<()>,
        done: bool,
    }
    impl Read for Blocked {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            if self.done {
                return Ok(0);
            }
            self.entered.send(()).map_err(io::Error::other)?;
            self.resume.recv().map_err(io::Error::other)?;
            b[..4].copy_from_slice(b"slow");
            self.done = true;
            Ok(4)
        }
    }
    let root = tempfile::tempdir().unwrap();
    let mut w = Writer::open(root.path()).unwrap();
    let quick = proof(&w, b"quick");
    let capture = w.blob_capture().unwrap();
    let (entered_rx_tx, entered) = mpsc::channel();
    let (resume, receiver) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        capture.capture(
            Blocked {
                entered: entered_rx_tx,
                resume: receiver,
                done: false,
            },
            &AtomicBool::new(false),
        )
    });
    entered.recv().unwrap();
    w.store_captured_blob(quick).unwrap();
    assert_eq!(rows(&w), 1);
    resume.send(()).unwrap();
    let p = worker.join().unwrap().unwrap();
    w.store_captured_blob(p).unwrap();
    assert_eq!(rows(&w), 2);
}
#[test]
fn recording_repairs_writing_state_but_refuses_conflicts() {
    let root = tempfile::tempdir().unwrap();
    let mut w = Writer::open(root.path()).unwrap();
    let h = w.store_captured_blob(proof(&w, b"body")).unwrap();
    w.archive
        .execute(
            "UPDATE blob SET state='writing',stored_size=0 WHERE hash=?1",
            [h.bytes().as_slice()],
        )
        .unwrap();
    let counter = w.status().unwrap().write_counter;
    w.store_captured_blob(proof(&w, b"body")).unwrap();
    assert_eq!(w.status().unwrap().write_counter, counter + 1);
    let state: String = w
        .archive
        .query_row("SELECT state FROM blob", [], |r| r.get(0))
        .unwrap();
    assert_eq!(state, "present");
    w.archive.execute("UPDATE blob SET size=5", []).unwrap();
    assert!(matches!(
        w.store_captured_blob(proof(&w, b"body")),
        Err(Error::Invariant(_))
    ));
    w.archive
        .execute("UPDATE blob SET size=4,state='deleting'", [])
        .unwrap();
    assert!(matches!(
        w.store_captured_blob(proof(&w, b"body")),
        Err(Error::Invariant(_))
    ));
}
#[test]
fn corrupt_existing_blob_file_fails_capture() {
    let root = tempfile::tempdir().unwrap();
    let w = Writer::open(root.path()).unwrap();
    let p = proof(&w, b"original");
    let hex = p.hash().to_string();
    let path = root
        .path()
        .join("blobs")
        .join(&hex[..2])
        .join(format!("{hex}.zst"));
    std::fs::write(&path, b"not a zstd object").unwrap();
    drop(p);
    assert!(
        w.blob_capture()
            .unwrap()
            .capture(&b"original"[..], &AtomicBool::new(false))
            .is_err()
    );
    assert_eq!(std::fs::read(path).unwrap(), b"not a zstd object");
    assert_eq!(rows(&w), 0);
}
#[test]
fn regular_file_capture_refuses_fifo_symlink_and_directory() {
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let input = tempfile::tempdir().unwrap();
    let mut w = Writer::open(root.path()).unwrap();
    let capture = w.blob_capture().unwrap();
    let regular = input.path().join("source");
    std::fs::write(&regular, b"regular").unwrap();
    let link = input.path().join("link");
    symlink(&regular, &link).unwrap();
    assert!(
        capture
            .capture_regular_file(&link, &AtomicBool::new(false))
            .is_err()
    );
    assert!(
        capture
            .capture_regular_file(input.path(), &AtomicBool::new(false))
            .is_err()
    );
    let fifo = input.path().join("pipe");
    let status = std::process::Command::new("mkfifo")
        .arg("--mode=600")
        .arg(&fifo)
        .status()
        .unwrap();
    assert!(status.success(), "mkfifo fixture failed: {status}");
    assert!(
        capture
            .capture_regular_file(&fifo, &AtomicBool::new(false))
            .is_err()
    );
    assert!(matches!(
        capture.capture_regular_file(&fifo, &AtomicBool::new(true)),
        Err(Error::Cancelled)
    ));
    let proof = capture
        .capture_regular_file(&regular, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(proof.length(), 7);
    w.store_captured_blob(proof).unwrap();
}
