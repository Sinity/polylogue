use super::*;

#[test]
fn duplicate_plain_acquisition_polls_cancellation_after_input_eof() {
    use std::cell::Cell;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    };
    struct Input<'a> {
        bytes: Cursor<&'a [u8]>,
        eof: &'a Cell<bool>,
    }
    impl Read for Input<'_> {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let n = self.bytes.read(out)?;
            if n == 0 {
                self.eof.set(true);
            }
            Ok(n)
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path(), vec![], 0o600).unwrap();
    let bytes = data(43, 8 << 20);
    let hash = store.put(&bytes[..], 1).unwrap();
    let path = file_of(&store, &hash);
    let original = fs::read(&path).unwrap();
    let eof = Cell::new(false);
    let mut completed_polls = 0;
    assert_eq!(
        store
            .put_plain_checked::<Error>(
                Input {
                    bytes: Cursor::new(&bytes),
                    eof: &eof
                },
                1,
                || {
                    if eof.get() {
                        completed_polls += 1;
                    }
                    Ok(())
                }
            )
            .unwrap(),
        hash
    );
    // There must be ongoing work after acquisition, rather than only a final poll.
    assert!(completed_polls > 128);
    eof.set(false);
    let cancelled = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&cancelled);
    let (request, requested) = mpsc::channel();
    let (ack, acknowledged) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        if requested.recv().is_ok() {
            signal.store(true, Ordering::Release);
            ack.send(()).unwrap();
        }
    });
    let before = fs::read_dir(dir.path().join("blobs/tmp")).unwrap().count();
    let mut interrupted_polls = 0;
    let mut sent = None;
    let result = store.put_plain_checked::<Error>(
        Input {
            bytes: Cursor::new(&bytes),
            eof: &eof,
        },
        1,
        || {
            if eof.get() {
                interrupted_polls += 1;
                if interrupted_polls == 64 {
                    sent = Some(std::time::Instant::now());
                    request.send(()).unwrap();
                    acknowledged.recv().unwrap();
                }
            }
            if cancelled.load(Ordering::Acquire) {
                Err(Error::Refused("synthetic acquisition cancellation".into()))
            } else {
                Ok(())
            }
        },
    );
    // A missing callback must fail the assertions instead of stranding the signal thread.
    drop(request);
    worker.join().unwrap();
    assert!(
        matches!(result, Err(Error::Refused(message)) if message == "synthetic acquisition cancellation")
    );
    assert_eq!(interrupted_polls, 64);
    assert!(interrupted_polls < completed_polls);
    let elapsed = sent.unwrap().elapsed();
    assert_eq!(fs::read(&path).unwrap(), original);
    assert_eq!(
        fs::read_dir(dir.path().join("blobs/tmp")).unwrap().count(),
        before + 1
    );
    eprintln!("CANCELLABLE_DEDUP input={} completed_after_eof_polls={completed_polls} cancelled_after_eof_polls={interrupted_polls} cancel_to_return_us={} candidate_kept=true", bytes.len(), elapsed.as_micros());
}
