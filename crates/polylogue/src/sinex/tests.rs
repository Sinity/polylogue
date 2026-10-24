use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, sync_channel},
};

struct Dropped(Arc<AtomicBool>);

impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn stop(mut publisher: Publisher) -> std::io::Result<()> {
    let (done, wait) = sync_channel(1);
    let thread = std::thread::spawn(move || {
        done.send(publisher.shutdown()).unwrap();
    });
    let result = wait
        .recv_timeout(Duration::from_secs(5))
        .expect("publisher did not join");
    thread.join().unwrap();
    result
}

fn pending() -> (Publisher, Receiver<()>, Arc<AtomicBool>, Arc<AtomicBool>) {
    let (started, wait) = sync_channel(1);
    let dropped = Arc::new(AtomicBool::new(false));
    let completed = Arc::new(AtomicBool::new(false));
    let held = Arc::clone(&dropped);
    let success = Arc::clone(&completed);
    let publisher = Publisher::spawn(runtime(), move || async move {
        let _held = Dropped(held);
        started.send(()).unwrap();
        std::future::pending::<()>().await;
        success.store(true, Ordering::SeqCst);
    })
    .unwrap();
    (publisher, wait, dropped, completed)
}

#[test]
fn stopping_a_pending_pass_drops_it_without_recording_success() {
    let (publisher, started, dropped, completed) = pending();
    started.recv_timeout(Duration::from_secs(5)).unwrap();
    stop(publisher).unwrap();
    assert!(dropped.load(Ordering::SeqCst));
    assert!(!completed.load(Ordering::SeqCst));
}

#[test]
fn stopping_the_publisher_wakes_its_periodic_wait() {
    let (started, wait) = sync_channel(1);
    let dropped = Arc::new(AtomicBool::new(false));
    let held = Arc::clone(&dropped);
    let publisher = Publisher::spawn(runtime(), move || async move {
        let _held = Dropped(held);
        let sleep = tokio::time::sleep(PUBLISH_EVERY);
        started.send(()).unwrap();
        sleep.await;
    })
    .unwrap();
    wait.recv_timeout(Duration::from_secs(5)).unwrap();
    stop(publisher).unwrap();
    assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn a_later_startup_error_drops_the_started_publisher() {
    fn fail(dropped: &mut Option<Arc<AtomicBool>>) -> std::io::Result<()> {
        let (_publisher, started, held, _) = pending();
        *dropped = Some(held);
        started.recv_timeout(Duration::from_secs(5)).unwrap();
        Err(std::io::Error::other("synthetic later startup failure"))
    }
    let mut dropped = None;
    assert!(fail(&mut dropped).is_err());
    assert!(dropped.unwrap().load(Ordering::SeqCst));
}

#[test]
fn a_publisher_panic_is_reported_when_joined() {
    struct PanicCleanup {
        unwinding: std::sync::mpsc::SyncSender<()>,
        release: Receiver<()>,
        cleaned: std::sync::mpsc::SyncSender<()>,
    }

    impl Drop for PanicCleanup {
        fn drop(&mut self) {
            self.unwinding.send(()).unwrap();
            self.release.recv().unwrap();
            self.cleaned.send(()).unwrap();
        }
    }

    let (unwinding, wait) = sync_channel(1);
    let (release, held) = sync_channel(1);
    let (cleaned, cleanup) = sync_channel(1);
    let publisher = Publisher::spawn(runtime(), move || async move {
        let _cleanup = PanicCleanup {
            unwinding,
            release: held,
            cleaned,
        };
        panic!("synthetic publisher failure");
    })
    .unwrap();

    // Panic handling and future cleanup precede the bounded shutdown check.
    // Nextest's process guard also bounds these synchronization waits.
    wait.recv().unwrap();
    assert!(!publisher.thread.as_ref().unwrap().is_finished());
    release.send(()).unwrap();
    cleanup.recv().unwrap();
    assert!(
        stop(publisher)
            .unwrap_err()
            .to_string()
            .contains("publisher panicked")
    );
}

#[test]
fn a_thread_local_snapshot_can_span_the_pending_pass() {
    let (started, wait) = sync_channel(1);
    let dropped = Arc::new(AtomicBool::new(false));
    let held = Arc::clone(&dropped);
    let publisher = Publisher::spawn(runtime(), move || async move {
        let local = std::rc::Rc::new(Dropped(held));
        started.send(()).unwrap();
        std::future::pending::<()>().await;
        drop(local);
    })
    .unwrap();
    wait.recv_timeout(Duration::from_secs(5)).unwrap();
    stop(publisher).unwrap();
    assert!(dropped.load(Ordering::SeqCst));
}
