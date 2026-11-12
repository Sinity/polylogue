use super::*;

#[test]
fn stopped_sender_does_not_start_another_network_operation() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let stop = Stop::default();
    stop.requested.store(true, Ordering::Release);
    let result = runtime.block_on(stop.until(std::future::poll_fn(
        |_| -> std::task::Poll<Result<()>> { panic!("stopped work was polled") },
    )));
    assert!(matches!(result, Err(Error::Invalid(reason)) if reason == "producer sender stopped"));
}

#[test]
fn stopping_interrupts_an_operation_waiting_for_an_acknowledgement() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let stop = Arc::new(Stop::default());
        let (started, waiting) = tokio::sync::oneshot::channel();
        let worker_stop = stop.clone();
        let worker = tokio::spawn(async move {
            worker_stop.until(async {
                started.send(()).unwrap();
                std::future::pending::<Result<()>>().await
            }).await
        });
        waiting.await.unwrap();
        stop.requested.store(true, Ordering::Release);
        stop.wake.notify_one();
        assert!(matches!(worker.await.unwrap(), Err(Error::Invalid(reason)) if reason == "producer sender stopped"));
    });
}

#[test]
fn transient_write_failures_retry_but_deterministic_refusals_stop() {
    assert!(retryable(&Error::Unreachable("synthetic outage".into())));
    assert!(retryable(&Error::Io(std::io::Error::other(
        "synthetic fsync failure"
    ))));
    assert!(retryable(&Error::Blobs(hashed_blobs::Error::Io(
        std::io::Error::other("synthetic body write failure")
    ))));
    assert!(retryable(&Error::Store(polylogue_store::StoreError::Io(
        std::io::Error::other("synthetic queue write failure")
    ))));
    assert!(!retryable(&Error::Refused {
        name: "synthetic frame".into(),
        reason: "invalid content".into()
    }));
    assert!(!retryable(&Error::Invalid(
        "corrupt immutable frame".into()
    )));
    assert!(!retryable(&Error::Store(
        polylogue_store::StoreError::Invalid("deterministic content failure".into())
    )));
}

#[test]
fn sender_stops_on_deterministic_storage_refusal_and_retries_io_interruption() {
    for kind in [
        std::io::ErrorKind::PermissionDenied,
        std::io::ErrorKind::ReadOnlyFilesystem,
        std::io::ErrorKind::InvalidInput,
        std::io::ErrorKind::Unsupported,
        std::io::ErrorKind::InvalidData,
    ] {
        assert!(
            !retryable(&Error::Io(std::io::Error::from(kind))),
            "{kind:?}"
        );
        assert!(
            !retryable(&Error::Blobs(hashed_blobs::Error::Io(
                std::io::Error::from(kind)
            ))),
            "blob {kind:?}"
        );
        assert!(
            !retryable(&Error::Store(polylogue_store::StoreError::Io(
                std::io::Error::from(kind)
            ))),
            "queue {kind:?}"
        );
    }
    for kind in [
        std::io::ErrorKind::Interrupted,
        std::io::ErrorKind::WouldBlock,
        std::io::ErrorKind::TimedOut,
        std::io::ErrorKind::Other,
    ] {
        assert!(
            retryable(&Error::Io(std::io::Error::from(kind))),
            "{kind:?}"
        );
    }
    assert!(
        retryable(&Error::Io(std::io::Error::from_raw_os_error(5))),
        "a genuine EIO can recover on retry"
    );
}
