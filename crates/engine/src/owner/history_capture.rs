//! One parked physical-history worker shares acquisition capacity with ordinary
//! imports. Its live SQLite/encoder state stays in that worker, never in a cursor.
use super::*;
pub(super) struct HistoryJob {
    pub job: CaptureJob,
    pub grant: Option<SyncSender<std::result::Result<(), StoreError>>>,
    pub regular_served: bool,
    pub progress: Option<polylogue_store::history_pages::turns::Progress>,
}
impl Actor {
    pub(super) fn start_history(
        &mut self,
        request: ImportRequest,
        reply: ImportReply,
    ) -> Result<()> {
        if self.history.is_some() || self.capture.is_some() {
            reply.send(Err(Error::Busy), false);
            return Ok(());
        }
        let preparation = (|| {
            self.writer.check_import_acquisition()?;
            Ok::<_, StoreError>((
                self.writer.blob_capture()?,
                self.writer.history_basis(&request.observation.borrowed())?,
            ))
        })();
        let (capture, basis) = match preparation {
            Ok(v) => v,
            Err(error) => {
                reply.send(Err(error.into()), false);
                return Ok(());
            }
        };
        let path = request.path;
        let cancelled = Arc::clone(&self.inner.cancelled);
        let sender = self.inner.sender.clone();
        let budget = self.history_budget;
        let join = thread::Builder::new()
            .name("polylogue-history-capture".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    capture.capture_history_pages_in_turns(
                        &path,
                        basis,
                        &cancelled,
                        budget,
                        |progress| {
                            let (grant, rx) = mpsc::sync_channel(1);
                            sender
                                .send(Message::HistoryYield(progress, grant))
                                .map_err(|_| StoreError::Cancelled)?;
                            loop {
                                if cancelled.load(Ordering::Acquire) {
                                    return Err(StoreError::Cancelled);
                                }
                                match rx.recv_timeout(Duration::from_millis(50)) {
                                    Ok(result) => return result,
                                    Err(RecvTimeoutError::Timeout) => {}
                                    Err(RecvTimeoutError::Disconnected) => {
                                        return Err(StoreError::Cancelled);
                                    }
                                }
                            }
                        },
                    )
                }))
                .map_err(|_| Error::WorkerPanicked)
                .and_then(|r| r.map_err(Error::Store));
                let _ = sender.send(Message::HistoryCaptured(result));
            });
        match join {
            Ok(join) => {
                self.history = Some(HistoryJob {
                    job: CaptureJob {
                        expected: None,
                        format: request.format,
                        context: request.context,
                        observation: request.observation,
                        allowance: request.metadata_allowance,
                        reply,
                        join,
                    },
                    grant: None,
                    progress: None,
                    regular_served: false,
                })
            }
            Err(error) => reply.send(Err(StoreError::Io(error).into()), false),
        }
        Ok(())
    }
    /// Called only after the bounded command mailbox has drained. A regular
    /// request arriving during an active quantum owns the next capture slot.
    pub(super) fn advance_history(&mut self) -> Result<()> {
        if self.capture.is_some() {
            return Ok(());
        }
        let parked = self.history.as_ref().is_some_and(|h| h.grant.is_some());
        if parked {
            if !self
                .history
                .as_ref()
                .expect("parked history")
                .regular_served
            {
                if let Some((request, configured, reply)) = self.waiting_capture.take() {
                    self.message(Message::Import(request, configured, reply))?;
                    return Ok(());
                }
            }
            let permission = self.writer.check_import_acquisition();
            if let Some(grant) = self.history.as_mut().and_then(|h| h.grant.take()) {
                let _ = grant.send(permission);
            }
        } else if self.history.is_none() {
            if let Some((request, configured, reply)) = self.waiting_capture.take() {
                self.message(Message::Import(request, configured, reply))?;
            }
        }
        Ok(())
    }
}
