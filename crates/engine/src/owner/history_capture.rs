//! One parked acquisition worker shares capacity with ordinary imports.
//! Physical history and a configured Claude main use the same grant loop.
use super::*;
pub(super) struct TurnCaptureJob {
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
        self.start_turn_capture(request, reply, None)
    }
    pub(super) fn start_claude(
        &mut self,
        request: ImportRequest,
        location: polylogue_store::configured_claude::Location,
        quantum_bytes: u64,
        reply: ImportReply,
    ) -> Result<()> {
        self.start_turn_capture(request, reply, Some((quantum_bytes, location)))
    }
    fn start_turn_capture(
        &mut self,
        request: ImportRequest,
        reply: ImportReply,
        claude: Option<(u64, polylogue_store::configured_claude::Location)>,
    ) -> Result<()> {
        let claude_quantum = claude.as_ref().map(|(bytes, _)| *bytes);
        if self.turn_capture.is_some() || self.capture.is_some() {
            reply.send(Err(Error::Busy), false);
            return Ok(());
        }
        let preparation = (|| {
            self.writer.check_import_acquisition()?;
            Ok::<_, StoreError>((
                self.writer.blob_capture()?,
                if claude_quantum.is_none() {
                    self.writer.history_basis(&request.observation.borrowed())?
                } else {
                    None
                },
                if claude_quantum.is_some() {
                    self.writer
                        .claude_input_at(&request.path, &request.observation.machine)?
                } else {
                    None
                },
            ))
        })();
        let (capture, basis, old) = match preparation {
            Ok(v) => v,
            Err(error) => {
                reply.send(Err(error.into()), false);
                return Ok(());
            }
        };
        let path = request.path;
        let cancelled = Arc::clone(&self.inner.cancelled);
        let sender = self.inner.sender.clone();
        let budget = claude_quantum
            .map(|bytes| polylogue_store::history_pages::turns::Budget {
                bytes,
                checkpoints: 512,
            })
            .unwrap_or(self.history_budget);
        let join = thread::Builder::new()
            .name("polylogue-source-capture".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let yield_turn = |progress| {
                        let (grant, rx) = mpsc::sync_channel(1);
                        sender
                            .send(Message::CaptureYield(progress, grant))
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
                    };
                    if let Some((_, location)) = claude {
                        crate::claude_capture::prepare(
                            capture, location, old, &cancelled, budget, yield_turn,
                        )
                        .map(PreparedCapture::Claude)
                    } else {
                        capture
                            .capture_history_pages_in_turns(
                                &path, basis, &cancelled, budget, yield_turn,
                            )
                            .map(PreparedCapture::History)
                    }
                }))
                .map_err(|_| Error::WorkerPanicked)
                .and_then(|r| r.map_err(Error::Store));
                let _ = sender.send(Message::TurnCaptured(result));
            });
        match join {
            Ok(join) => {
                self.turn_capture = Some(TurnCaptureJob {
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
    pub(super) fn advance_capture_turn(&mut self) -> Result<()> {
        if self.capture.is_some() {
            return Ok(());
        }
        let parked = self
            .turn_capture
            .as_ref()
            .is_some_and(|h| h.grant.is_some());
        if parked {
            if !self
                .turn_capture
                .as_ref()
                .expect("parked capture")
                .regular_served
            {
                if let Some((request, configured, reply)) = self.waiting_capture.take() {
                    self.message(Message::Import(request, configured, reply))?;
                    return Ok(());
                }
            }
            let permission = self.writer.check_import_acquisition();
            if let Some(grant) = self.turn_capture.as_mut().and_then(|h| h.grant.take()) {
                let _ = grant.send(permission);
            }
        } else if self.turn_capture.is_none() {
            if let Some((request, configured, reply)) = self.waiting_capture.take() {
                self.message(Message::Import(request, configured, reply))?;
            }
        }
        Ok(())
    }
}
