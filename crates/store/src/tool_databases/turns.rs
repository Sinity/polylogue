use super::{ExportError, Exporter};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;

type Inputs = Vec<(PathBuf, PathBuf)>;

enum Command {
    Start(Inputs, usize),
    Continue(usize),
}

#[derive(Default)]
struct Updates {
    paths: Vec<(PathBuf, Option<PathBuf>)>,
    errors: Vec<String>,
}

enum Reply {
    Paused(Updates),
    Finished(Updates),
}

/// Work completed during one cooperative capture turn.
#[derive(Default)]
pub struct CaptureTurn {
    pub pending: bool,
    pub errors: Vec<String>,
}

/// One capture worker owns its SQLite cursors on its stack. It cannot write
/// between permits, so the Engine can read completed exports between turns.
#[derive(Default)]
pub struct Captures {
    worker: Option<Worker>,
    pending: bool,
    available: HashMap<PathBuf, PathBuf>,
}

impl Captures {
    pub fn pending(&self) -> bool {
        self.pending
    }

    /// A finite discovery batch runs in order before another batch can start.
    /// Each unit is one row, replayed line or 64 KiB prefix-verification block.
    /// A SQLite call, individual record and final publication can exceed it.
    pub fn turn(&mut self, inputs: Inputs, limit: usize) -> Result<CaptureTurn, ExportError> {
        if !self.pending && inputs.is_empty() {
            return Ok(CaptureTurn::default());
        }
        if self.worker.is_none() {
            self.worker = Some(Worker::start()?);
        }
        let worker = self.worker.as_ref().expect("started capture worker");
        let command = if self.pending {
            Command::Continue(limit.max(1))
        } else {
            Command::Start(inputs, limit.max(1))
        };
        let reply = worker
            .tx
            .as_ref()
            .expect("open capture permits")
            .send(command)
            .map_err(|_| stopped())
            .and_then(|()| worker.rx.recv().map_err(|_| stopped()));
        let reply = match reply {
            Ok(reply) => reply,
            Err(e) => {
                self.worker.take();
                self.pending = false;
                self.available.clear();
                return Err(e);
            }
        };
        let updates = match reply {
            Reply::Paused(updates) => {
                self.pending = true;
                updates
            }
            Reply::Finished(updates) => {
                self.pending = false;
                updates
            }
        };
        for (db, out) in updates.paths {
            if let Some(out) = out {
                self.available.insert(db, out);
            } else {
                self.available.remove(&db);
            }
        }
        Ok(CaptureTurn {
            pending: self.pending,
            errors: updates.errors,
        })
    }

    /// An output accepted by the worker, which is now paused or idle.
    pub fn paths(&self) -> impl Iterator<Item = &Path> {
        self.available.values().map(PathBuf::as_path)
    }
}

fn stopped() -> ExportError {
    ExportError::Invalid("SQLite capture worker stopped".into())
}

struct Worker {
    tx: Option<SyncSender<Command>>,
    rx: Receiver<Reply>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn start() -> Result<Self, ExportError> {
        let (tx, commands) = sync_channel(1);
        let (replies, rx) = sync_channel(1);
        let thread = std::thread::Builder::new()
            .name("polylogue-capture".into())
            .spawn(move || run(commands, replies))?;
        Ok(Self {
            tx: Some(tx),
            rx,
            thread: Some(thread),
        })
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct Budget<'a> {
    commands: &'a Receiver<Command>,
    replies: &'a SyncSender<Reply>,
    limit: usize,
    used: usize,
    updates: Updates,
    cancelled: bool,
}

impl Budget<'_> {
    fn checkpoint(&mut self) -> Result<(), ExportError> {
        self.used += 1;
        if self.used < self.limit {
            return Ok(());
        }
        self.replies
            .send(Reply::Paused(std::mem::take(&mut self.updates)))
            .map_err(|_| stopped())?;
        match self.commands.recv() {
            Ok(Command::Continue(limit)) => {
                self.limit = limit;
                self.used = 0;
                Ok(())
            }
            _ => {
                self.cancelled = true;
                Err(stopped())
            }
        }
    }
}

fn run(commands: Receiver<Command>, replies: SyncSender<Reply>) {
    let mut exporter = Exporter::default();
    while let Ok(Command::Start(inputs, limit)) = commands.recv() {
        let mut budget = Budget {
            commands: &commands,
            replies: &replies,
            limit,
            used: 0,
            updates: Updates::default(),
            cancelled: false,
        };
        for (db, out) in inputs {
            // Revalidation may itself yield. Do not let an earlier acceptance
            // authorize physically changed bytes while their hash is checked.
            budget.updates.paths.push((db.clone(), None));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                exporter.export_with(&db, &out, &mut || budget.checkpoint())
            }));
            if budget.cancelled {
                return;
            }
            let result = match result {
                Ok(result) => result,
                Err(_) => {
                    exporter.dbs.remove(&db);
                    Err(ExportError::Invalid("SQLite capture panicked".into()))
                }
            };
            match result {
                Ok(_) => budget.updates.paths.push((db, Some(out))),
                Err(e) => {
                    budget
                        .updates
                        .errors
                        .push(format!("export {}: {e}", db.display()));
                    budget.updates.paths.push((db, None));
                }
            }
            if budget.checkpoint().is_err() {
                return;
            }
        }
        if replies.send(Reply::Finished(budget.updates)).is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests;
