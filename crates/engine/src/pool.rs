//! The parse pool: N workers admitted by bytes in flight, feeding the one
//! writer through a bounded channel (architecture 2, 5.3). Live appends go
//! first.

use polylogue_sources::StepParser;
use polylogue_store::Batch;
use std::sync::mpsc::SyncSender;

pub enum Priority {
    Live,
    Bulk,
}

pub struct Job {
    pub input: i64,
    pub parser: Box<dyn StepParser>,
    pub carry: Option<serde_json::Value>,
    pub bytes: Vec<u8>,
    pub base: u64,
    pub priority: Priority,
}

pub struct Pool {
    pub workers: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    pub fn start(_workers: usize, _bytes_in_flight: u64, _to_writer: SyncSender<Batch>) -> Self {
        // Architecture 2: parse pool with bytes-in-flight admission.
        todo!()
    }

    pub fn submit(&self, _job: Job) {
        // Architecture 5.3: live before bulk.
        todo!()
    }
}
