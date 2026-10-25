//! A stand-in for `sinexd`'s ingress over a real NATS server. It records
//! frames by name, answers `have` from that record and a resent frame from
//! its record, reads material bytes and indirect frames from the spool as
//! Sinex does, and can lose chosen frames or their acknowledgments, telling
//! the test each time it does.
use crate::{Config, Mode, Sinex};
use async_nats::jetstream;
use futures_util::StreamExt;
use sinex_model::MaterialId;
use sinex_protocol::{Acknowledgment, Have, Ingress, Subjects};
use sinex_transport::spool;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

const PRODUCER: &str = "polylogue";
const SUBJECTS: &str = "stand-in";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Failure {
    /// The frame never reaches Sinex: nothing is recorded or acknowledged.
    Outage,
    /// Sinex records the frame, but its acknowledgment is lost.
    LostAcknowledgment,
}

#[derive(Default)]
pub(crate) struct Record {
    /// Recorded frame names, in the order Sinex recorded them.
    pub order: Vec<String>,
    /// Each recorded frame, as Sinex read it, with its acknowledgment.
    pub frames: HashMap<String, (Ingress, Acknowledgment)>,
    /// Names of the frames that arrived as indirect frames.
    pub indirect: Vec<String>,
    /// How many frames arrived, resent ones included.
    pub received: usize,
    /// The failure that hits the frame with this arrival number (from 1).
    pub failures: HashMap<usize, Failure>,
    /// Woken each time a failure hits a frame.
    pub failure_hit: Arc<Notify>,
}

impl Record {
    /// Where a frame stands in the recording order.
    pub fn position(&self, name: &str) -> Option<usize> {
        self.order.iter().position(|recorded| recorded == name)
    }
}

pub(crate) struct StandIn {
    pub record: Arc<Mutex<Record>>,
    pub root: PathBuf,
    nats: devkit::nats::Server,
    _dir: tempfile::TempDir,
}

impl StandIn {
    pub async fn start() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("sinex-store");
        let nats = devkit::nats::Server::start().unwrap();
        let subjects = Subjects::new(SUBJECTS);
        let client = async_nats::connect(nats.url()).await.unwrap();
        jetstream::new(client.clone())
            .create_stream(jetstream::stream::Config {
                name: subjects.ingress_stream(),
                subjects: vec![subjects.ingress(PRODUCER)],
                ..Default::default()
            })
            .await
            .unwrap();
        let mut ingress = client.subscribe(subjects.ingress(PRODUCER)).await.unwrap();
        let mut have = client.subscribe(subjects.have(PRODUCER)).await.unwrap();
        // A flush only writes the subscriptions out. The server handles one
        // connection's messages in order, so once a message published after
        // them comes back, it has registered them, and a producer on another
        // connection cannot reach Sinex before Sinex listens.
        let echo_subject = client.new_inbox();
        let mut echo = client.subscribe(echo_subject.clone()).await.unwrap();
        client.publish(echo_subject, "".into()).await.unwrap();
        echo.next().await.unwrap();
        drop(echo);
        let record = Arc::new(Mutex::new(Record::default()));
        let state = record.clone();
        let store = root.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    Some(message) = ingress.next() => {
                        let answer = receive(&state, &store, &message.payload);
                        if let Some(acknowledgment) = answer {
                            client
                                .publish(
                                    subjects.ack(PRODUCER),
                                    serde_json::to_vec(&acknowledgment).unwrap().into(),
                                )
                                .await
                                .unwrap();
                        }
                    }
                    Some(message) = have.next() => {
                        let request: Have = serde_json::from_slice(&message.payload).unwrap();
                        let held: Vec<Acknowledgment> = {
                            let record = state.lock().unwrap();
                            request
                                .frames
                                .iter()
                                .filter_map(|name| record.frames.get(name).map(|(_, a)| a.clone()))
                                .collect()
                        };
                        client
                            .publish(
                                message.reply.unwrap(),
                                serde_json::to_vec(&held).unwrap().into(),
                            )
                            .await
                            .unwrap();
                    }
                    else => return,
                }
            }
        });
        Self {
            record,
            root,
            nats,
            _dir: dir,
        }
    }

    pub fn config(&self) -> Config {
        Config {
            mode: Mode::Primary,
            nats: self.nats.url().into(),
            subjects: SUBJECTS.into(),
            producer: PRODUCER.into(),
            store: self.root.clone(),
            archive: None,
        }
    }

    pub async fn connect(&self) -> Sinex {
        Sinex::connect(&self.config()).await.unwrap()
    }

    pub fn fail(&self, at: usize, failure: Failure) {
        self.record.lock().unwrap().failures.insert(at, failure);
    }

    /// Returns once a failure has hit a frame since the last return.
    pub async fn failure_hit(&self) {
        let hit = Arc::clone(&self.record.lock().unwrap().failure_hit);
        hit.notified().await;
    }

    /// Whether any frame's spool directory is left.
    pub fn spool_is_empty(&self) -> bool {
        std::fs::read_dir(self.root.join("incoming/producers"))
            .map(|mut entries| entries.next().is_none())
            .unwrap_or(true)
    }
}

/// Handles one frame and returns the acknowledgment to publish, if any.
fn receive(state: &Mutex<Record>, root: &Path, payload: &[u8]) -> Option<Acknowledgment> {
    let frame: Ingress = serde_json::from_slice(payload).unwrap();
    let name = frame.frame().to_owned();
    let mut record = state.lock().unwrap();
    record.received += 1;
    let failure = record.failures.get(&record.received).copied();
    if failure.is_some() {
        record.failure_hit.notify_one();
    }
    if failure == Some(Failure::Outage) {
        return None;
    }
    if let Some((_, acknowledgment)) = record.frames.get(&name) {
        let acknowledgment = acknowledgment.clone();
        spool::discard(root, PRODUCER, &name).unwrap();
        return (failure.is_none()).then_some(acknowledgment);
    }
    let read = match frame {
        Ingress::Indirect { blob, length, .. } => {
            record.indirect.push(name.clone());
            spool::snapshot(root, PRODUCER, &name, &blob, length)
                .and_then(|file| serde_json::from_reader(file).map_err(|e| e.to_string()))
        }
        frame => Ok(frame),
    };
    let checked = read.and_then(|frame| check(&record, root, &name, &frame).map(|m| (frame, m)));
    let (frame, material, refusal) = match checked {
        Ok((frame, material)) => (frame, material, None),
        Err(reason) => (
            Ingress::Events {
                frame: name.clone(),
                events: vec![],
                failures: vec![],
            },
            None,
            Some(reason),
        ),
    };
    let acknowledgment = Acknowledgment {
        frame: name.clone(),
        material,
        admitted: vec![],
        outcome: serde_json::json!({}),
        refusal,
        commit_seq: Some(record.order.len() as i64 + 1),
    };
    record.order.push(name.clone());
    record
        .frames
        .insert(name.clone(), (frame, acknowledgment.clone()));
    spool::discard(root, PRODUCER, &name).unwrap();
    (failure.is_none()).then_some(acknowledgment)
}

/// What Sinex checks before it records a frame: a material's bytes are in
/// its frame's spool directory, and events cite registered materials.
fn check(
    record: &Record,
    root: &Path,
    name: &str,
    frame: &Ingress,
) -> Result<Option<MaterialId>, String> {
    match frame {
        Ingress::Material { material, .. } => {
            spool::snapshot(root, PRODUCER, name, &material.blob, material.length)?;
            Ok(Some(material.id))
        }
        Ingress::Events { events, .. } => {
            for event in events {
                if let Some(location) = &event.location
                    && !record
                        .frames
                        .values()
                        .any(|(_, a)| a.material == Some(location.material))
                {
                    return Err(format!("{name} cites an unregistered material"));
                }
            }
            Ok(None)
        }
        _ => Err("an unexpected frame".into()),
    }
}
