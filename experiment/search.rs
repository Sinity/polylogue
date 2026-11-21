//! One native, distinct-text search population for the published generation.
use super::{Message, Result, Work, hex, sync};
use polylogue_model::Block;
use rusqlite::{Connection, OpenFlags, params};
use serde_json::{Value, json};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex, mpsc};

const FEED_BYTES: usize = 4 << 20;
const SCHEMA: &str = "
CREATE TABLE text(id INTEGER PRIMARY KEY,hash BLOB NOT NULL UNIQUE);
CREATE VIRTUAL TABLE text_fts USING fts5(body,content='',contentless_delete=1,tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE output_fts USING fts5(body,content='',contentless_delete=1,tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE fts_vocab_text USING fts5vocab(text_fts,'row');
CREATE VIRTUAL TABLE fts_vocab_output USING fts5vocab(output_fts,'row');
INSERT INTO text_fts(text_fts,rank) VALUES('automerge',16);
INSERT INTO output_fts(output_fts,rank) VALUES('automerge',16);
INSERT INTO text_fts(text_fts,rank) VALUES('hashsize',67108864);
INSERT INTO output_fts(output_fts,rank) VALUES('hashsize',67108864);";

struct Text {
    id: i64,
    hash: [u8; 32],
    body: String,
    bit: i64,
    new: bool,
}

type Keys = Arc<Mutex<HashMap<[u8; 32], (i64, i64)>>>;

#[derive(Clone)]
pub struct Sender {
    keys: Keys,
    sender: mpsc::SyncSender<Vec<Text>>,
}

pub struct Build {
    sender: Sender,
    worker: std::thread::JoinHandle<Result<Value>>,
}

pub struct Feed {
    source: Sender,
    pending: Vec<Text>,
    bytes: usize,
}

impl Build {
    pub fn start(path: &Path) -> Result<Self> {
        let path = path.to_owned();
        // One bounded batch may be queued while the owner consumes another.
        // A single text larger than the window is sent alone, never truncated.
        let (sender, receiver) = mpsc::sync_channel::<Vec<Text>>(1);
        let (ready, initialized) = mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            let db = match Connection::open(&path) {
                Ok(db) => db,
                Err(error) => {
                    let _ = ready.send(Err(error.to_string()));
                    return Err(error.into());
                }
            };
            let setup = db.execute_batch(
                "PRAGMA page_size=8192; PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL; PRAGMA wal_autocheckpoint=0;
                PRAGMA cache_size=-65536; PRAGMA temp_store=MEMORY;",
            );
            let setup = setup
                .and_then(|()| db.execute_batch(SCHEMA))
                .and_then(|()| db.execute_batch("BEGIN IMMEDIATE"));
            if let Err(error) = setup {
                let _ = ready.send(Err(error.to_string()));
                return Err(error.into());
            }
            ready
                .send(Ok(()))
                .map_err(|_| "search initializer lost its owner")?;
            let mut texts = 0_u64;
            let mut postings = 0_u64;
            let mut decoded_bytes = 0_u64;
            for batch in receiver {
                for text in batch {
                    if text.new {
                        db.prepare_cached("INSERT INTO text VALUES(?1,?2)")?
                            .execute(params![text.id, text.hash.as_slice()])?;
                        texts += 1;
                    }
                    decoded_bytes += text.body.len() as u64;
                    let folded = if text.body.is_ascii() {
                        std::borrow::Cow::Borrowed(text.body.as_str())
                    } else {
                        std::borrow::Cow::Owned(polylogue_model::text::fold(&text.body))
                    };
                    let sql = if text.bit == 1 {
                        "INSERT INTO text_fts(rowid,body) VALUES(?1,?2)"
                    } else {
                        "INSERT INTO output_fts(rowid,body) VALUES(?1,?2)"
                    };
                    db.prepare_cached(sql)?
                        .execute(params![text.id, &*folded])?;
                    postings += 1;
                }
            }
            db.execute_batch("COMMIT")?;
            let wal = std::fs::metadata(path.with_extension("db-wal"))?.len();
            let checkpoint: (i64, i64, i64) =
                db.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                })?;
            assert_eq!(checkpoint, (0, 0, 0));
            let mut writes = 0;
            let mut high = 0;
            assert_eq!(
                unsafe {
                    rusqlite::ffi::sqlite3_db_status(
                        db.handle(),
                        rusqlite::ffi::SQLITE_DBSTATUS_CACHE_WRITE,
                        &mut writes,
                        &mut high,
                        0,
                    )
                },
                0
            );
            let changes = unsafe { rusqlite::ffi::sqlite3_total_changes64(db.handle()) };
            assert_eq!(
                db.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
                "ok"
            );
            db.close().map_err(|(_, error)| error)?;
            sync(&path)?;
            assert_eq!(
                std::fs::metadata(path.with_extension("db-wal"))
                    .map(|m| m.len())
                    .unwrap_or(0),
                0
            );
            Ok(
                json!({"distinct_texts":texts,"fts_rows":postings,"decoded_bytes":decoded_bytes,
                "cache_page_writes":writes,"sqlite_total_changes_including_fts":changes,
                "index_bytes":std::fs::metadata(path)?.len(),"wal_before_checkpoint":wal,
                "checkpoint":checkpoint,"index_wal_bytes":0,"feed_batch_bytes":FEED_BYTES,
                "feed_queued_batches":1,"oversize_texts":"sent alone; no truncation"}),
            )
        });
        initialized
            .recv()
            .map_err(|_| "search initializer terminated")?
            .map_err(|message| -> Box<dyn std::error::Error + Send + Sync> { message.into() })?;
        Ok(Self {
            sender: Sender {
                keys: Arc::default(),
                sender,
            },
            worker,
        })
    }

    pub fn sender(&self) -> Sender {
        self.sender.clone()
    }

    pub fn finish(self) -> Result<Value> {
        drop(self.sender);
        self.worker.join().map_err(|_| "search owner panicked")?
    }
}

impl Sender {
    pub fn feed(self) -> Feed {
        Feed {
            source: self,
            pending: Vec::new(),
            bytes: 0,
        }
    }
}

impl Feed {
    pub fn flush(&mut self) -> Result<()> {
        if !self.pending.is_empty() {
            self.source
                .sender
                .send(std::mem::take(&mut self.pending))
                .map_err(|_| "search owner terminated")?;
        }
        self.bytes = 0;
        Ok(())
    }

    pub fn message(
        &mut self,
        db: &Connection,
        owner: i64,
        message: &Message,
        mode: &str,
        work: &mut Work,
    ) -> Result<()> {
        for (idx, block) in message.blocks.iter().enumerate() {
            let (channel, value) = match block {
                Block::ToolCall { written_text, .. } => (1, written_text.as_deref()),
                Block::ToolResult { output, .. } => (1, output.as_deref()),
                _ => (0, block.text()),
            };
            let Some(body) = value.filter(|body| !body.is_empty()) else {
                continue;
            };
            let hash = *blake3::hash(body.as_bytes()).as_bytes();
            let bit = 1 << channel;
            let (id, new, send) = {
                let mut keys = self
                    .source
                    .keys
                    .lock()
                    .map_err(|_| "text identity owner poisoned")?;
                let next = i64::try_from(keys.len() + 1)?;
                let new = !keys.contains_key(&hash);
                let (id, bits) = keys.entry(hash).or_insert((next, 0));
                let send = *bits & bit == 0;
                *bits |= bit;
                (*id, new, send)
            };
            let column = if mode == "versioned" {
                "version"
            } else {
                "message"
            };
            db.prepare_cached(&format!(
                "INSERT INTO hit({column},channel,text_id,idx) VALUES(?1,?2,?3,?4)"
            ))?
            .execute(params![owner, channel, id, idx as i64])?;
            work.hits += 1;
            if send {
                if body.len() > FEED_BYTES.saturating_sub(self.bytes) {
                    self.flush()?;
                }
                self.pending.push(Text {
                    id,
                    hash,
                    body: body.to_owned(),
                    bit,
                    new,
                });
                if body.len() > FEED_BYTES {
                    self.flush()?;
                } else {
                    self.bytes += body.len();
                }
            }
        }
        Ok(())
    }
}

pub fn search(path: &Path, term: &str) -> Result<Value> {
    let db = Connection::open_with_flags(path.join("search.db"), OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mode = std::fs::read_to_string(path.join("mode"))?;
    let parts = (0..super::PARTS)
        .map(|part| {
            Connection::open_with_flags(
                path.join(part.to_string()).join("index.db"),
                OpenFlags::SQLITE_OPEN_READ_ONLY,
            )
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut hits = BTreeSet::new();
    let mut ranks = Vec::new();
    for (channel, table) in [(0, "text_fts"), (1, "output_fts")] {
        let mut select = db.prepare(&format!("SELECT t.id,t.hash,bm25({table}) FROM {table}
            JOIN text t ON t.id={table}.rowid WHERE {table} MATCH ?1 ORDER BY bm25({table}),hex(t.hash)"))?;
        let rows = select
            .query_map([term], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Vec<u8>>(1)?,
                    r.get::<_, f64>(2)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for (id, hash, score) in rows {
            ranks.push(json!({"channel":channel,"hash":hex(&hash),"score":score}));
            for part in &parts {
                let sql = if mode == "versioned" {
                    "SELECT v.session,v.id FROM hit h JOIN version v ON v.pk=h.version WHERE h.text_id=?1 AND h.channel=?2"
                } else {
                    "SELECT m.session,m.id FROM hit h JOIN message m ON m.pk=h.message WHERE h.text_id=?1 AND h.channel=?2"
                };
                let mut query = part.prepare_cached(sql)?;
                for row in query.query_map(params![id, channel], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, channel))
                })? {
                    hits.insert(row?);
                }
            }
        }
    }
    Ok(
        json!({"hits":hits,"ranked_texts":ranks,"ranking_qualified":"global distinct-text native bm25; production oracle required"}),
    )
}
