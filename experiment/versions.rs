// Synthetic extension of the partitioned SQLite experiment, not a product format.
use polylogue_model::{Block, Message, MessageRow, parser_context::Handle};
use polylogue_sources::claude_code;
use rusqlite::{Connection, OpenFlags, params};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
mod compact;

const PARTS: usize = 2;
const SCHEMA_LATEST: &str = include_str!("latest.sql");
const SCHEMA_VERSIONED: &str = include_str!("versioned.sql");
const CHOOSE: &str = include_str!("choose.sql");

#[derive(Clone)]
struct Input {
    name: String,
    path: PathBuf,
    session: String,
    id: String,
    start: u64,
    end: u64,
    priority: i64,
    texts: Vec<String>,
    many: bool,
    control: Option<String>,
}

fn route(session: &str) -> usize {
    let hash = session.bytes().fold(14695981039346656037u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(1099511628211)
    });
    hash as usize % PARTS
}

fn inputs(path: &Path, reverse: bool) -> Result<Vec<Input>> {
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    let mut inputs = Vec::new();
    for row in value["observations"]
        .as_array()
        .ok_or("missing observations")?
    {
        let string = |name: &str| -> Result<String> {
            Ok(row[name].as_str().ok_or("missing input field")?.to_owned())
        };
        inputs.push(Input {
            name: string("input")?,
            path: path
                .parent()
                .ok_or("manifest directory")?
                .join(string("file")?),
            session: string("session")?,
            id: row["message"].as_str().unwrap_or_default().to_owned(),
            start: row["start"].as_u64().unwrap_or_default(),
            end: row["end"].as_u64().unwrap_or_default(),
            priority: row["priority"].as_i64().ok_or("missing priority")?,
            texts: row["texts"]
                .as_array()
                .map(Vec::as_slice)
                .unwrap_or_default()
                .iter()
                .map(|v| v.as_str().ok_or("non-string text").map(str::to_owned))
                .collect::<std::result::Result<_, _>>()?,
            many: row["many"].as_bool().unwrap_or(false),
            control: row["control"].as_str().map(str::to_owned),
        });
    }
    if reverse {
        inputs.reverse();
    }
    Ok(inputs)
}

fn texts(message: &Message) -> Vec<String> {
    message
        .blocks
        .iter()
        .filter_map(Block::text)
        .map(str::to_owned)
        .collect()
}

fn parse(input: &Input, raw: &[u8]) -> Result<(Vec<MessageRow>, u64)> {
    let parser = claude_code::open(&input.path).ok_or("not Claude JSONL")?;
    let context = Handle::default();
    let mut latest = BTreeMap::new();
    let mut state = None;
    let mut at = 0;
    let mut records = 0;
    while at < raw.len() {
        let aim = (at + (1 << 20)).min(raw.len());
        let end = if aim == raw.len() {
            aim
        } else {
            raw[aim..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(raw.len(), |n| aim + n + 1)
        };
        let (steps, next) = parser.steps(
            state.as_ref(),
            &raw[at..end],
            at as u64,
            end == raw.len(),
            &context,
        )?;
        let mut consumed = 0;
        for step in steps {
            consumed += step.consumed;
            records += step.counts.records;
            assert_eq!(step.counts.unreadable.values().sum::<u64>(), 0);
            for row in step.output.messages {
                assert_eq!(
                    row.message.session.to_string(),
                    input.session,
                    "input crosses its routed session"
                );
                latest.insert(row.message.id.clone(), row);
            }
        }
        assert_eq!(consumed, (end - at) as u64);
        state = Some(next);
        at = end;
    }
    assert!(!latest.is_empty());
    if !input.many {
        assert_eq!(latest.len(), 1);
        let row = latest.values().next().unwrap();
        assert_eq!(row.message.id.to_string(), input.id);
        assert_eq!((row.offset, row.end), (input.start, input.end));
        assert_eq!(texts(&row.message), input.texts);
    }
    Ok((latest.into_values().collect(), records))
}

fn sync(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn unhex(value: &str) -> Result<Vec<u8>> {
    if !value.len().is_multiple_of(2) {
        return Err("odd hex".into());
    }
    (0..value.len())
        .step_by(2)
        .map(|n| Ok(u8::from_str_radix(&value[n..n + 2], 16)?))
        .collect()
}

#[derive(Default)]
struct Work {
    files: u64,
    model_attempts: u64,
    model_rows: u64,
    claims: u64,
    selection_attempts: u64,
    selections_changed: u64,
    hits: u64,
    raw_bytes: u64,
    stored_model_bytes: u64,
    parsed_messages: u64,
    records: u64,
}

fn add_terms(
    db: &Connection,
    owner: i64,
    message: &Message,
    mode: &str,
    work: &mut Work,
) -> Result<()> {
    for block in &message.blocks {
        let (channel, value) = match block {
            Block::ToolCall { written_text, .. } => (1, written_text.as_deref()),
            Block::ToolResult { output, .. } => (1, output.as_deref()),
            _ => (0, block.text()),
        };
        if let Some(value) = value.filter(|value| !value.is_empty()) {
            let column = if mode == "versioned" {
                "version"
            } else {
                "message"
            };
            db.execute(
                &format!("INSERT INTO hit({column},channel) VALUES(?1,?2)"),
                params![owner, channel],
            )?;
            db.execute(
                "INSERT INTO terms(rowid,body) VALUES(?1,?2)",
                params![db.last_insert_rowid(), value],
            )?;
            work.hits += 1;
        }
    }
    Ok(())
}

fn partition(
    root: &Path,
    part: usize,
    mode: &str,
    inputs: Vec<Input>,
    ready: Option<Arc<Barrier>>,
) -> Result<Value> {
    let dir = root.join(part.to_string());
    fs::create_dir(&dir)?;
    let db_path = dir.join("index.db");
    let db = Connection::open(&db_path)?; // This is the final partition, never an in-memory model.
    db.execute_batch(
        "PRAGMA page_size=8192; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
        PRAGMA wal_autocheckpoint=0; PRAGMA cache_size=-1024; PRAGMA foreign_keys=ON;
        PRAGMA temp_store=MEMORY;",
    )?;
    db.execute_batch(if mode == "versioned" {
        SCHEMA_VERSIONED
    } else if mode == "compact" {
        include_str!("compact.sql")
    } else {
        SCHEMA_LATEST
    })?;
    db.execute_batch("BEGIN IMMEDIATE")?;
    // Both final databases can hold write transactions concurrently.
    if let Some(ready) = ready {
        ready.wait();
    }
    let mut raw_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(dir.join("originals.raw"))?;
    let mut work = Work::default();
    let mut latest: BTreeMap<(String, String), (Input, MessageRow, i64)> = BTreeMap::new();
    let mut transitions = Vec::new();
    let mut compact_builder = compact::Builder::default();
    let mut compact_stats = Value::Null;
    for input in inputs {
        let raw = fs::read(&input.path)?;
        let offset = work.raw_bytes;
        raw_file.write_all(&raw)?;
        let digest = blake3::hash(&raw);
        db.execute(
            "INSERT INTO file(name,offset,compressed,bytes,hash) VALUES(?1,?2,?3,?3,?4)",
            params![
                input.name,
                offset,
                raw.len() as u64,
                digest.as_bytes().as_slice()
            ],
        )?;
        let file = db.last_insert_rowid();
        work.files += 1;
        work.raw_bytes += raw.len() as u64;
        let (rows, records) = parse(&input, &raw)?;
        work.records += records;
        work.parsed_messages += rows.len() as u64;
        for row in rows {
            let id = row.message.id.to_string();
            if mode == "versioned" {
                let hash = row.message.content_hash()?;
                let packed =
                    zstd::stream::encode_all(serde_json::to_vec(&row.message)?.as_slice(), 1)?;
                work.model_attempts += 1;
                let inserted = db.execute(
                "INSERT OR IGNORE INTO version(hash,session,id,parent,body) VALUES(?1,?2,?3,?4,?5)",
                params![
                    hash.0.as_slice(),
                    input.session,
                    id,
                    row.message.parent.as_ref().map(ToString::to_string),
                    packed
                ],
            )?;
                work.model_rows += inserted as u64;
                if inserted != 0 {
                    work.stored_model_bytes += packed.len() as u64;
                }
                let version: i64 = db.query_row(
                    "SELECT pk FROM version WHERE hash=?1",
                    [hash.0.as_slice()],
                    |r| r.get(0),
                )?;
                db.execute(
                "INSERT INTO observation(file,version,start,end,priority) VALUES(?1,?2,?3,?4,?5)",
                params![file, version, row.offset, row.end, input.priority],
            )?;
                work.claims += 1;
                work.selection_attempts += 1;
                work.selections_changed += db.execute(CHOOSE, params![input.session, id])? as u64;
                let chosen: Vec<u8> = db.query_row(
                    "SELECT v.hash FROM selected s JOIN observation o ON o.pk=s.observation
                JOIN version v ON v.pk=o.version WHERE s.session=?1 AND s.id=?2",
                    params![input.session, id],
                    |r| r.get(0),
                )?;
                if !input.many || input.control.as_deref() == Some(id.as_str()) {
                    transitions.push(json!({"input":input.name,"message":id,
                    "observed_version":hex(&hash.0),"selected_version":hex(&chosen)}));
                }
            } else if mode == "compact" {
                compact_builder.observe(input.clone(), row, file);
            } else {
                // The original prototype's map was per file. Lift only that map
                // to the partition so its latest-only mode accepts this overlap.
                let key = (input.session.clone(), id.clone());
                let wins = latest.get(&key).is_none_or(|(old, _, _)| {
                    input.priority > old.priority
                        || input.priority == old.priority && input.name < old.name
                });
                if wins {
                    latest.insert(key, (input.clone(), row, file));
                }
            }
        }
    }
    if mode == "versioned" {
        let selected = {
            let mut query = db.prepare(
                "SELECT v.pk,v.body FROM selected s JOIN observation o ON o.pk=s.observation
                JOIN version v ON v.pk=o.version ORDER BY s.session,s.id",
            )?;
            let rows = query
                .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            rows
        };
        for (version, packed) in selected {
            let message: Message =
                serde_json::from_slice(&zstd::stream::decode_all(packed.as_slice())?)?;
            add_terms(&db, version, &message, mode, &mut work)?;
        }
    } else if mode == "compact" {
        compact_stats = compact_builder.store(&db, &dir, &mut work)?;
        transitions = serde_json::from_value(compact_stats["selection_transitions"].clone())?;
    } else {
        for (_, (input, row, file)) in latest {
            work.model_attempts += 1;
            work.model_rows += db.execute(
                "INSERT INTO message(id,session,parent,file,start,end) VALUES(?1,?2,?3,?4,?5,?6)",
                params![
                    row.message.id.to_string(),
                    input.session,
                    row.message.parent.as_ref().map(ToString::to_string),
                    file,
                    row.offset,
                    row.end
                ],
            )? as u64;
            add_terms(&db, db.last_insert_rowid(), &row.message, mode, &mut work)?;
        }
        db.execute_batch("CREATE INDEX by_session ON message(session,start,id)")?;
    }
    assert_eq!(
        db.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r
            .get::<_, i64>(
            0
        ))?,
        0
    );
    raw_file.sync_all()?; // Custody before model commit.
    db.execute_batch("COMMIT")?;
    let wal_before_checkpoint = fs::metadata(db_path.with_extension("db-wal"))?.len();
    let checkpoint: (i64, i64, i64) = db.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| {
        Ok((r.get(0)?, r.get(1)?, r.get(2)?))
    })?;
    assert_eq!(checkpoint, (0, 0, 0));
    assert_eq!(
        db.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))?,
        "ok"
    );
    let pages: i64 = db.query_row("PRAGMA page_count", [], |r| r.get(0))?;
    let mut cache_writes = 0;
    let mut high = 0;
    assert_eq!(
        unsafe {
            rusqlite::ffi::sqlite3_db_status(
                db.handle(),
                rusqlite::ffi::SQLITE_DBSTATUS_CACHE_WRITE,
                &mut cache_writes,
                &mut high,
                0,
            )
        },
        0
    );
    let changes = unsafe { rusqlite::ffi::sqlite3_total_changes64(db.handle()) };
    let selected_count: i64 = db.query_row(
        if mode == "versioned" {
            "SELECT count(*) FROM selected"
        } else {
            "SELECT count(*) FROM message"
        },
        [],
        |r| r.get(0),
    )?;
    db.close().map_err(|(_, error)| error)?;
    sync(&db_path)?;
    sync(&dir)?;
    use std::os::unix::fs::MetadataExt;
    let wal = db_path.with_extension("db-wal");
    assert_eq!(fs::metadata(&wal).map(|m| m.len()).unwrap_or(0), 0);
    Ok(
        json!({"part":part,"mode":mode,"files":work.files,"retained_bytes":work.raw_bytes,
        "records":work.records,"parsed_messages":work.parsed_messages,"selected_messages":selected_count,
        "model_insert_attempts":work.model_attempts,"model_rows_inserted":work.model_rows,
        "observation_rows":work.claims,"selection_attempts":work.selection_attempts,
        "selection_rows_changed":work.selections_changed,"hit_and_fts_rows_each":work.hits,
        "packed_model_bytes":work.stored_model_bytes,"sqlite_total_changes_including_fts":changes,
        "cache_page_writes":cache_writes,"page_count":pages,"page_size":8192,
        "index_bytes":fs::metadata(&db_path)?.len(),"allocated_index_bytes":fs::metadata(&db_path)?.blocks()*512,
        "raw_bytes":fs::metadata(dir.join("originals.raw"))?.len(),
        "allocated_raw_bytes":fs::metadata(dir.join("originals.raw"))?.blocks()*512,
        "wal_before_checkpoint":wal_before_checkpoint,"index_wal_bytes":0,
        "checkpoint":checkpoint,"selection_transitions":transitions,"compact":compact_stats}),
    )
}

fn build(
    manifest: &Path,
    root: &Path,
    generation: &str,
    mode: &str,
    reverse: bool,
    interrupt: bool,
) -> Result<()> {
    if !matches!(mode, "latest" | "versioned" | "compact") {
        return Err("unknown mode".into());
    }
    if generation.is_empty()
        || !generation
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    {
        return Err("invalid generation".into());
    }
    let path = root.join(generation);
    fs::create_dir(&path)?;
    let mut parts = vec![Vec::new(); PARTS];
    for input in inputs(manifest, reverse)? {
        parts[route(&input.session)].push(input);
    }
    assert!(
        parts.iter().all(|p| !p.is_empty()),
        "fixture must populate both routable partitions"
    );
    let reports = if interrupt {
        let first = partition(&path, 0, mode, parts.remove(0), None)?;
        println!(
            "{}",
            json!({"phase":"one-partition-committed","report":first})
        );
        std::io::stdout().flush()?;
        // The control parent kills this process at the real committed boundary.
        loop {
            std::thread::park();
        }
    } else {
        let barrier = Arc::new(Barrier::new(PARTS));
        let handles: Vec<_> = parts
            .into_iter()
            .enumerate()
            .map(|(part, inputs)| {
                let path = path.clone();
                let mode = mode.to_owned();
                let barrier = barrier.clone();
                std::thread::spawn(move || partition(&path, part, &mode, inputs, Some(barrier)))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().map_err(|_| "partition panicked")?)
            .collect::<Result<Vec<_>>>()?
    };
    fs::write(path.join("mode"), mode)?;
    sync(&path.join("mode"))?;
    sync(&path)?;
    let pointer = root.join(format!("CURRENT-{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&pointer)?;
    file.write_all(generation.as_bytes())?;
    file.sync_all()?;
    fs::rename(pointer, root.join("CURRENT"))?;
    sync(root)?;
    println!(
        "{}",
        json!({"generation":generation,"mode":mode,"parts":reports,
        "process_io":fs::read_to_string("/proc/self/io").ok(),
        "original_prototype_adjustment":"latest map lifted from per-file to partition scope; final SQLite file opened directly",
        "ranking":"current per-block hit membership only; ordinary distinct-text/global ranking is unimplemented"})
    );
    Ok(())
}

fn lookup(root: &Path, session: &str, id: &str, version: Option<&str>) -> Result<Value> {
    let generation = fs::read_to_string(root.join("CURRENT"))?;
    if generation.is_empty()
        || !generation
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    {
        return Err("invalid publication pointer".into());
    }
    let path = root.join(&generation).join(route(session).to_string());
    let mode = fs::read_to_string(root.join(&generation).join("mode"))?;
    let db = Connection::open_with_flags(path.join("index.db"), OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let (name, base, length, file_hash, start, end, body, hash): (
        String,
        u64,
        u64,
        Vec<u8>,
        u64,
        u64,
        Option<Vec<u8>>,
        Option<Vec<u8>>,
    ) = if mode == "compact" {
        compact::lookup(&db, &path, session, id, version)?
    } else if mode == "versioned" {
        if let Some(version) = version {
            db.query_row(
                "SELECT f.name,f.offset,f.bytes,f.hash,o.start,o.end,v.body,v.hash
                FROM version v JOIN observation o ON o.version=v.pk JOIN file f ON f.id=o.file
                WHERE v.session=?1 AND v.id=?2 AND v.hash=?3
                ORDER BY o.priority DESC,f.name ASC LIMIT 1",
                params![session, id, unhex(version)?],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                        r.get(7)?,
                    ))
                },
            )?
        } else {
            db.query_row("SELECT f.name,f.offset,f.bytes,f.hash,o.start,o.end,v.body,v.hash FROM selected s
                JOIN observation o ON o.pk=s.observation JOIN version v ON v.pk=o.version JOIN file f ON f.id=o.file
                WHERE s.session=?1 AND s.id=?2",params![session,id],|r|
                Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?)))?
        }
    } else {
        if version.is_some() {
            return Err("latest-only mode has no historical version lookup".into());
        }
        db.query_row("SELECT f.name,f.offset,f.bytes,f.hash,m.start,m.end,NULL,NULL FROM message m JOIN file f ON f.id=m.file
            WHERE m.session=?1 AND m.id=?2",params![session,id],|r|
            Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?)))?
    };
    let mut original = vec![0; usize::try_from(length)?];
    let mut raw = File::open(path.join("originals.raw"))?;
    raw.seek(SeekFrom::Start(base))?;
    raw.read_exact(&mut original)?;
    assert_eq!(blake3::hash(&original).as_bytes().as_slice(), file_hash);
    let model = body
        .map(|packed| -> Result<Value> {
            let decoded = if mode == "compact" {
                packed
            } else {
                zstd::stream::decode_all(packed.as_slice())?
            };
            let message: Message = serde_json::from_slice(&decoded)?;
            assert_eq!(message.session.to_string(), session);
            assert_eq!(message.id.to_string(), id);
            assert_eq!(message.content_hash()?.0.as_slice(), hash.as_ref().unwrap());
            Ok(serde_json::to_value(message)?)
        })
        .transpose()?;
    Ok(
        json!({"generation":generation,"partition":route(session),"mode":mode,"session":session,"message":id,
        "version":hash.map(|hash|hex(&hash)),"model":model,
        "citation":{"input":name,"start":start,"end":end,"source_bytes":length,
            "record":std::str::from_utf8(&original[usize::try_from(start)?..usize::try_from(end)?])?}}),
    )
}

fn search(root: &Path, term: &str) -> Result<Value> {
    let generation = fs::read_to_string(root.join("CURRENT"))?;
    if generation.is_empty()
        || !generation
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    {
        return Err("invalid publication pointer".into());
    }
    let mut hits = std::collections::BTreeSet::new();
    for part in 0..PARTS {
        let path = root.join(&generation);
        let mode = fs::read_to_string(path.join("mode"))?;
        let db = Connection::open_with_flags(
            path.join(part.to_string()).join("index.db"),
            OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        let query = if mode == "versioned" {
            "SELECT v.session,v.id,h.channel FROM terms JOIN hit h ON h.pk=terms.rowid JOIN version v ON v.pk=h.version WHERE terms MATCH ?1"
        } else {
            "SELECT m.session,m.id,h.channel FROM terms JOIN hit h ON h.pk=terms.rowid JOIN message m ON m.pk=h.message WHERE terms MATCH ?1"
        };
        let mut query = db.prepare(query)?;
        for row in query.query_map([term], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })? {
            hits.insert(row?);
        }
    }
    Ok(json!({"generation":generation,"hits":hits,"ranking_qualified":false}))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build") if args.len()==8 => build(Path::new(&args[2]),Path::new(&args[3]),&args[4],&args[5],args[6]=="reverse",args[7]=="interrupt"),
        Some("lookup") if (5..=6).contains(&args.len()) => {
            println!("{}",lookup(Path::new(&args[2]),&args[3],&args[4],args.get(5).map(String::as_str))?); Ok(())
        }
        Some("search") if args.len()==4 => { println!("{}",search(Path::new(&args[2]),&args[3])?); Ok(()) }
        _ => Err("build MANIFEST ROOT GENERATION latest|versioned forward|reverse complete|interrupt; lookup ROOT SESSION MESSAGE [VERSION]; search ROOT TERM".into()),
    }
}
