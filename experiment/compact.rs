//! One immutable current body and source claim, with explicit older observations.
use super::{Input, MessageRow, Result, Work, add_terms};
use polylogue_model::Message;
use rusqlite::{Connection, params};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

struct Observation {
    input: Input,
    row: MessageRow,
    file: i64,
}

struct Entry {
    current: Observation,
    other: Vec<Observation>,
}

#[derive(Default)]
pub struct Builder {
    entries: BTreeMap<(String, String), Entry>,
    observations: u64,
    replacements: u64,
    controls: Vec<(String, String, Message, Message)>,
}

impl Builder {
    pub fn observe(&mut self, input: Input, row: MessageRow, file: i64) {
        self.observations += 1;
        let key = (input.session.clone(), row.message.id.to_string());
        let control = (!input.many || input.control.as_deref() == Some(key.1.as_str()))
            .then(|| (input.name.clone(), key.1.clone(), row.message.clone()));
        let incoming = Observation { input, row, file };
        let selected = match self.entries.entry(key) {
            std::collections::btree_map::Entry::Vacant(v) => {
                let entry = v.insert(Entry {
                    current: incoming,
                    other: Vec::new(),
                });
                control.as_ref().map(|_| entry.current.row.message.clone())
            }
            std::collections::btree_map::Entry::Occupied(mut v) => {
                let entry = v.get_mut();
                let old = &entry.current.input;
                let wins = incoming.input.priority > old.priority
                    || incoming.input.priority == old.priority && incoming.input.name < old.name;
                let previous = if wins {
                    self.replacements += 1;
                    std::mem::replace(&mut entry.current, incoming)
                } else {
                    incoming
                };
                entry.other.push(previous);
                control.as_ref().map(|_| entry.current.row.message.clone())
            }
        };
        if let Some((input, id, observed)) = control {
            self.controls.push((input, id, observed, selected.unwrap()));
        }
    }

    pub fn store(self, db: &Connection, dir: &Path, work: &mut Work) -> Result<Value> {
        let mut payload = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(dir.join("bodies.zst"))?;
        let mut position = 0_u64;
        let mut older_rows = 0_u64;
        let mut patch_bytes = 0_u64;
        let mut claims = 0_u64;

        let current_rows = self.entries.len() as u64;
        for (_, entry) in self.entries {
            let current = entry.current;
            let id = current.row.message.id.to_string();
            let hash = current.row.message.content_hash()?;
            let json = serde_json::to_vec(&current.row.message)?;
            let packed = zstd::stream::encode_all(json.as_slice(), 1)?;
            let offset = position;
            payload.write_all(&packed)?;
            position += packed.len() as u64;
            db.execute("INSERT INTO message(id,session,parent,file,start,end,priority,hash,body_offset,body_bytes)
                VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)", params![id,current.input.session,
                current.row.message.parent.as_ref().map(ToString::to_string),current.file,current.row.offset,
                current.row.end,current.input.priority,hash.0.as_slice(),offset,packed.len() as u64])?;
            let message = db.last_insert_rowid();
            work.model_attempts += 1;
            work.model_rows += 1;
            work.stored_model_bytes += packed.len() as u64;
            add_terms(db, message, &current.row.message, "compact", work)?;
            let mut old_hashes: BTreeMap<Vec<u8>, i64> = BTreeMap::new();
            for old in entry.other {
                let old_hash = if old.row.message == current.row.message {
                    hash.clone()
                } else {
                    old.row.message.content_hash()?
                };
                let older = if old_hash == hash {
                    None
                } else if let Some(pk) = old_hashes.get(old_hash.0.as_slice()) {
                    Some(*pk)
                } else {
                    let old_json = serde_json::to_vec(&old.row.message)?;
                    let prefix = json
                        .iter()
                        .zip(&old_json)
                        .take_while(|(a, b)| a == b)
                        .count();
                    let suffix = json[prefix..]
                        .iter()
                        .rev()
                        .zip(old_json[prefix..].iter().rev())
                        .take_while(|(a, b)| a == b)
                        .count();
                    let patch = &old_json[prefix..old_json.len() - suffix];
                    // These coordinates bind the patch to this immutable generation's base,
                    // independently of any later selected message or publication pointer.
                    db.execute("INSERT INTO older(message,hash,base_hash,base_offset,base_bytes,prefix,suffix,patch)
                        VALUES(?1,?2,?3,?4,?5,?6,?7,?8)", params![message,old_hash.0.as_slice(),hash.0.as_slice(),
                        offset,packed.len() as u64,prefix as u64,suffix as u64,patch])?;
                    let pk = db.last_insert_rowid();
                    old_hashes.insert(old_hash.0.to_vec(), pk);
                    older_rows += 1;
                    patch_bytes += patch.len() as u64;
                    work.model_attempts += 1;
                    work.model_rows += 1;
                    Some(pk)
                };
                db.execute("INSERT INTO extra_claim(message,older,file,start,end,priority) VALUES(?1,?2,?3,?4,?5,?6)",
                    params![message,older,old.file,old.row.offset,old.row.end,old.input.priority])?;
                claims += 1;
                work.claims += 1;
            }
        }
        db.execute_batch("CREATE INDEX by_session ON message(session,start,id)")?;
        let mut transitions = Vec::new();
        for (input, id, observed, selected) in self.controls {
            transitions.push(json!({"input":input,"message":id,
                "observed_version":super::hex(&observed.content_hash()?.0),
                "selected_version":super::hex(&selected.content_hash()?.0)}));
        }
        let sync_started = std::time::Instant::now();
        payload.sync_all()?;
        let sync_seconds = sync_started.elapsed().as_secs_f64();
        use std::os::unix::fs::MetadataExt;
        let allocated = payload.metadata()?.blocks() * 512;
        Ok(json!({"current_rows":current_rows,"older_rows":older_rows,
            "logical_versions":current_rows+older_rows,"logical_observations":self.observations,
            "implicit_current_claims":current_rows,"explicit_extra_claims":claims,
            "in_memory_replacements":self.replacements,"body_file_bytes":position,
            "allocated_body_bytes":allocated,"body_sync_seconds":sync_seconds,
            "older_patch_bytes":patch_bytes,"selection_transitions":transitions}))
    }
}

fn body(dir: &Path, offset: u64, length: u64) -> Result<Vec<u8>> {
    let mut file = File::open(dir.join("bodies.zst"))?;
    file.seek(SeekFrom::Start(offset))?;
    let mut packed = vec![0; usize::try_from(length)?];
    file.read_exact(&mut packed)?;
    Ok(zstd::stream::decode_all(packed.as_slice())?)
}

type Located = (
    String,
    u64,
    u64,
    Vec<u8>,
    u64,
    u64,
    Option<Vec<u8>>,
    Option<Vec<u8>>,
);

pub fn lookup(
    db: &Connection,
    dir: &Path,
    session: &str,
    id: &str,
    requested: Option<&str>,
) -> Result<Located> {
    let current: Vec<u8> = db.query_row(
        "SELECT hash FROM message WHERE session=?1 AND id=?2",
        params![session, id],
        |r| r.get(0),
    )?;
    if requested
        .map(super::unhex)
        .transpose()?
        .as_ref()
        .is_none_or(|hash| hash == &current)
    {
        let (name, base, length, file_hash, start, end, offset, size): (
            String,
            u64,
            u64,
            Vec<u8>,
            u64,
            u64,
            u64,
            u64,
        ) = db.query_row(
            "SELECT f.name,f.offset,f.bytes,f.hash,m.start,m.end,m.body_offset,m.body_bytes
                FROM message m JOIN file f ON f.id=m.file WHERE m.session=?1 AND m.id=?2",
            params![session, id],
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
        )?;
        return Ok((
            name,
            base,
            length,
            file_hash,
            start,
            end,
            Some(body(dir, offset, size)?),
            Some(current),
        ));
    }
    let requested = super::unhex(requested.unwrap())?;
    let (name,base,length,file_hash,start,end,base_hash,offset,size,prefix,suffix,patch):
        (String,u64,u64,Vec<u8>,u64,u64,Vec<u8>,u64,u64,usize,usize,Vec<u8>) =
        db.query_row("SELECT f.name,f.offset,f.bytes,f.hash,c.start,c.end,o.base_hash,o.base_offset,o.base_bytes,o.prefix,o.suffix,o.patch
            FROM older o JOIN message m ON m.pk=o.message JOIN extra_claim c ON c.older=o.pk JOIN file f ON f.id=c.file
            WHERE m.session=?1 AND m.id=?2 AND o.hash=?3 ORDER BY c.priority DESC,f.name ASC LIMIT 1",
            params![session,id,requested], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,
                r.get(6)?,r.get(7)?,r.get(8)?,r.get(9)?,r.get(10)?,r.get(11)?)))?;
    let base_body = body(dir, offset, size)?;
    let base_model: Message = serde_json::from_slice(&base_body)?;
    assert_eq!(base_model.content_hash()?.0.as_slice(), base_hash);
    assert!(prefix + suffix <= base_body.len());
    let mut decoded = Vec::with_capacity(prefix + patch.len() + suffix);
    decoded.extend_from_slice(&base_body[..prefix]);
    decoded.extend_from_slice(&patch);
    decoded.extend_from_slice(&base_body[base_body.len() - suffix..]);
    Ok((
        name,
        base,
        length,
        file_hash,
        start,
        end,
        Some(decoded),
        Some(requested),
    ))
}
