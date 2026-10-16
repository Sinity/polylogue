//! Immutable typed Session-owned operation publication and version-pinned reads.
//! This is the first bounded CommandExecution admission, not a general merger.
use crate::{Error, ReadHandle, Result, Writer, db, parser_state::Token};
use polylogue_model::tool_item::{CODEX_CONSTRUCTOR, Operation, ToolOrigin};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Citation {
    pub input: String,
    pub chain: i64,
    pub start: u64,
    pub end: u64,
    pub origin: ToolOrigin,
    pub version_hash: String,
    pub constructor: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Item {
    pub reference: String,
    pub version: i64,
    pub version_hash: String,
    pub operation: Operation,
    pub sources: Vec<Citation>,
    pub representations: Vec<Citation>,
    pub conflicting_representations: bool,
}
#[derive(Serialize)]
pub struct Population {
    pub items: Vec<Item>,
    pub tool_calls: u64,
    pub failed: u64,
    pub correspondence_complete: bool,
    pub retained_representations: usize,
    pub retained_rows: u64,
    pub decoded_rows: u64,
    pub unsupported_rows: u64,
    pub suppressed_wrappers: Vec<String>,
}
fn bad(s: impl Into<String>) -> Error {
    Error::Invariant(s.into())
}
fn payload(tx: &Transaction<'_>, path: &Path, codec: &str, bytes: &[u8]) -> Result<i64> {
    let hash = blake3::hash(bytes);
    if let Some(pk) = db(path, || {
        tx.query_row(
            "SELECT pk FROM body_value WHERE codec=?1 AND hash=?2 AND state='ready'",
            params![codec, hash.as_bytes().as_slice()],
            |r| r.get(0),
        )
        .optional()
    })? {
        return Ok(pk);
    }
    db(path, || {
        tx.execute(
            "INSERT INTO body_value(codec,length,hash,state,incoming) VALUES(?1,?2,?3,'ready',1)",
            params![codec, bytes.len() as i64, hash.as_bytes().as_slice()],
        )
    })?;
    let pk = tx.last_insert_rowid();
    for (n, bytes) in bytes.chunks(crate::values::MAX_CHUNK).enumerate() {
        db(path, || {
            tx.execute(
                "INSERT INTO body_value_chunk(value_pk,offset,bytes) VALUES(?1,?2,?3)",
                params![pk, (n * crate::values::MAX_CHUNK) as i64, bytes],
            )
        })?;
    }
    Ok(pk)
}
fn bytes(c: &Connection, p: &Path, pk: i64, allowance: usize) -> Result<Vec<u8>> {
    let (length, hash): (i64, Vec<u8>) = db(p, || {
        c.query_row(
            "SELECT length,hash FROM body_value WHERE pk=?1 AND state='ready'",
            [pk],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
    })?;
    if length < 0 || length as usize > allowance {
        return Err(Error::ParserAdmission {
            resource: "operation payload bytes",
            required: length.max(0) as u64,
        });
    }
    let mut out = Vec::with_capacity(length as usize);
    let mut s = db(p, || {
        c.prepare("SELECT offset,bytes FROM body_value_chunk WHERE value_pk=?1 ORDER BY offset")
    })?;
    let mut q = db(p, || s.query([pk]))?;
    while let Some(r) = db(p, || q.next())? {
        let offset: i64 = db(p, || r.get(0))?;
        let b: Vec<u8> = db(p, || r.get(1))?;
        if offset != out.len() as i64 {
            return Err(bad("operation payload gap"));
        }
        out.extend_from_slice(&b);
    }
    if out.len() != length as usize || blake3::hash(&out).as_bytes().as_slice() != hash {
        return Err(bad("operation payload hash differs"));
    }
    Ok(out)
}
impl Writer {
    /// Caller supplies one already-decoded operation from this pinned parser
    /// snapshot. Publication is idempotent by exact input and evidence digest.
    pub fn publish_tool_item(
        &mut self,
        token: &Token,
        operation: &Operation,
        digest: polylogue_model::BlobHash,
    ) -> Result<i64> {
        let duration = operation
            .duration_ms
            .map(i64::try_from)
            .transpose()
            .map_err(|_| Error::Range)?;
        let id = operation.id().map_err(|e| bad(e.to_string()))?;
        let mut metadata = operation.clone();
        let output = metadata.output.take();
        let encoded = serde_json::to_vec(&metadata).map_err(|e| bad(e.to_string()))?;
        if encoded.len() + output.as_ref().map_or(0, String::len) > 2 * 1024 * 1024 {
            return Err(Error::ParserAdmission {
                resource: "operation publication bytes",
                required: (encoded.len() + output.as_ref().map_or(0, String::len)) as u64,
            });
        }
        if operation.origins.is_empty() {
            return Err(bad("operation has no source evidence"));
        }

        let p = self.root.join("index.db");
        let tx = db(&p, || self._index.transaction())?;
        let h = super::contributions::writer::source_head(&tx, &p, token)?;
        if h.status != "finalizing" && h.status != "done" {
            return Err(Error::ParserPredecessorChanged);
        }
        let native = &operation.native_session;
        if polylogue_model::SessionId::new(
            &polylogue_model::Origin::new("codex").map_err(|_| bad("namespace"))?,
            native,
        )
        .map_err(|e| bad(e.to_string()))?
            != operation.session
        {
            return Err(bad("operation Session scope differs"));
        }
        db(&p, || {
            tx.execute("INSERT INTO session(id,origin,native_id,kind) VALUES(?1,'codex',?2,'main') ON CONFLICT(id) DO NOTHING",params![operation.session.as_str(),native])
        })?;
        let session: i64 = db(&p, || {
            tx.query_row(
                "SELECT pk FROM session WHERE id=?1",
                [operation.session.as_str()],
                |r| r.get(0),
            )
        })?;
        db(&p, || {
            tx.execute("INSERT INTO parsed(input_pk,chain,session_pk,parser_version) SELECT ?1,?2,?3,parser_version FROM parse_state WHERE input_pk=?1 AND chain=?2 ON CONFLICT DO NOTHING", params![token.input_pk,token.chain,session])
        })?;
        db(&p, || {
            tx.execute("INSERT INTO tool_item(id,session_pk,provider_namespace,identity_schema,identity_kind,native_turn_id,native_item_id) VALUES(?1,?2,'codex',?3,?4,?5,?6) ON CONFLICT(id) DO NOTHING",params![id,session,CODEX_CONSTRUCTOR,if operation.native{"native"}else{"provisional"},if operation.native{operation.turn.as_deref()}else{None},operation.native.then_some(operation.invocation.as_str())])
        })?;
        let item: i64 = db(&p, || {
            tx.query_row(
                "SELECT pk FROM tool_item WHERE id=?1 AND session_pk=?2",
                params![id, session],
                |r| r.get(0),
            )
        })?;
        let existing: Option<i64> = db(&p, || {
            tx.query_row("SELECT v.pk FROM tool_item_version v JOIN tool_item_source s ON s.tool_item_version_pk=v.pk WHERE v.tool_item_pk=?1 AND v.version_hash=?2 AND s.input_pk=?3 AND s.source_chain=?4 LIMIT 1",params![item,digest.bytes().as_slice(),token.input_pk,token.chain],|r|r.get(0)).optional()
        })?;
        if let Some(v) = existing {
            db(&p, || tx.commit())?;
            return Ok(v);
        }
        let input = payload(&tx, &p, "json", &encoded)?;
        let text = if let Some(output) = &output {
            let value = payload(&tx, &p, "utf8", output.as_bytes())?;
            let hash = blake3::hash(output.as_bytes());
            db(&p, || {
                tx.execute(
                    "INSERT INTO text(hash,value_pk) VALUES(?1,?2) ON CONFLICT(hash) DO NOTHING",
                    params![hash.as_bytes().as_slice(), value],
                )
            })?;
            Some(db(&p, || {
                tx.query_row(
                    "SELECT id FROM text WHERE hash=?1",
                    [hash.as_bytes().as_slice()],
                    |r| r.get::<_, i64>(0),
                )
            })?)
        } else {
            None
        };
        db(&p, || {
            tx.execute("INSERT INTO tool_item_version(tool_item_pk,session_pk,evidence_revision,ready,version_hash,kind,agent_class,correspondence,call_present,call_id,input_value_pk,result_present,output_present,result_text_id,result_outcome,result_exit_code,result_duration_ms) VALUES(?1,?2,?3,1,?4,?16,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",params![item,session,token.step,digest.bytes().as_slice(),operation.agent_class,operation.correspondence,operation.call_present,operation.invocation,if operation.call_present{Some(input)}else{None},operation.result_present,output.is_some(),text,operation.outcome,operation.exit_code,duration,operation.kind])
        })?;
        let version = tx.last_insert_rowid();
        for (slot, origin) in operation.origins.iter().enumerate() {
            let (start, end, constructor) = match origin {
                ToolOrigin::Rollout { start, end, .. } => {
                    (*start, *end, format!("{CODEX_CONSTRUCTOR}/core"))
                }
                ToolOrigin::History { .. } => (
                    0,
                    h.acquired_upto,
                    format!("{CODEX_CONSTRUCTOR}/core-to-ui-empty-output-filter"),
                ),
            };
            let field = serde_json::to_string(origin).map_err(|e| bad(e.to_string()))?;
            db(&p, || {
                tx.execute("INSERT INTO tool_item_source(tool_item_version_pk,tool_item_pk,session_pk,input_pk,source_chain,source_start,source_end,field,slot,constructor) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",params![version,item,session,token.input_pk,token.chain,start as i64,end as i64,field,slot as i64,constructor])
            })?;
        }
        for anchor in &operation.direct_anchors {
            if !operation.native {
                return Err(bad("only native keys may receive direct operation anchors"));
            }
            let (start, end) = operation
                .origins
                .iter()
                .find_map(|o| match o {
                    ToolOrigin::Rollout {
                        tag, start, end, ..
                    } if tag == "call" => Some((*start, *end)),
                    _ => None,
                })
                .ok_or_else(|| bad("operation anchor has no direct call witness"))?;
            let old: Option<i64> = db(&p, || {
                tx.query_row("SELECT t.pk FROM tool_item t JOIN tool_item_version v ON v.tool_item_pk=t.pk JOIN tool_item_source s ON s.tool_item_version_pk=v.pk WHERE t.id=?1 AND t.session_pk=?2 AND t.identity_kind='provisional' AND s.input_pk=?3 AND s.source_chain=?4 AND s.source_start=?5 AND s.source_end=?6 LIMIT 1",params![anchor,session,token.input_pk,token.chain,start as i64,end as i64],|r|r.get(0)).optional()
            })?;
            let Some(old) = old else {
                return Err(bad(
                    "operation anchor lacks its exact prior Session/source witness",
                ));
            };
            let conflict: Option<i64> = db(&p, || {
                tx.query_row(
                    "SELECT tool_item_pk FROM tool_call_anchor WHERE anchor=?1",
                    [anchor],
                    |r| r.get(0),
                )
                .optional()
            })?;
            if conflict.is_some_and(|v| v != item) {
                return Err(bad(
                    "direct operation anchor conflicts with an established native key",
                ));
            }
            db(&p, || {
                tx.execute("INSERT INTO tool_call_anchor(anchor,tool_item_pk,tool_item_version_pk,session_pk,input_pk,source_chain,source_start,source_end,constructor) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9) ON CONFLICT(anchor) DO NOTHING",params![anchor,item,version,session,token.input_pk,token.chain,start as i64,end as i64,CODEX_CONSTRUCTOR])
            })?;
            db(&p, || {
                tx.execute(
                    "UPDATE tool_item SET selected_version_pk=NULL WHERE pk=?1",
                    [old],
                )
            })?;
        }
        // A Message association requires its independently published identity in
        // this Session. Unknown owners leave no association row.
        if let Some(message) = &operation.message {
            let owner: Option<i64> = db(&p, || {
                tx.query_row(
                    "SELECT pk FROM message WHERE id=?1 AND session_pk=?2",
                    params![message, session],
                    |r| r.get(0),
                )
                .optional()
            })?;
            if let Some(owner) = owner {
                let (start, end) = operation
                    .origins
                    .iter()
                    .find_map(|o| match o {
                        ToolOrigin::Rollout { start, end, .. } => Some((*start, *end)),
                        _ => None,
                    })
                    .ok_or_else(|| bad("Message owner has no direct rollout witness"))?;
                db(&p, || {
                    tx.execute("INSERT INTO tool_item_association(tool_item_version_pk,tool_item_pk,message_pk,session_pk,kind,input_pk,source_chain,source_start,source_end,constructor) VALUES(?1,?2,?3,?4,'message_owner',?5,?6,?7,?8,?9)",params![version,item,owner,session,token.input_pk,token.chain,start as i64,end as i64,CODEX_CONSTRUCTOR])
                })?;
            }
        }
        let refined: bool = db(&p, || {
            tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM tool_call_anchor WHERE anchor=?1)",
                [&id],
                |r| r.get(0),
            )
        })?;
        if !refined {
            db(&p, || {
                tx.execute("UPDATE tool_item SET selected_version_pk=(SELECT pk FROM tool_item_version WHERE tool_item_pk=?1 ORDER BY result_present DESC,output_present DESC,(SELECT count(*) FROM tool_item_association a WHERE a.tool_item_version_pk=tool_item_version.pk) DESC,call_present DESC,version_hash ASC,pk ASC LIMIT 1) WHERE pk=?1",[item])
            })?;
        }
        // Keep one selected Session-owned summary. Block-backed summaries have
        // immutable body leases and are left to their existing owner.
        db(&p, || {
            tx.execute(
                "DELETE FROM tool_call WHERE tool_item_pk=?1 AND call_block_pk IS NULL",
                [item],
            )
        })?;
        let selected: i64 = db(&p, || {
            tx.query_row(
                "SELECT coalesce(selected_version_pk,0) FROM tool_item WHERE pk=?1",
                [item],
                |r| r.get(0),
            )
        })?;
        if selected != 0 {
            db(&p, || {
                tx.execute("INSERT INTO tool_call(session_pk,operation_id,tool_item_pk,tool_item_version_pk,outcome,outcome_reason,exit_code,duration_ms,action) SELECT session_pk,?2,tool_item_pk,pk,coalesce(result_outcome,'unknown'),CASE WHEN result_present=0 THEN 'no_result' ELSE NULL END,result_exit_code,result_duration_ms,'shell' FROM tool_item_version WHERE pk=?1 AND agent_class='agent' AND correspondence='established' AND call_present=1",params![selected,id])
            })?;
        }
        db(&p, || tx.commit())?;
        Ok(version)
    }
}

fn citations(reader: &ReadHandle, id: &str, version: Option<i64>) -> Result<Vec<Citation>> {
    let p = reader.root.join("index.db");
    let mut s = db(&p, || {
        reader.index.prepare("SELECT i.id,s.source_chain,s.source_start,s.source_end,s.field,s.constructor,lower(hex(v.version_hash)) FROM tool_item_source s JOIN input_ref i ON i.pk=s.input_pk JOIN tool_item_version v ON v.pk=s.tool_item_version_pk JOIN tool_item t ON t.pk=s.tool_item_pk WHERE t.id=?1 AND (?2 IS NULL OR v.pk=?2) ORDER BY i.id,v.pk,s.source_start,s.slot LIMIT 4097")
    })?;
    let raw = db(&p, || {
        s.query_map(params![id, version], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, String>(5)?,
                r.get::<_, String>(6)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()
    })?;
    if raw.len() > 4096 {
        return Err(Error::ParserAdmission {
            resource: "operation source citations",
            required: raw.len() as u64,
        });
    }
    raw.into_iter()
        .map(
            |(input, chain, start, end, origin, constructor, version_hash)| {
                Ok(Citation {
                    input,
                    chain,
                    start: u64::try_from(start).map_err(|_| Error::Range)?,
                    end: u64::try_from(end).map_err(|_| Error::Range)?,
                    origin: serde_json::from_str(&origin).map_err(|e| bad(e.to_string()))?,
                    constructor,
                    version_hash,
                })
            },
        )
        .collect()
}
fn verify_sources(
    reader: &ReadHandle,
    sources: &[Citation],
    verified: &mut std::collections::BTreeSet<(String, i64, u64, u64)>,
) -> Result<()> {
    for source in sources {
        if !verified.insert((source.input.clone(), source.chain, source.start, source.end)) {
            continue;
        }
        let retained = reader
            .source(&source.input)?
            .ok_or_else(|| bad("operation source unavailable"))?;
        if source.end < source.start {
            return Err(Error::Range);
        }
        let mut at = source.start;
        let mut hash = blake3::Hasher::new();
        while at < source.end {
            let chunk = reader.read_source_range(
                &source.input,
                source.chain,
                retained.binding,
                at,
                (source.end - at).min(65536),
            )?;
            if chunk.is_empty() {
                return Err(bad("operation retained source range unavailable"));
            }
            hash.update(&chunk);
            at += chunk.len() as u64;
        }
        if matches!(source.origin, ToolOrigin::History { .. })
            && (source.start != 0
                || source.end != retained.input.length
                || hash.finalize().as_bytes() != retained.binding.bytes())
        {
            return Err(bad("physical history snapshot digest differs"));
        }
    }
    Ok(())
}
fn load_item(
    reader: &ReadHandle,
    reference: String,
    version: i64,
    hash: String,
    input: Option<i64>,
    text: Option<i64>,
    verified: &mut std::collections::BTreeSet<(String, i64, u64, u64)>,
) -> Result<Item> {
    let p = reader.root.join("index.db");
    let mut operation: Operation = serde_json::from_slice(&bytes(
        &reader.index,
        &p,
        input.ok_or_else(|| bad("operation call payload unavailable"))?,
        2 * 1024 * 1024,
    )?)
    .map_err(|e| bad(e.to_string()))?;
    if operation.id().map_err(|e| bad(e.to_string()))? != reference {
        return Err(bad(
            "operation payload identity differs from immutable owner",
        ));
    }
    if let Some(text) = text {
        let value: i64 = db(&p, || {
            reader
                .index
                .query_row("SELECT value_pk FROM text WHERE id=?1", [text], |r| {
                    r.get(0)
                })
        })?;
        operation.output = Some(
            String::from_utf8(bytes(&reader.index, &p, value, 2 * 1024 * 1024)?)
                .map_err(|e| bad(e.to_string()))?,
        );
    }
    let sources = citations(reader, &reference, Some(version))?;
    verify_sources(reader, &sources, verified)?;
    let representations = citations(reader, &reference, None)?;
    let conflicting_representations: bool = db(&p, || {
        reader.index.query_row("SELECT EXISTS(SELECT 1 FROM tool_item_version a JOIN tool_item_version b ON a.tool_item_pk=b.tool_item_pk WHERE a.pk=?1 AND ((a.agent_class IN ('agent','non_agent') AND b.agent_class IN ('agent','non_agent') AND a.agent_class!=b.agent_class) OR (a.result_present=1 AND b.result_present=1 AND ((a.result_outcome IS NOT NULL AND b.result_outcome IS NOT NULL AND a.result_outcome!=b.result_outcome) OR (a.result_exit_code IS NOT NULL AND b.result_exit_code IS NOT NULL AND a.result_exit_code!=b.result_exit_code) OR (a.output_present=1 AND b.output_present=1 AND a.result_text_id!=b.result_text_id)))))", [version], |r|r.get(0))
    })?;
    Ok(Item {
        reference,
        version,
        version_hash: hash.to_lowercase(),
        operation,
        sources,
        representations,
        conflicting_representations,
    })
}
impl ReadHandle {
    /// One pinned index transaction. Payload bytes and old versions are immutable;
    /// source availability is checked only for this selected body's contributors.
    pub fn tool_items(&self, session: &str) -> Result<Population> {
        let p = self.root.join("index.db");
        let _snapshot = db(&p, || self.index.unchecked_transaction())?;
        let mut s = db(&p, || {
            self.index.prepare("SELECT t.id,v.pk,hex(v.version_hash),v.input_value_pk,v.result_text_id FROM tool_item t JOIN tool_item_version v ON v.pk=t.selected_version_pk JOIN session s ON s.pk=t.session_pk WHERE s.id=?1 ORDER BY t.id LIMIT 4097")
        })?;
        let rows = db(&p, || {
            s.query_map([session], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, Option<i64>>(3)?,
                    r.get::<_, Option<i64>>(4)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()
        })?;
        if rows.len() > 4096 {
            return Err(Error::ParserAdmission {
                resource: "operation population",
                required: rows.len() as u64,
            });
        }
        let mut items = Vec::new();
        let mut verified = std::collections::BTreeSet::new();
        let mut size = 0usize;
        for (id, v, h, input, text) in rows {
            let item = load_item(self, id, v, h, input, text, &mut verified)?;
            size = size.saturating_add(
                serde_json::to_vec(&item)
                    .map_err(|e| bad(e.to_string()))?
                    .len(),
            );
            if size > 4 * 1024 * 1024 {
                return Err(Error::ParserAdmission {
                    resource: "operation population bytes",
                    required: size as u64,
                });
            }
            items.push(item);
        }
        let suppressed_wrappers: std::collections::BTreeSet<_> = items
            .iter()
            .filter(|i| i.operation.kind != "CodeExecution")
            .flat_map(|i| i.operation.wrapper_overlaps.iter().cloned())
            .collect();
        let mut calls = 0;
        let mut failed = 0;
        let mut complete = !items.is_empty();
        for item in &items {
            if item.operation.counted()
                && !item.conflicting_representations
                && !suppressed_wrappers.contains(&item.reference)
            {
                calls += 1;
                if item.operation.outcome.as_deref() == Some("error") {
                    failed += 1;
                }
            } else if item.operation.agent_class != "non_agent"
                && !suppressed_wrappers.contains(&item.reference)
            {
                complete = false;
            }
        }
        let retained_representations: i64 = db(&p, || {
            self.index.query_row("SELECT count(*) FROM tool_item_version v JOIN session s ON s.pk=v.session_pk WHERE s.id=?1",[session],|r|r.get(0))
        })?;
        let (retained_rows, decoded_rows, unsupported_rows): (i64, i64, i64) = db(&p, || {
            self.index.query_row("SELECT coalesce(sum(framed),0),coalesce(sum(used+ignored),0),coalesce(sum(unrecognized_n+undecodable),0) FROM parse_state p WHERE EXISTS(SELECT 1 FROM parsed x JOIN session s ON s.pk=x.session_pk WHERE x.input_pk=p.input_pk AND x.chain=p.chain AND s.id=?1)", [session], |r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))
        })?;
        complete &= unsupported_rows == 0;
        Ok(Population {
            items,
            tool_calls: calls,
            failed,
            correspondence_complete: complete,
            retained_representations: retained_representations as usize,
            retained_rows: retained_rows as u64,
            decoded_rows: decoded_rows as u64,
            unsupported_rows: unsupported_rows as u64,
            suppressed_wrappers: suppressed_wrappers.into_iter().collect(),
        })
    }
    pub fn tool_item(&self, reference: &str) -> Result<Option<Item>> {
        let (mut id, hash) = reference
            .split_once('@')
            .map_or((reference, None), |(id, h)| (id, Some(h)));
        if hash.is_some_and(|h| h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit())) {
            return Err(bad("invalid operation version hash"));
        }
        let p = self.root.join("index.db");
        let _snapshot = db(&p, || self.index.unchecked_transaction())?;
        let anchored: Option<String> = if hash.is_none() {
            db(&p, || {
                self.index.query_row("SELECT t.id FROM tool_call_anchor a JOIN tool_item t ON t.pk=a.tool_item_pk WHERE a.anchor=?1",[id],|r|r.get(0)).optional()
            })?
        } else {
            None
        };
        if let Some(anchor) = &anchored {
            id = anchor;
        }
        let row: Option<(i64, String, Option<i64>, Option<i64>)> = db(&p, || {
            self.index.query_row("SELECT v.pk,hex(v.version_hash),v.input_value_pk,v.result_text_id FROM tool_item t JOIN tool_item_version v ON v.tool_item_pk=t.pk WHERE t.id=?1 AND ((?2 IS NULL AND v.pk=t.selected_version_pk) OR lower(hex(v.version_hash))=lower(?2)) ORDER BY v.pk LIMIT 1",params![id,hash],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).optional()
        })?;
        row.map(|(v, h, input, text)| {
            load_item(
                self,
                id.into(),
                v,
                h,
                input,
                text,
                &mut std::collections::BTreeSet::new(),
            )
        })
        .transpose()
    }
}
