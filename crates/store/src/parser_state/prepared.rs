//! Durable provider-state application. This barrier never mutates canonical
//! messages. Whole encoded cells are admitted explicitly; chunked huge-cell
//! installation and physical SQLite work remain unqualified.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkToken {
    pub predecessor: Token,
    pub work_id: i64,
    pub staged: i64,
    pub applied: i64,
}
impl WorkToken {
    /// Admit one staging transaction using the same encoded framing as stage().
    /// An unfit first member fails before mutation; a later unfit member remains
    /// owed. The returned count never exceeds either caller allowance.
    pub fn staging_prefix(&self, changes: &[CellChange], budget: Budget) -> Result<usize> {
        if changes.is_empty() {
            return Ok(0);
        }
        validate(&changes[..1], budget)?;
        let mut bytes = 0usize;
        let mut count = 0;
        for change in changes.iter().take(budget.cells) {
            let required = staging_bytes(self, change)?;
            let next = bytes.checked_add(required).ok_or(StoreError::Range)?;
            if next > budget.encoded_bytes {
                if count == 0 {
                    return Err(StoreError::ParserAdmission {
                        resource: "prepared encoded bytes",
                        required: next as u64,
                    });
                }
                break;
            }
            bytes = next;
            count += 1;
        }
        Ok(count)
    }
}
fn staging_bytes(token: &WorkToken, change: &CellChange) -> Result<usize> {
    [
        25,
        namespace(token.work_id).len(),
        change.namespace.len(),
        change.key.len(),
        change.value.as_ref().map_or(0, Vec::len),
    ]
    .into_iter()
    .try_fold(0usize, |sum, n| sum.checked_add(n).ok_or(StoreError::Range))
}
pub struct Work {
    pub token: WorkToken,
    pub applying: bool,
    pub discarding: bool,
    pub acquired_upto: u64,
    pub progress: Progress,
}
fn namespace(id: i64) -> String {
    format!("@work:{id}")
}
pub(super) fn ensure_idle(db: &Connection, path: &Path, token: &Token) -> Result<()> {
    let pending: bool = sql!(
        path,
        db.query_row(
            "SELECT EXISTS(SELECT 1 FROM parse_work WHERE input_pk=?1 AND chain=?2)",
            params![token.input_pk, token.chain],
            |r| r.get(0)
        )
    )?;
    if pending {
        Err(StoreError::ParserWorkPending)
    } else {
        Ok(())
    }
}
pub(super) fn invalidate(db: &Connection, path: &Path, input: i64, chain: i64) -> Result<()> {
    sql!(
        path,
        db.execute(
            "DELETE FROM parse_work WHERE input_pk=?1 AND chain=?2",
            params![input, chain]
        )
    )?;
    Ok(())
}
fn load(db: &Connection, path: &Path, token: &Token) -> Result<Option<Work>> {
    current(db, path, token)?;
    let row=sql!(path,db.query_row(
        "SELECT work_id,state_generation,state_step,basis_digest,acquired_upto,phase,staged,applied,
         inspected_upto,parsed_upto,next_status,finalization_pending,framed,used,ignored,unrecognized_n,undecodable
         FROM parse_work WHERE input_pk=?1 AND chain=?2",params![token.input_pk,token.chain],|r| {
            Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,i64>(2)?,r.get::<_,Vec<u8>>(3)?,
                unsigned(r,4)?,r.get::<_,String>(5)?,r.get::<_,i64>(6)?,r.get::<_,i64>(7)?,
                unsigned(r,8)?,unsigned(r,9)?,r.get::<_,String>(10)?,r.get::<_,bool>(11)?,
                Accounting{framed:unsigned(r,12)?,used:unsigned(r,13)?,ignored:unsigned(r,14)?,unrecognized:unsigned(r,15)?,undecodable:unsigned(r,16)?}))
        }).optional())?;
    row.map(
        |(
            id,
            generation,
            step,
            basis,
            acquired,
            phase,
            staged,
            applied,
            inspected,
            parsed,
            status,
            pending,
            counts,
        )| {
            if generation != token.generation || step != token.step || basis != token.basis {
                return Err(StoreError::StaleParserState);
            }
            let discarding = phase == "discarding";
            let phase = match phase.as_str() {
                "staging" => false,
                "applying" => true,
                "discarding" => false,
                _ => return Err(StoreError::Invariant("invalid prepared phase".into())),
            };
            let next = match status.as_str() {
                "parsing" => Phase::Parsing,
                "waiting_tail" => Phase::WaitingTail,
                "finalizing" => Phase::Finalizing,
                "done" => Phase::Done,
                _ => {
                    return Err(StoreError::Invariant(
                        "invalid prepared target phase".into(),
                    ));
                }
            };
            Ok(Work {
                token: WorkToken {
                    predecessor: token.clone(),
                    work_id: id,
                    staged,
                    applied,
                },
                applying: phase,
                discarding,
                acquired_upto: acquired,
                progress: Progress {
                    inspected_upto: inspected,
                    parsed_upto: parsed,
                    phase: next,
                    finalization_pending: pending,
                    counts,
                },
            })
        },
    )
    .transpose()
}
fn expected(db: &Connection, path: &Path, token: &WorkToken) -> Result<Work> {
    load(db, path, &token.predecessor)?
        .filter(|w| w.token == *token)
        .ok_or(StoreError::StaleParserState)
}
fn encode(change: &CellChange) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(
        17 + change.namespace.len() + change.key.len() + change.value.as_ref().map_or(0, Vec::len),
    );
    bytes.push(u8::from(change.value.is_some()));
    bytes.extend_from_slice(&(change.namespace.len() as u64).to_be_bytes());
    bytes.extend_from_slice(&(change.key.len() as u64).to_be_bytes());
    bytes.extend_from_slice(change.namespace.as_bytes());
    bytes.extend_from_slice(&change.key);
    if let Some(value) = &change.value {
        bytes.extend_from_slice(value)
    }
    bytes
}
fn decode(bytes: &[u8]) -> Result<CellChange> {
    let invalid = || StoreError::Invariant("corrupt prepared provider cell".into());
    if bytes.len() < 17 || bytes[0] > 1 {
        return Err(invalid());
    }
    let n = usize::try_from(u64::from_be_bytes(
        bytes[1..9].try_into().map_err(|_| invalid())?,
    ))
    .map_err(|_| invalid())?;
    let k = usize::try_from(u64::from_be_bytes(
        bytes[9..17].try_into().map_err(|_| invalid())?,
    ))
    .map_err(|_| invalid())?;
    let middle = 17usize.checked_add(n).ok_or_else(invalid)?;
    let end = middle
        .checked_add(k)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(invalid)?;
    if bytes[0] == 0 && end != bytes.len() {
        return Err(invalid());
    }
    let change = CellChange {
        namespace: std::str::from_utf8(&bytes[17..middle])
            .map_err(|_| invalid())?
            .into(),
        key: bytes[middle..end].to_vec(),
        value: (bytes[0] == 1).then(|| bytes[end..].to_vec()),
    };
    validate(
        std::slice::from_ref(&change),
        Budget {
            cells: 1,
            encoded_bytes: bytes.len(),
        },
    )?;
    Ok(change)
}
pub(crate) fn admit(
    db: &mut Connection,
    path: &Path,
    head: &Head,
    progress: Progress,
) -> Result<Work> {
    // Use the same progress validator as immediate small-step publication.
    let tx = sql!(path, db.transaction())?;
    let now = current(&tx, path, &head.token)?;
    validate_expected_head(head, &now)?;
    ensure_idle(&tx, path, &head.token)?;
    if now.status == "retired" {
        return Err(StoreError::StaleParserState);
    }
    validate_progress(head, &progress)?;
    let id = allocate_generation(&tx, path)?;
    let values = [
        progress.inspected_upto,
        progress.parsed_upto,
        progress.counts.framed,
        progress.counts.used,
        progress.counts.ignored,
        progress.counts.unrecognized,
        progress.counts.undecodable,
    ]
    .map(integer)
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    let acquired = integer(head.acquired_upto)?;
    sql!(path,tx.execute("INSERT INTO parse_work(input_pk,chain,work_id,state_generation,state_step,basis_digest,acquired_upto,phase,
        inspected_upto,parsed_upto,next_status,finalization_pending,framed,used,ignored,unrecognized_n,undecodable)
        VALUES(?1,?2,?3,?4,?5,?6,?7,'staging',?8,?9,?10,?11,?12,?13,?14,?15,?16)",
        params![head.token.input_pk,head.token.chain,id,head.token.generation,head.token.step,head.token.basis.as_slice(),acquired,
        values[0],values[1],progress.phase.name(),progress.finalization_pending,values[2],values[3],values[4],values[5],values[6]]))?;
    let work = load(&tx, path, &head.token)?.ok_or(StoreError::StaleParserState)?;
    sql!(path, tx.commit())?;
    Ok(work)
}
pub(crate) fn stage(
    db: &mut Connection,
    path: &Path,
    token: &WorkToken,
    changes: &[CellChange],
    budget: Budget,
) -> Result<Work> {
    validate(changes, budget)?;
    // Prefix selection and whole-slice staging share exact encoded framing.
    let bytes = changes.iter().try_fold(0usize, |sum, change| {
        sum.checked_add(staging_bytes(token, change)?)
            .ok_or(StoreError::Range)
    })?;
    if bytes > budget.encoded_bytes {
        return Err(StoreError::ParserAdmission {
            resource: "prepared encoded bytes",
            required: bytes as u64,
        });
    }
    let encoded: Vec<_> = changes.iter().map(encode).collect();
    let count = i64::try_from(encoded.len())
        .map_err(|_| StoreError::Invariant("prepared ordinal overflow".into()))?;
    let end = token
        .staged
        .checked_add(count)
        .ok_or_else(|| StoreError::Invariant("prepared ordinal overflow".into()))?;
    let tx = sql!(path, db.transaction())?;
    let work = expected(&tx, path, token)?;
    if work.applying || work.discarding {
        return Err(StoreError::Invariant("cannot stage after seal".into()));
    }
    let ns = namespace(token.work_id);
    for (ordinal, value) in (token.staged..end).zip(encoded) {
        sql!(path,tx.execute("INSERT INTO parser_cell(input_pk,chain,state_generation,namespace,key,value) VALUES(?1,?2,?3,?4,?5,?6)",
            params![token.predecessor.input_pk,token.predecessor.chain,token.predecessor.generation,ns,ordinal.to_be_bytes().as_slice(),value]))?;
    }
    sql!(
        path,
        tx.execute(
            "UPDATE parse_work SET staged=?3 WHERE input_pk=?1 AND chain=?2",
            params![token.predecessor.input_pk, token.predecessor.chain, end]
        )
    )?;
    let result = load(&tx, path, &token.predecessor)?.ok_or(StoreError::StaleParserState)?;
    sql!(path, tx.commit())?;
    Ok(result)
}
pub(crate) fn seal(db: &mut Connection, path: &Path, token: &WorkToken) -> Result<Work> {
    let tx = sql!(path, db.transaction())?;
    let work = expected(&tx, path, token)?;
    if work.discarding {
        return Err(StoreError::Invariant(
            "cannot seal discarded preparation".into(),
        ));
    }
    sql!(
        path,
        tx.execute(
            "UPDATE parse_work SET phase='applying' WHERE input_pk=?1 AND chain=?2",
            params![token.predecessor.input_pk, token.predecessor.chain]
        )
    )?;
    let result = load(&tx, path, &token.predecessor)?.ok_or(StoreError::StaleParserState)?;
    sql!(path, tx.commit())?;
    Ok(result)
}
pub(crate) fn apply(
    db: &mut Connection,
    path: &Path,
    token: &WorkToken,
    budget: Budget,
) -> Result<Work> {
    if budget.cells == 0 {
        return Err(StoreError::ParserAdmission {
            resource: "prepared apply cells",
            required: 1,
        });
    }
    let tx = sql!(path, db.transaction())?;
    let work = expected(&tx, path, token)?;
    if !work.applying {
        return Err(StoreError::Invariant(
            "seal all provider changes before application".into(),
        ));
    }
    let ns = namespace(token.work_id);
    let mut cursor = token.applied;
    let mut charged = 0usize;
    while cursor < token.staged
        && (cursor - token.applied) < i64::try_from(budget.cells).unwrap_or(i64::MAX)
    {
        let allowance = budget.encoded_bytes.saturating_sub(charged);
        let allowance_i = integer(allowance as u64)?;
        let (required,bytes)=sql!(path,tx.query_row("SELECT length(value),CASE WHEN length(value)<=?6 THEN value ELSE NULL END FROM parser_cell
            WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key=?5",
            params![token.predecessor.input_pk,token.predecessor.chain,token.predecessor.generation,ns,cursor.to_be_bytes().as_slice(),allowance_i],
            |r|Ok((unsigned(r,0)?,r.get::<_,Option<Vec<u8>>>(1)?))))?;
        let Some(bytes) = bytes else {
            if cursor > token.applied {
                break;
            }
            return Err(StoreError::ParserAdmission {
                resource: "prepared apply bytes",
                required,
            });
        };
        let c = decode(&bytes)?;
        if let Some(value) = c.value {
            sql!(path,tx.execute("INSERT INTO parser_cell(input_pk,chain,state_generation,namespace,key,value) VALUES(?1,?2,?3,?4,?5,?6)
            ON CONFLICT(input_pk,chain,state_generation,namespace,key) DO UPDATE SET value=excluded.value",params![token.predecessor.input_pk,token.predecessor.chain,token.predecessor.generation,c.namespace,c.key,value]))?;
        } else {
            sql!(path,tx.execute("DELETE FROM parser_cell WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key=?5",params![token.predecessor.input_pk,token.predecessor.chain,token.predecessor.generation,c.namespace,c.key]))?;
        }
        sql!(path,tx.execute("DELETE FROM parser_cell WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key=?5",params![token.predecessor.input_pk,token.predecessor.chain,token.predecessor.generation,ns,cursor.to_be_bytes().as_slice()]))?;
        charged = charged.saturating_add(bytes.len());
        cursor += 1;
    }
    sql!(
        path,
        tx.execute(
            "UPDATE parse_work SET applied=?3 WHERE input_pk=?1 AND chain=?2",
            params![token.predecessor.input_pk, token.predecessor.chain, cursor]
        )
    )?;
    let result = load(&tx, path, &token.predecessor)?.ok_or(StoreError::StaleParserState)?;
    sql!(path, tx.commit())?;
    Ok(result)
}
/// An unsealed preparation has made no provider changes and may be discarded
/// after a crash. Once sealed, it must finish or be invalidated by a new basis.
/// Cleanup charges deleted member lengths without materializing their bodies.
pub(crate) fn discard(
    db: &mut Connection,
    path: &Path,
    token: &WorkToken,
    budget: Budget,
) -> Result<Option<Work>> {
    if budget.cells == 0 {
        return Err(StoreError::ParserAdmission {
            resource: "prepared discard cells",
            required: 1,
        });
    }
    let tx = sql!(path, db.transaction())?;
    let work = expected(&tx, path, token)?;
    if work.applying {
        return Err(StoreError::Invariant(
            "sealed provider work must finish or change basis".into(),
        ));
    }
    let ns = namespace(token.work_id);
    let mut cursor = token.applied;
    let mut charged = 0usize;
    while cursor < token.staged
        && (cursor - token.applied) < i64::try_from(budget.cells).unwrap_or(i64::MAX)
    {
        let required=sql!(path,tx.query_row("SELECT length(value) FROM parser_cell WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key=?5",
            params![token.predecessor.input_pk,token.predecessor.chain,token.predecessor.generation,ns,cursor.to_be_bytes().as_slice()],|r|unsigned(r,0)))?;
        if required > budget.encoded_bytes.saturating_sub(charged) as u64 {
            if cursor > token.applied {
                break;
            }
            return Err(StoreError::ParserAdmission {
                resource: "prepared discard bytes",
                required,
            });
        }
        sql!(path,tx.execute("DELETE FROM parser_cell WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key=?5",params![token.predecessor.input_pk,token.predecessor.chain,token.predecessor.generation,ns,cursor.to_be_bytes().as_slice()]))?;
        charged = charged.saturating_add(required as usize);
        cursor += 1;
    }
    let result = if cursor == token.staged {
        invalidate(
            &tx,
            path,
            token.predecessor.input_pk,
            token.predecessor.chain,
        )?;
        None
    } else {
        sql!(path,tx.execute("UPDATE parse_work SET phase='discarding',applied=?3 WHERE input_pk=?1 AND chain=?2",params![token.predecessor.input_pk,token.predecessor.chain,cursor]))?;
        load(&tx, path, &token.predecessor)?
    };
    sql!(path, tx.commit())?;
    Ok(result)
}
pub(crate) fn publish(db: &mut Connection, path: &Path, token: &WorkToken) -> Result<Head> {
    let work = expected(db, path, token)?;
    if !work.applying || token.applied != token.staged {
        return Err(StoreError::ParserWorkPending);
    }
    let head = current(db, path, &token.predecessor)?;
    let admitted = Head {
        acquired_upto: work.acquired_upto,
        ..head
    };
    write_step_inner(
        db,
        path,
        &admitted,
        work.progress,
        &[],
        Budget {
            cells: 0,
            encoded_bytes: 0,
        },
        true,
        |tx| {
            let work = expected(tx, path, token)?;
            if !work.applying || work.token.applied != work.token.staged {
                return Err(StoreError::ParserWorkPending);
            }
            let remains:bool=sql!(path,tx.query_row("SELECT EXISTS(SELECT 1 FROM parser_cell WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4)",params![token.predecessor.input_pk,token.predecessor.chain,token.predecessor.generation,namespace(token.work_id)],|r|r.get(0)))?;
            if remains {
                return Err(StoreError::Invariant(
                    "prepared cursor skipped retained cells".into(),
                ));
            }
            invalidate(
                tx,
                path,
                token.predecessor.input_pk,
                token.predecessor.chain,
            )
        },
    )
}

impl crate::Writer {
    pub fn parser_work(&self, token: &Token) -> Result<Option<Work>> {
        load(&self._index, &self.root.join("index.db"), token)
    }
    pub fn prepare_parser(&mut self, head: &Head, progress: Progress) -> Result<Work> {
        admit(
            &mut self._index,
            &self.root.join("index.db"),
            head,
            progress,
        )
    }
    pub fn stage_parser(
        &mut self,
        token: &WorkToken,
        changes: &[CellChange],
        budget: Budget,
    ) -> Result<Work> {
        stage(
            &mut self._index,
            &self.root.join("index.db"),
            token,
            changes,
            budget,
        )
    }
    pub fn seal_parser(&mut self, token: &WorkToken) -> Result<Work> {
        seal(&mut self._index, &self.root.join("index.db"), token)
    }
    pub fn apply_parser(&mut self, token: &WorkToken, budget: Budget) -> Result<Work> {
        apply(&mut self._index, &self.root.join("index.db"), token, budget)
    }
    pub fn discard_parser_preparation(
        &mut self,
        token: &WorkToken,
        budget: Budget,
    ) -> Result<Option<Work>> {
        discard(&mut self._index, &self.root.join("index.db"), token, budget)
    }
    pub fn publish_parser(&mut self, token: &WorkToken) -> Result<Head> {
        publish(&mut self._index, &self.root.join("index.db"), token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::reopen_after_drop;
    fn budget() -> Budget {
        Budget {
            cells: 1,
            encoded_bytes: 4096,
        }
    }
    fn basis(input_pk: i64, end: u64) -> Basis<'static> {
        Basis {
            input_pk,
            chain: 1,
            parser_version: 1,
            digest: [7; 32],
            acquired_upto: end,
            companions_json: None,
            excisions_seen: 0,
        }
    }
    fn progress() -> Progress {
        Progress {
            inspected_upto: 100,
            parsed_upto: 100,
            phase: Phase::Done,
            finalization_pending: false,
            counts: Accounting {
                framed: 1,
                used: 1,
                ..Accounting::default()
            },
        }
    }
    fn change(key: &str, value: &str) -> CellChange {
        CellChange {
            namespace: "provider".into(),
            key: key.as_bytes().to_vec(),
            value: Some(value.as_bytes().to_vec()),
        }
    }
    fn fixture() -> (tempfile::TempDir, std::path::PathBuf, crate::Writer, Head) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let mut writer = crate::Writer::open(&root).unwrap();
        writer
            ._index
            .execute("INSERT INTO input_ref(id) VALUES('prepared-fixture')", [])
            .unwrap();
        let input = writer
            ._index
            .query_row("SELECT pk FROM input_ref", [], |r| r.get(0))
            .unwrap();
        let head = writer.begin_parser(&basis(input, 100)).unwrap();
        (dir, root, writer, head)
    }
    #[test]
    fn bounded_application_blocks_new_parser_views_but_preserves_old_snapshot_and_newer_suffix() {
        let (_dir, root, mut writer, head) = fixture();
        let old = writer.parser_snapshot(&head.token).unwrap();
        let mut work = writer.prepare_parser(&head, progress()).unwrap();
        assert!(matches!(
            writer.parser_snapshot(&head.token),
            Err(StoreError::ParserWorkPending)
        ));
        assert!(matches!(
            writer.prepare_parser(&head, progress()),
            Err(StoreError::ParserWorkPending)
        ));
        assert!(matches!(
            write_step(
                &mut writer._index,
                &root.join("index.db"),
                &head,
                progress(),
                &[],
                budget(),
                |_| Ok(())
            ),
            Err(StoreError::ParserWorkPending)
        ));
        work = writer
            .stage_parser(&work.token, &[change("a", "one")], budget())
            .unwrap();
        work = writer
            .stage_parser(&work.token, &[change("b", "two")], budget())
            .unwrap();
        assert!(matches!(
            writer.publish_parser(&work.token),
            Err(StoreError::ParserWorkPending)
        ));
        work = writer.seal_parser(&work.token).unwrap();
        let stale = work.token.clone();
        work = writer.apply_parser(&work.token, budget()).unwrap();
        assert_eq!(work.token.applied, 1);
        assert!(matches!(
            writer.apply_parser(&stale, budget()),
            Err(StoreError::StaleParserState)
        ));
        assert_eq!(old.cell("provider", b"a", 100).unwrap(), None);
        assert_eq!(
            writer
                .parser_head(head.token.input_pk, 1)
                .unwrap()
                .unwrap()
                .parsed_upto,
            0
        );
        assert!(matches!(
            writer.parser_snapshot(&head.token),
            Err(StoreError::ParserWorkPending)
        ));
        writer
            .begin_parser(&basis(head.token.input_pk, 200))
            .unwrap();
        work = writer.apply_parser(&work.token, budget()).unwrap();
        let published = writer.publish_parser(&work.token).unwrap();
        assert_eq!(
            (
                published.parsed_upto,
                published.acquired_upto,
                published.status.as_str()
            ),
            (100, 200, "parsing")
        );
        assert_eq!(published.counts.framed, 1);
        assert!(matches!(
            writer.publish_parser(&work.token),
            Err(StoreError::StaleParserState)
        ));
        let new = writer.parser_snapshot(&published.token).unwrap();
        assert_eq!(new.cell("provider", b"a", 100).unwrap().unwrap(), b"one");
        assert_eq!(
            new.neighbor("provider", b"a", true, 100)
                .unwrap()
                .unwrap()
                .0,
            b"b"
        );
        assert_eq!(old.cell("provider", b"b", 100).unwrap(), None);
    }
    #[test]
    fn restart_resumes_staging_and_partial_application_without_reinterpretation() {
        let (_dir, root, mut writer, head) = fixture();
        let work = writer.prepare_parser(&head, progress()).unwrap();
        let work = writer
            .stage_parser(&work.token, &[change("a", "one")], budget())
            .unwrap();
        drop(writer);
        let mut writer = reopen_after_drop(&root);
        let resumed = writer.parser_work(&head.token).unwrap().unwrap();
        assert_eq!(resumed.token, work.token);
        let work = writer
            .stage_parser(&resumed.token, &[change("b", "two")], budget())
            .unwrap();
        let work = writer.seal_parser(&work.token).unwrap();
        let work = writer.apply_parser(&work.token, budget()).unwrap();
        drop(writer);
        let mut writer = reopen_after_drop(&root);
        let resumed = writer.parser_work(&head.token).unwrap().unwrap();
        assert_eq!(resumed.token, work.token);
        assert_eq!(resumed.token.applied, 1);
        assert!(matches!(
            writer.parser_snapshot(&head.token),
            Err(StoreError::ParserWorkPending)
        ));
        let work = writer.apply_parser(&resumed.token, budget()).unwrap();
        let published = writer.publish_parser(&work.token).unwrap();
        assert_eq!(published.status, "done");
        assert_eq!(published.counts.used, 1);
        drop(writer);
        let writer = reopen_after_drop(&root);
        assert!(writer.parser_work(&published.token).unwrap().is_none());
        assert_eq!(
            writer
                .parser_snapshot(&published.token)
                .unwrap()
                .cell("provider", b"b", 100)
                .unwrap()
                .unwrap(),
            b"two"
        );
    }
    #[test]
    fn invalidated_work_cannot_publish_and_prepared_members_have_bounded_gc() {
        let (_dir, root, mut writer, head) = fixture();
        let work = writer.prepare_parser(&head, progress()).unwrap();
        let work = writer
            .stage_parser(&work.token, &[change("a", "one")], budget())
            .unwrap();
        let work = writer.seal_parser(&work.token).unwrap();
        let work = writer.apply_parser(&work.token, budget()).unwrap();
        let mut new_basis = basis(head.token.input_pk, 100);
        new_basis.digest = [8; 32];
        let next = writer.begin_parser(&new_basis).unwrap();
        assert!(matches!(
            writer.publish_parser(&work.token),
            Err(StoreError::StaleParserState)
        ));
        assert_eq!(
            writer
                .parser_snapshot(&next.token)
                .unwrap()
                .cell("provider", b"a", 100)
                .unwrap(),
            None
        );
        let gc = collect_retired(
            &mut writer._index,
            &root.join("index.db"),
            &next.token,
            budget(),
        )
        .unwrap();
        assert_eq!(gc.deleted, 1);
        assert!(!gc.remaining);
        let new_work = writer.prepare_parser(&next, progress()).unwrap();
        assert_ne!(new_work.token.work_id, work.token.work_id);
        assert!(matches!(
            writer.seal_parser(&work.token),
            Err(StoreError::StaleParserState)
        ));
    }
    #[test]
    fn admission_and_seal_keep_invalid_or_unadmitted_changes_out_of_live_state() {
        let (_dir, _root, mut writer, head) = fixture();
        let work = writer.prepare_parser(&head, progress()).unwrap();
        assert!(matches!(
            writer.stage_parser(
                &work.token,
                &[change("a", "one")],
                Budget {
                    cells: 1,
                    encoded_bytes: 3
                }
            ),
            Err(StoreError::ParserAdmission { .. })
        ));
        let reserved = CellChange {
            namespace: namespace(work.token.work_id),
            key: vec![],
            value: Some(vec![]),
        };
        assert!(matches!(
            writer.stage_parser(&work.token, &[reserved], budget()),
            Err(StoreError::Invariant(_))
        ));
        assert_eq!(
            writer
                .parser_work(&head.token)
                .unwrap()
                .unwrap()
                .token
                .staged,
            0
        );
        let work = writer
            .stage_parser(&work.token, &[change("a", "one")], budget())
            .unwrap();
        let work = writer.seal_parser(&work.token).unwrap();
        assert!(matches!(
            writer.stage_parser(&work.token, &[change("b", "two")], budget()),
            Err(StoreError::Invariant(_))
        ));
        assert!(matches!(
            writer.apply_parser(
                &work.token,
                Budget {
                    cells: 1,
                    encoded_bytes: 1
                }
            ),
            Err(StoreError::ParserAdmission { .. })
        ));
        assert_eq!(
            writer
                .parser_work(&head.token)
                .unwrap()
                .unwrap()
                .token
                .applied,
            0
        );
    }

    #[test]
    fn interrupted_unsealed_preparation_is_discarded_in_bounded_steps() {
        let (_dir, root, mut writer, head) = fixture();
        let work = writer.prepare_parser(&head, progress()).unwrap();
        let work = writer
            .stage_parser(&work.token, &[change("a", "one")], budget())
            .unwrap();
        writer
            .stage_parser(&work.token, &[change("b", "two")], budget())
            .unwrap();
        drop(writer);
        let mut writer = reopen_after_drop(&root);
        let work = writer.parser_work(&head.token).unwrap().unwrap();
        let work = writer
            .discard_parser_preparation(&work.token, budget())
            .unwrap()
            .unwrap();
        assert_eq!(work.token.applied, 1);
        assert!(work.discarding);
        assert!(matches!(
            writer.seal_parser(&work.token),
            Err(StoreError::Invariant(_))
        ));
        drop(writer);
        let mut writer = reopen_after_drop(&root);
        let work = writer.parser_work(&head.token).unwrap().unwrap();
        assert!(
            writer
                .discard_parser_preparation(&work.token, budget())
                .unwrap()
                .is_none()
        );
        assert_eq!(
            writer
                .parser_snapshot(&head.token)
                .unwrap()
                .cell("provider", b"a", 100)
                .unwrap(),
            None
        );
        assert_eq!(
            writer
                .parser_head(head.token.input_pk, 1)
                .unwrap()
                .unwrap()
                .parsed_upto,
            0
        );
        let replacement = writer.prepare_parser(&head, progress()).unwrap();
        assert_ne!(replacement.token.work_id, work.token.work_id);
        assert!(matches!(
            writer.seal_parser(&work.token),
            Err(StoreError::StaleParserState)
        ));
    }

    #[test]
    fn process_exit_between_apply_and_publish_keeps_checkpoint_owed() {
        let (_dir, root, mut writer, head) = fixture();
        let work = writer.prepare_parser(&head, progress()).unwrap();
        let work = writer
            .stage_parser(&work.token, &[change("a", "one")], budget())
            .unwrap();
        let work = writer
            .stage_parser(&work.token, &[change("b", "two")], budget())
            .unwrap();
        writer.seal_parser(&work.token).unwrap();
        drop(writer);
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "parser_state::prepared::tests::prepared_crash_child",
                "--nocapture",
            ])
            .env("POLYLOGUE_PREPARED_CRASH_ROOT", &root)
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(73));
        let mut writer = crate::Writer::open(&root).unwrap();
        let resumed = writer.parser_work(&head.token).unwrap().unwrap();
        assert_eq!(resumed.token.applied, 1);
        assert_eq!(
            writer
                .parser_head(head.token.input_pk, 1)
                .unwrap()
                .unwrap()
                .parsed_upto,
            0
        );
        assert!(matches!(
            writer.parser_snapshot(&head.token),
            Err(StoreError::ParserWorkPending)
        ));
        let completed = writer.apply_parser(&resumed.token, budget()).unwrap();
        let head = writer.publish_parser(&completed.token).unwrap();
        assert_eq!(head.counts.framed, 1);
        assert_eq!(head.parsed_upto, 100);
    }

    #[test]
    fn prepared_crash_child() {
        let Some(root) = std::env::var_os("POLYLOGUE_PREPARED_CRASH_ROOT") else {
            return;
        };
        let mut writer = crate::Writer::open(std::path::Path::new(&root)).unwrap();
        let input = writer
            ._index
            .query_row("SELECT pk FROM input_ref", [], |r| r.get(0))
            .unwrap();
        let head = writer.parser_head(input, 1).unwrap().unwrap();
        let work = writer.parser_work(&head.token).unwrap().unwrap();
        writer.apply_parser(&work.token, budget()).unwrap();
        std::process::exit(73);
    }
}
