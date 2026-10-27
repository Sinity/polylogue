//! Event and link values shared by compact per-chain membership rows.
use super::*;
use polylogue_model::MessageId;
use serde::Serialize;

impl RowWriter<'_, '_> {
    pub(super) fn refresh_event(&self, pk: i64) -> Result<()> {
        self.tx.prepare_cached("UPDATE session SET changed=?2 WHERE pk=(SELECT session_pk FROM event WHERE pk=?1) AND changed IS NOT ?2")?
            .execute(params![pk,self.counter])?;
        let winner=self.tx.prepare_cached(
            "SELECT s.input_pk,s.chain,s.session_pk,s.seq,s.source_start,s.source_end,s.after_message_id,s.at,v.body
             FROM event_source s JOIN input_ref i ON i.pk=s.input_pk JOIN event_value v ON v.pk=s.value_pk
             WHERE s.event_pk=?1 ORDER BY s.arrival_rank,i.id,s.chain DESC LIMIT 1"
        )?.query_row([pk],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,Option<i64>>(2)?,r.get::<_,Option<i64>>(3)?,r.get::<_,Option<i64>>(4)?,r.get::<_,Option<i64>>(5)?,r.get::<_,Option<String>>(6)?,r.get::<_,Option<i64>>(7)?,r.get::<_,Vec<u8>>(8)?))).optional()?;
        let Some((input, chain, session, seq, start, end, after_id, at, data)) = winner else {
            self.tx.execute("DELETE FROM event WHERE pk=?1", [pk])?;
            return Ok(());
        };
        self.tx
            .prepare_cached("UPDATE session SET changed=?2 WHERE pk=?1 AND changed IS NOT ?2")?
            .execute(params![session, self.counter])?;
        let after = after_id
            .as_ref()
            .map(|id| self.message_pk(id))
            .transpose()?
            .flatten();
        let seq_from_message = seq.is_none();
        let seq = match seq {
            Some(seq) => seq,
            None => after
                .map(|pk| {
                    self.tx
                        .query_row("SELECT seq FROM message WHERE pk=?1", [pk], |r| r.get(0))
                })
                .transpose()?
                .unwrap_or(0),
        };
        self.tx.prepare_cached("UPDATE event SET session_pk=?2,seq=?3,after_message_pk=?4,at=?5,input_pk=?6,source_chain=?7,source_start=?8,source_end=?9,data=?10,after_message_id=?11,seq_from_message=?12 WHERE pk=?1")?
            .execute(params![pk,session,seq,after,at,input,chain,start,end,data,after_id,seq_from_message])?;
        Ok(())
    }

    fn store_assertion<T: Serialize>(&self, kind: &str, pk: i64, value: &T) -> Result<()> {
        let mut sorted = serde_json::to_value(value)?;
        sorted.sort_all_objects();
        let bytes = serde_json::to_vec(&sorted)?;
        let hash = blake3::hash(&bytes);
        let body = crate::encoding::RowPacker::new()?.pack_json(&sorted)?;
        self.tx
            .prepare_cached(&format!(
                "INSERT OR IGNORE INTO {kind}_value(hash,body) VALUES (?1,?2)"
            ))?
            .execute(params![hash.as_bytes().as_slice(), body])?;
        self.tx.prepare_cached(&format!(
            "INSERT INTO {kind}_source({kind}_pk,input_pk,chain,value_pk,arrival_rank)
             VALUES (?1,?2,?3,(SELECT pk FROM {kind}_value WHERE hash=?4),?5)
             ON CONFLICT({kind}_pk,input_pk,chain) DO UPDATE SET value_pk=excluded.value_pk,arrival_rank=excluded.arrival_rank"
        ))?.execute(params![pk,self.input_pk,self.chain,hash.as_bytes().as_slice(),self.rank])?;
        Ok(())
    }

    pub(super) fn refresh_link(&self, pk: i64) -> Result<()> {
        let winner = self.tx.prepare_cached(
            "SELECT s.input_pk,s.chain,v.body FROM link_source s JOIN input_ref i ON i.pk=s.input_pk
             JOIN link_value v ON v.pk=s.value_pk WHERE s.link_pk=?1
             ORDER BY s.arrival_rank,i.id,s.chain DESC LIMIT 1"
        )?.query_row([pk], |r| Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,Vec<u8>>(2)?))).optional()?;
        let Some((input, chain, body)) = winner else {
            self.tx.execute("DELETE FROM link WHERE pk=?1", [pk])?;
            return Ok(());
        };
        let mut link: Link = crate::encoding::unpack_json(&body)?;
        let mut statement = self.tx.prepare_cached(
            "SELECT DISTINCT v.pk,v.body FROM link_source s JOIN link_value v ON v.pk=s.value_pk WHERE s.link_pk=?1",
        )?;
        let values = statement.query_map([pk], |r| r.get::<_, Vec<u8>>(1))?;
        for body in values {
            let other: Link = crate::encoding::unpack_json(&body?)?;
            link.at = link.at.into_iter().chain(other.at).min();
            link.call_id = link.call_id.into_iter().chain(other.call_id).min();
        }
        self.tx.prepare_cached("UPDATE link SET branch_point=?2,spawning_call=?3,evidence=?4,input_pk=?5,source_chain=?6 WHERE pk=?1")?
            .execute(params![pk,link.at.as_ref().map(ToString::to_string),link.call_id,link.evidence,input,chain])?;
        Ok(())
    }

    pub(super) fn store_link(&self, link: &Link) -> Result<()> {
        let (kind, how) = link.kind.columns();
        let child = link.child.to_string();
        self.tx.prepare_cached("INSERT OR IGNORE INTO link(parent_id,child_id,child_pk,kind,how,evidence,input_pk,source_chain)
            VALUES (?1,?2,(SELECT pk FROM session WHERE id=?2),?3,?4,?5,?6,?7)")?
            .execute(params![link.parent.to_string(),child,kind,how,link.evidence,self.input_pk,self.chain])?;
        let pk = self
            .tx
            .prepare_cached(
                "SELECT pk FROM link WHERE parent_id=?1 AND child_id=?2 AND kind=?3 AND how=?4",
            )?
            .query_row(params![link.parent.to_string(), child, kind, how], |r| {
                r.get::<_, i64>(0)
            })?;
        let previous: Option<Vec<u8>> = self.tx.prepare_cached("SELECT v.body FROM link_source s JOIN link_value v ON v.pk=s.value_pk WHERE s.link_pk=?1 AND s.input_pk=?2 AND s.chain=?3")?
            .query_row(params![pk,self.input_pk,self.chain], |r| r.get(0)).optional()?;
        let mut merged = link.clone();
        if let Some(body) = previous {
            let other: Link = crate::encoding::unpack_json(&body)?;
            merged.at = merged.at.into_iter().chain(other.at).min();
            merged.call_id = merged.call_id.into_iter().chain(other.call_id).min();
            merged.evidence = merged.evidence.min(other.evidence);
        }
        let (rank, input, chain, branch, call): (Option<i64>, String, i64, Option<String>, Option<String>) = self.tx.prepare_cached(
            "SELECT s.arrival_rank,i.id,l.source_chain,l.branch_point,l.spawning_call FROM link l
             JOIN input_ref i ON i.pk=l.input_pk LEFT JOIN link_source s
             ON s.link_pk=l.pk AND s.input_pk=l.input_pk AND s.chain=l.source_chain WHERE l.pk=?1",
        )?.query_row([pk], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?)))?;
        let wins = rank.is_none_or(|rank| {
            (self.rank, self.input_id, std::cmp::Reverse(self.chain))
                <= (rank, input.as_str(), std::cmp::Reverse(chain))
        });
        self.store_assertion("link", pk, &merged)?;
        if input == self.input_id
            && chain == self.chain
            && rank.is_some_and(|rank| rank != self.rank)
        {
            return self.refresh_link(pk);
        }
        // Adding a statement can only lower these merged fields. Retraction
        // recomputes them from the surviving statements instead.
        let branch = branch
            .as_deref()
            .map(MessageId::parse)
            .transpose()
            .map_err(|error| StoreError::Invalid(error.to_string()))?;
        let at = merged.at.into_iter().chain(branch).min();
        let call = merged.call_id.into_iter().chain(call).min();
        self.tx
            .prepare_cached(
                "UPDATE link SET branch_point=?2,spawning_call=?3,
            evidence=CASE WHEN ?4 THEN ?5 ELSE evidence END,
            input_pk=CASE WHEN ?4 THEN ?6 ELSE input_pk END,
            source_chain=CASE WHEN ?4 THEN ?7 ELSE source_chain END WHERE pk=?1",
            )?
            .execute(params![
                pk,
                at.as_ref().map(ToString::to_string),
                call,
                wins,
                merged.evidence,
                self.input_pk,
                self.chain
            ])?;
        Ok(())
    }

    pub(super) fn store_event_value(
        &self,
        pk: i64,
        event: &Event,
        data: &[u8],
        seq: Option<i64>,
    ) -> Result<()> {
        let mut value: Json = crate::encoding::unpack_json(data)?;
        value.sort_all_objects();
        let hash = blake3::hash(&serde_json::to_vec(&value)?);
        let body = crate::encoding::RowPacker::new()?.pack_json(&value)?;
        self.tx
            .prepare_cached("INSERT OR IGNORE INTO event_value(hash,body) VALUES (?1,?2)")?
            .execute(params![hash.as_bytes().as_slice(), body])?;
        let session = event
            .session
            .as_ref()
            .map(|id| self.session_pk(&mut HashMap::new(), id))
            .transpose()?;
        self.tx.prepare_cached("INSERT INTO event_source(event_pk,input_pk,chain,value_pk,arrival_rank,session_pk,seq,source_start,source_end,after_message_id,at)
            VALUES (?1,?2,?3,(SELECT pk FROM event_value WHERE hash=?4),?5,?6,?7,?8,?9,?10,?11)
            ON CONFLICT(event_pk,input_pk,chain) DO UPDATE SET value_pk=excluded.value_pk,arrival_rank=excluded.arrival_rank,
                session_pk=excluded.session_pk,seq=excluded.seq,source_start=excluded.source_start,source_end=excluded.source_end,after_message_id=excluded.after_message_id,at=excluded.at")?
            .execute(params![pk,self.input_pk,self.chain,hash.as_bytes().as_slice(),self.rank,session,seq,event.range.map(|r|r.0 as i64),event.range.map(|r|r.1 as i64),event.after.as_ref().map(ToString::to_string),ms(event.at)])?;
        self.refresh_event(pk)
    }
}
