//! Apply worker-resolved event and link values and their compact memberships.
use super::*;

impl RowWriter<'_, '_> {
    pub(super) fn refresh_event(&self, pk: i64) -> Result<()> {
        self.tx.prepare_cached("UPDATE session SET changed=?2 WHERE pk=(SELECT session_pk FROM event WHERE pk=?1) AND changed IS NOT ?2")?.execute(params![pk,self.counter])?;
        let choice = self
            .choices
            .events
            .get(&pk)
            .ok_or_else(|| StoreError::Invalid("unresolved event version".into()))?;
        let Some(choice) = choice else {
            self.tx.execute("DELETE FROM event WHERE pk=?1", [pk])?;
            return Ok(());
        };
        self.tx
            .prepare_cached("UPDATE session SET changed=?2 WHERE pk=?1 AND changed IS NOT ?2")?
            .execute(params![choice.session, self.counter])?;
        let after = choice
            .after
            .as_ref()
            .map(|id| self.message_pk(id))
            .transpose()?
            .flatten();
        let seq = match choice.seq {
            Some(seq) => seq,
            None => after
                .map(|pk| {
                    self.tx
                        .query_row("SELECT seq FROM message WHERE pk=?1", [pk], |row| {
                            row.get(0)
                        })
                })
                .transpose()?
                .unwrap_or(0),
        };
        self.tx.prepare_cached("UPDATE event SET session_pk=?2,seq=?3,after_message_pk=?4,at=?5,input_pk=?6,source_chain=?7,source_start=?8,source_end=?9,data=?10,after_message_id=?11,seq_from_message=?12,kind=?13,fields=?14 WHERE pk=?1")?.execute(params![pk,choice.session,seq,after,choice.at,choice.input,choice.chain,choice.start,choice.end,choice.body,choice.after,choice.seq.is_none(),choice.kind,choice.fields])?;
        Ok(())
    }
    pub(super) fn refresh_link(&self, pk: i64) -> Result<()> {
        self.mark_link_sessions(pk)?;
        let choice = self
            .choices
            .links
            .get(&pk)
            .ok_or_else(|| StoreError::Invalid("unresolved link version".into()))?;
        let Some(choice) = choice else {
            self.tx.execute("DELETE FROM link WHERE pk=?1", [pk])?;
            return Ok(());
        };
        self.tx.prepare_cached("UPDATE link SET branch_point=?2,spawning_call=?3,evidence=?4,input_pk=?5,source_chain=?6 WHERE pk=?1")?.execute(params![pk,choice.link.at.as_ref().map(ToString::to_string),choice.link.call_id,choice.link.evidence,choice.input,choice.chain])?;
        Ok(())
    }
    fn mark_link_sessions(&self, pk: i64) -> Result<()> {
        self.tx.prepare_cached("UPDATE session SET changed=?2 WHERE id IN (SELECT parent_id FROM link WHERE pk=?1 UNION SELECT child_id FROM link WHERE pk=?1) AND changed IS NOT ?2")?.execute(params![pk,self.counter])?;
        Ok(())
    }
    pub(super) fn store_link(&self, link: &Link, index: usize) -> Result<()> {
        let pk = self
            .keys
            .get(resolve::Kind::Link, &resolve::link_key(link)?)?
            .pk;
        let child = self
            .keys
            .get(resolve::Kind::Session, link.child.to_string().as_bytes())?
            .pk;
        let (kind, how) = link.kind.columns();
        self.tx.prepare_cached("INSERT OR IGNORE INTO link(pk,parent_id,child_id,child_pk,kind,how,evidence,input_pk,source_chain) VALUES (?1,?2,?3,(SELECT pk FROM session WHERE pk=?4),?5,?6,?7,?8,?9)")?.execute(params![pk,link.parent.to_string(),link.child.to_string(),child,kind,how,link.evidence,self.input_pk,self.chain])?;
        let value = &self.choices.link_values[index];
        self.tx
            .prepare_cached("INSERT OR IGNORE INTO link_value(pk,hash,body) VALUES (?1,?2,?3)")?
            .execute(params![value.key.pk, value.hash.as_slice(), value.body])?;
        self.tx.prepare_cached("INSERT INTO link_source(link_pk,input_pk,chain,value_pk,arrival_rank) VALUES (?1,?2,?3,?4,?5) ON CONFLICT(link_pk,input_pk,chain) DO UPDATE SET value_pk=excluded.value_pk,arrival_rank=excluded.arrival_rank")?.execute(params![pk,self.input_pk,self.chain,value.key.pk,self.rank])?;
        Ok(())
    }
    pub(super) fn store_event_value(&self, index: usize) -> Result<()> {
        let value = &self.choices.event_values[index];
        let (pk, source) = &self.choices.event_sources[index];
        self.tx
            .prepare_cached("INSERT OR IGNORE INTO event_value(pk,hash,body) VALUES (?1,?2,?3)")?
            .execute(params![value.key.pk, value.hash.as_slice(), value.body])?;
        let imported = &source.imported;
        let admitted = source.admitted.or(self.source_admitted);
        self.tx.prepare_cached("INSERT INTO event_source(event_pk,input_pk,chain,value_pk,arrival_rank,session_pk,seq,source_start,source_end,after_message_id,at,fields,imported_provenance,admitted)
            VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)
            ON CONFLICT(event_pk,input_pk,chain) DO UPDATE SET value_pk=excluded.value_pk,arrival_rank=excluded.arrival_rank,session_pk=excluded.session_pk,seq=excluded.seq,source_start=excluded.source_start,source_end=excluded.source_end,after_message_id=excluded.after_message_id,at=excluded.at,fields=excluded.fields,imported_provenance=excluded.imported_provenance,admitted=excluded.admitted")?
            .execute(params![pk,self.input_pk,self.chain,value.key.pk,self.rank,source.session,source.seq,source.start,source.end,source.after,source.at,source.fields,imported,admitted])?;
        Ok(())
    }
}
