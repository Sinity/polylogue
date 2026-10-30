//! Cursor content bindings fold stored versions without loading record bodies.
use super::*;

fn fold(hash: &mut blake3::Hasher, value: &impl serde::Serialize) -> Result<(), SelectionError> {
    let bytes = serde_json::to_vec(value).map_err(StoreError::from)?;
    hash.update(&(bytes.len() as u64).to_le_bytes());
    hash.update(&bytes);
    Ok(())
}
impl Reader<'_> {
    /// Uses the same prepared selection, its total answer and captured ranking
    /// scope. The caller restores an authenticated clock before this check.
    pub fn selection_content_fingerprint(
        &self,
        selection: &PreparedSelection<'_>,
        with_parents: bool,
        near: Option<&NearFn<'_>>,
    ) -> Result<String, SelectionError> {
        let mut hash = blake3::Hasher::new();
        hash.update(b"polylogue.read.selected-content.v1\0");
        self.in_selection_context(selection, near, || {
            if let Some(reference) = &selection.direct {
                return self.fingerprint_reference(reference, with_parents, &mut hash);
            }
            if matches!(
                selection.program.unit,
                Unit::Sessions
                    | Unit::Messages
                    | Unit::Documents
                    | Unit::Events
                    | Unit::Blocks
                    | Unit::ToolCalls
            ) && let RowBasis::Native { .. } = &selection.program.row_basis
            {
                return self.selection_facts_scoped(selection, &mut |fact: SelectionFactKey| {
                    let reference = fact.reference().ok_or_else(|| {
                        invalid(
                            "not_yet_available",
                            "native row has no retained reference identity",
                        )
                    })?;
                    self.fingerprint_reference(&reference, with_parents, &mut hash)?;
                    Ok(ControlFlow::Continue(()))
                });
            }
            self.query_each(
                &selection.program.sql,
                &values(&selection.program.params),
                |mut row| {
                    selection.scope.check()?;
                    if let Some(fields) = row.as_object_mut() {
                        fields.retain(|name, _| !name.starts_with('_'));
                    }
                    fold(&mut hash, &row)
                },
            )
        })?;
        selection.scope.check()?;
        Ok(hash.finalize().to_hex().to_string())
    }

    fn fingerprint_reference(
        &self,
        reference: &Reference,
        with_parents: bool,
        hash: &mut blake3::Hasher,
    ) -> Result<(), SelectionError> {
        fold(hash, &reference.to_string())?;
        match reference {
            Reference::Span { container, .. } => self.fingerprint_reference(container, with_parents, hash),
            Reference::Session(id) => {
                let selected = self.select_messages(id, with_parents)?;
                // A structured read binds the available content together with
                // its exact parent issues; strict full-history callers retain
                // MessageSelection::require_complete at their own boundary.
                fold(hash, &selected.fingerprint()?)?;
                fold(hash, &selected.attachment_fingerprint()?)?;
                selected.visit_available_tool_message_ids(|id| {
                    let message = MessageId::parse(id).map_err(|error| StoreError::Invalid(error.to_string()))?;
                    self.fold_message_tool_pairs(&message, None, hash)
                })?;
                Ok(())
            }
            Reference::Message(_) | Reference::Block { .. } | Reference::MessageRange { .. } => {
                let mut params = Vec::new();
                let members = membership(Unit::Messages, reference, &mut params)?.ok_or_else(|| invalid("not_built", "reference has no message membership"))?;
                let sql = format!("SELECT m.id,m.order_input_id,m.seq,m.record_position,m.at_time,s.version_hash,p.layout FROM message m LEFT JOIN message_source s ON s.message_pk=m.pk AND s.shown=1 LEFT JOIN message_projection p ON p.version_hash=s.version_hash WHERE m.pk IN (SELECT pk FROM ({members})) ORDER BY m.order_input_id,m.seq,m.record_position,m.id");
                self.fold_content_rows_with(&sql, &params, "version_hash", hash, |row, hash| {
                    let id = row["id"].as_str().ok_or_else(|| invalid("not_yet_available", "selected message has no retained identity"))?;
                    let id = MessageId::parse(id).map_err(|error| StoreError::Invalid(error.to_string()))?;
                    let ordinal = match reference { Reference::Block { n, .. } => Some(*n), _ => None };
                    self.fold_message_tool_pairs(&id, ordinal, hash)?;
                    Ok(())
                })
            }
            Reference::Document(id) => {
                self.fold_content_rows("SELECT d.id,s.version_hash,s.time FROM document d LEFT JOIN document_source s ON s.document_pk=d.pk AND s.shown=1 WHERE d.id=?1 ORDER BY s.input_pk,s.chain", &[SqlValue::Text(id.to_string())], "version_hash", hash)
            }
            Reference::Event(id) => {
                self.fold_content_rows("SELECT e.id,v.hash,s.fields FROM event e LEFT JOIN event_source s ON s.event_pk=e.pk AND s.input_pk=e.input_pk AND s.chain=e.source_chain LEFT JOIN event_value v ON v.pk=s.value_pk WHERE e.id=?1", &[SqlValue::Text(id.clone())], "hash", hash)
            }
            _ => Err(invalid("not_built", "selected reference has no retained content version")),
        }
    }
    fn fold_message_tool_pairs(
        &self,
        message: &MessageId,
        ordinal: Option<u64>,
        hash: &mut blake3::Hasher,
    ) -> crate::Result<()> {
        self.check()?;
        let exact = if ordinal.is_some() {
            "AND coalesce(b.ordinal,cast(b.idx AS TEXT))=?2"
        } else {
            ""
        };
        // Read only call/result positions, not every plain block or its payload.
        let sql = format!("SELECT coalesce(b.ordinal,cast(b.idx AS TEXT)) FROM message m JOIN block b ON b.message_pk=m.pk
            WHERE m.id=?1 AND m.duplicate_of IS NULL AND b.kind IN ('tool_call','tool_result') {exact}
            AND EXISTS(SELECT 1 FROM tool_call t WHERE t.call_block_pk=b.pk OR t.result_block_pk=b.pk) ORDER BY b.idx");
        let mut statement = self.conn()?.prepare_cached(&sql)?;
        let mut rows = if let Some(ordinal) = ordinal {
            statement.query(rusqlite::params![message.to_string(), ordinal.to_string()])?
        } else {
            statement.query([message.to_string()])?
        };
        let mut ordinals = Vec::new();
        while let Some(row) = rows.next()? {
            self.check()?;
            let ordinal: String = row.get(0)?;
            ordinals.push(
                ordinal.parse::<u64>().map_err(|_| {
                    StoreError::Invalid("invalid original tool block ordinal".into())
                })?,
            );
        }
        if !ordinals.is_empty() {
            let fingerprint = self.message_tool_pair_fingerprint(message, &ordinals)?;
            hash.update(b"tool-pair-dependencies\0");
            hash.update(&fingerprint.0);
        }
        self.check()
    }

    fn fold_content_rows(
        &self,
        sql: &str,
        params: &[SqlValue],
        required: &str,
        hash: &mut blake3::Hasher,
    ) -> Result<(), SelectionError> {
        self.fold_content_rows_with(sql, params, required, hash, |_, _| Ok(()))
    }
    fn fold_content_rows_with(
        &self,
        sql: &str,
        params: &[SqlValue],
        required: &str,
        hash: &mut blake3::Hasher,
        mut extra: impl FnMut(&serde_json::Value, &mut blake3::Hasher) -> Result<(), SelectionError>,
    ) -> Result<(), SelectionError> {
        let mut found = false;
        self.query_each(sql, params, |row| {
            if row[required].is_null() {
                return Err(invalid(
                    "not_yet_available",
                    "selected content has no canonical source version",
                ));
            }
            found = true;
            fold(hash, &row)?;
            extra(&row, hash)
        })?;
        if !found {
            return Err(invalid(
                "not_found",
                "selected content is no longer present",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "content/tests.rs"]
mod tests;
