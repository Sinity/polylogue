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
            if let RowBasis::Native { columns, .. } = &selection.program.row_basis {
                return self.selection_facts_scoped(selection, &mut |fact: SelectionFactKey| {
                    let row = Value::Object(columns.iter().cloned().zip(fact.values).collect());
                    let reference = row_reference(&selection.program, &row).ok_or_else(|| {
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
                selected.require_complete()?;
                fold(hash, &selected.fingerprint()?)?;
                fold(hash, &selected.attachment_fingerprint()?)
            }
            Reference::Message(_) | Reference::Block { .. } | Reference::MessageRange { .. } => {
                let mut params = Vec::new();
                let members = membership(Unit::Messages, reference, &mut params)?.ok_or_else(|| invalid("not_built", "reference has no message membership"))?;
                let sql = format!("SELECT m.id,m.order_input_id,m.seq,m.record_position,m.at_time,s.version_hash,p.layout FROM message m LEFT JOIN message_source s ON s.message_pk=m.pk AND s.shown=1 LEFT JOIN message_projection p ON p.version_hash=s.version_hash WHERE m.pk IN (SELECT pk FROM ({members})) ORDER BY m.order_input_id,m.seq,m.record_position,m.id");
                self.fold_content_rows(&sql, &params, "version_hash", hash)
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
    fn fold_content_rows(
        &self,
        sql: &str,
        params: &[SqlValue],
        required: &str,
        hash: &mut blake3::Hasher,
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
            fold(hash, &row)
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
