//! Derived member metadata stays local; immutable bindings require certified publication.
use super::*;
use crate::assets::AssetRow;

impl Inner {
    pub(super) fn install_assets(
        &mut self,
        mut staged: crate::export_items::ExportItems,
    ) -> Result<()> {
        if self.sinex.is_none() {
            return crate::assets::install(&mut self.index, &self.archive, staged);
        }
        // Private custody accepts a frozen body even while the cache cannot
        // authorize any read, association or publication.
        let ready = self.primary_cache_ready();
        let sinex = Arc::clone(self.sinex.as_ref().expect("primary asset writer"));
        let mut local = crate::export_items::ExportItems::new()?;
        let mut ordinal = 0_u64;
        let mut missing = None;
        let cache = crate::primary::cache_identity(&self.index)?;
        let generation: String =
            self.index
                .query_row("SELECT value FROM meta WHERE key='generation'", [], |row| {
                    row.get(0)
                })?;
        let tx = self
            .index
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        while let Some(item) = staged.next_item()? {
            let row: AssetRow = serde_json::from_reader(item.reader)?;
            match row {
                AssetRow::Binding { record } => {
                    record.validate()?;
                    if ready.is_err() || !crate::assets::bindings::known(&tx, &record)? {
                        // This identity retains a private claim and its original
                        // witness. It confers no admission or publication authority.
                        let identity = format!(
                            "binding-custody:{}",
                            blake3::hash(&serde_json::to_vec(&(
                                "polylogue.binding.custody.v1",
                                &cache,
                                &generation,
                                record.selection(),
                            ))?)
                            .to_hex()
                        );
                        missing.get_or_insert_with(|| record.context().message.to_string());
                        let write = crate::primary::ProductWrite {
                            product: crate::primary::Product::AttachmentBinding {
                                record: Box::new(record),
                            },
                            location: None,
                            parents: Vec::new(),
                            machine: None,
                            output_index: 0,
                        };
                        let receipt = sinex.stage_canonical_products(
                            &identity,
                            std::slice::from_ref(&write),
                            None,
                        )?;
                        if receipt.identity.is_empty() {
                            return Err(StoreError::Invalid(
                                "durable canonical custody has no body reference".into(),
                            ));
                        }
                        crate::primary::derived::stage_products(&tx, &identity, &[write], None)?;
                        tx.execute(
                            "UPDATE primary_derived SET custody=?2 WHERE identity=?1",
                            params![identity, receipt.identity],
                        )?;
                    }
                }
                row => {
                    let bytes = serde_json::to_vec(&row)?;
                    local.push(item.range, ordinal.to_be_bytes().to_vec(), None, &bytes)?;
                    ordinal = ordinal.checked_add(1).ok_or_else(|| {
                        StoreError::Invalid("asset metadata order overflow".into())
                    })?;
                }
            }
        }
        tx.commit()?;
        ready?;
        crate::assets::install(&mut self.index, &self.archive, local)?;
        if let Some(message) = missing {
            return Err(StoreError::ReferenceUnavailable {
                reference: format!("attachment-binding:{message}"),
                reason: "canonical receiving admission and owner publication proof are unavailable"
                    .into(),
            });
        }
        Ok(())
    }
}

impl Inner {
    pub(super) fn restore_canonical_custody(&mut self) -> Result<()> {
        let Some(sinex) = self.sinex.as_ref().map(Arc::clone) else {
            return Ok(());
        };
        let tx = self
            .index
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        sinex.visit_canonical_custody(&mut |custody| {
            crate::primary::derived::restore_custody(&tx, custody)
        })?;
        tx.commit()?;
        Ok(())
    }

    pub(super) fn revoke_canonical_custody(&self) -> Result<()> {
        let Some(sinex) = &self.sinex else {
            return Ok(());
        };
        crate::primary::derived::hold(&self.index, sinex.as_ref())?;
        sinex.revoke_canonical_custody()
    }
}
