//! Derived member metadata stays local; immutable binding facts need confirmation.
use super::*;
use crate::assets::AssetRow;
use crate::primary::{Product, ProductWrite};

impl Inner {
    pub(super) fn install_assets(&mut self, mut staged: crate::export_items::ExportItems) -> Result<()> {
        let Some(sinex) = self.sinex.as_ref().cloned() else {
            return crate::assets::install(&mut self.index, &self.archive, staged);
        };
        self.primary_cache_ready()?;
        let mut local = crate::export_items::ExportItems::new()?;
        let mut ordinal = 0_u64;
        let mut pending = None;
        while let Some(item) = staged.next_item()? {
            let row: AssetRow = serde_json::from_reader(item.reader)?;
            match row {
                AssetRow::Binding { record } => {
                    if crate::assets::bindings::known(&self.index, &record)? { continue; }
                    let bytes = serde_json::to_vec(&record)?;
                    let mut hash = blake3::Hasher::new();
                    hash.update(b"polylogue-attachment-binding-v1\0");
                    hash.update(&bytes);
                    let identity = format!("attachment-binding:{}", hash.finalize().to_hex());
                    let prepared = sinex.prepare_named_products(&identity, &[ProductWrite {
                        product: Product::AttachmentBinding { record: Box::new(record) },
                        location: None, parents: Vec::new(), machine: None, output_index: 0,
                    }])?;
                    if prepared.pending { pending = Some(prepared.identity); }
                }
                row => {
                    let bytes = serde_json::to_vec(&row)?;
                    local.push(item.range, ordinal.to_be_bytes().to_vec(), None, &bytes)?;
                    ordinal = ordinal.checked_add(1).ok_or_else(|| StoreError::Invalid("asset metadata order overflow".into()))?;
                }
            }
        }
        crate::assets::install(&mut self.index, &self.archive, local)?;
        let Some(identity) = pending else { return Ok(()); };
        match self.poll_primary() {
            Ok(_) | Err(StoreError::Sinex(_)) => Err(StoreError::PrimaryPending { identity }),
            Err(error) => Err(error),
        }
    }
}
