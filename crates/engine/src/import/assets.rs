//! Derive export associations from stored observations and sidecars.
use super::*;
use polylogue_store::assets::AssetRow;
use polylogue_store::export_items::ExportItems;

fn stage(items: &mut ExportItems, row: &AssetRow) -> polylogue_store::Result<()> {
    items.push(0..0, Vec::new(), None, &serde_json::to_vec(row)?)
}

impl Engine {
    pub(crate) fn refresh_import_assets(&mut self, pass: &mut Pass) -> Result<(), EngineError> {
        let (members, through) = self.reads.reader()?.unindexed_import_members()?;
        let mut staged = ExportItems::new()?;
        for source in members {
            let source_id =
                polylogue_sources::chatgpt::assets::member_id(&source.membership.member)
                    .map(str::to_owned);
            let sidecar =
                polylogue_sources::chatgpt::assets::sidecar_kind(&source.membership.member)
                    .map(str::to_owned);
            stage(
                &mut staged,
                &AssetRow::Member {
                    source,
                    source_id,
                    sidecar,
                },
            )?;
        }
        if let Some(through) = through {
            stage(&mut staged, &AssetRow::Scan { through })?;
            self.writer.import_assets(staged)?;
        }

        let sidecars = self.reads.reader()?.pending_asset_sidecars()?;
        for (input, member) in sidecars {
            let reader = self.reads.reader()?;
            let Some(progress) = reader.input_progress(&input)? else {
                continue;
            };
            let source = reader
                .asset_source(&input)?
                .ok_or_else(|| std::io::Error::other("a stored sidecar has no current revision"))?;
            // Bytes still being written are read in a later pass.
            let chain = match reader.stored_chain(&input, progress.chain, progress.length) {
                Ok(chain) => chain,
                Err(polylogue_store::StoreError::InputPending { .. }) => continue,
                Err(error) => return Err(error.into()),
            };
            let mut staged = ExportItems::new()?;
            stage(
                &mut staged,
                &AssetRow::ResetMetadata {
                    input: input.clone(),
                    sidecar: member.clone(),
                },
            )?;
            staged.begin_scan()?;
            let mut ordinal = 0;
            let result = polylogue_sources::chatgpt::assets::metadata(
                &member,
                chain.reader(Arc::clone(&self.blobs)),
                |metadata| {
                    stage(
                        &mut staged,
                        &AssetRow::Metadata {
                            input: input.clone(),
                            sidecar: member.clone(),
                            ordinal,
                            source_id: metadata.source_id,
                            name: metadata.name,
                            media_type: metadata.media_type,
                            size: metadata.size,
                        },
                    )
                    .map_err(|e| e.to_string())?;
                    ordinal += 1;
                    Ok(())
                },
            );
            let error = match result {
                Ok(_) => {
                    staged.commit_scan()?;
                    None
                }
                Err(error) => {
                    staged.rollback_scan()?;
                    let error = format!("asset sidecar {input} ({member}): {error}");
                    pass.errors.push(error.clone());
                    Some(error)
                }
            };
            drop(reader);
            stage(
                &mut staged,
                &AssetRow::Sidecar {
                    input,
                    sidecar: member,
                    source_version: source.version,
                    error,
                },
            )?;
            self.writer.import_assets(staged)?;
        }
        Ok(())
    }
}
