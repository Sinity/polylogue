//! Derive export associations from stored observations and sidecars.
use super::*;
use polylogue_sources::origin::export_assets;
use polylogue_store::assets::AssetRow;
use polylogue_store::export_items::ExportItems;

fn stage(items: &mut ExportItems, row: &AssetRow) -> polylogue_store::Result<()> {
    items.push(0..0, Vec::new(), None, &serde_json::to_vec(row)?)
}

impl Engine {
    /// Installs the export associations not yet indexed: into `rebuild`
    /// when one is given, otherwise into the installed index. With neither
    /// there is nothing to install into; the rebuild installs them before it
    /// assembles.
    pub(crate) fn refresh_import_assets(
        &mut self,
        pass: &mut Pass,
        mut rebuild: Option<&mut polylogue_store::index::rebuild::Rebuild>,
    ) -> Result<(), EngineError> {
        if rebuild.is_none() && !self.archive.index_files()?.serves() {
            return Ok(());
        }
        let cut = rebuild.as_ref().map(|rebuild| rebuild.position());
        let mut install = |engine: &Self, staged: ExportItems| -> Result<(), EngineError> {
            match rebuild.as_deref_mut() {
                Some(rebuild) => rebuild.import_assets(staged)?,
                None => engine.writer.import_assets(staged)?,
            }
            Ok(())
        };
        let (members, through) = {
            let reader = self.reads.reader()?;
            match cut {
                Some(cut) => reader.import_members_through(cut)?,
                None => reader.unindexed_import_members()?,
            }
        };
        let mut staged = ExportItems::new()?;
        for source in members {
            let member = &source.membership.member;
            let source_id = export_assets()
                .find_map(|rules| (rules.member_id)(member))
                .map(str::to_owned);
            let sidecar = export_assets()
                .find_map(|rules| (rules.sidecar_kind)(member))
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
            install(self, staged)?;
        }

        // The live index supplies this list and the current revisions.
        // For a rebuild, its own member list becomes readable at install;
        // the ordinary pass then parses sidecars from that list.
        if cut.is_some() {
            return Ok(());
        }
        let sidecars = self.reads.reader()?.pending_asset_sidecars()?;
        for (input, member) in sidecars {
            let reader = self.reads.reader()?;
            let Some(progress) = reader.latest_chain(&input)? else {
                continue;
            };
            let source = reader.asset_source(&input)?.ok_or_else(|| {
                crate::EngineError::internal("a stored sidecar has no current revision")
            })?;
            // Bytes still being written are read in a later pass.
            let chain = reader.stored_chain(&input, progress.chain, progress.stored)?;
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
            let Some(rules) = export_assets().find(|rules| (rules.sidecar_kind)(&member).is_some())
            else {
                continue;
            };
            let result = (rules.metadata)(
                &member,
                &mut chain.reader(Arc::clone(&self.blobs)),
                &mut |metadata| {
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
                    let failure = polylogue_model::Failure::stated(
                        format_args!("asset sidecar {input} ({member})"),
                        polylogue_model::Cause::Content,
                        polylogue_model::Code::Import,
                        error.to_string(),
                    );
                    let error = failure.to_string();
                    pass.errors.push(failure);
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
            install(self, staged)?;
        }
        Ok(())
    }
}
