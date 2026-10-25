//! A ZIP import stores the ZIP as one input and each file member as its own
//! input that names the ZIP, all in one writer group.
use super::*;

impl Engine {
    pub(super) fn import_zip(
        &mut self,
        path: &Path,
        mut file: std::fs::File,
        facts: FileFacts,
        selected: &polylogue_store::ImportMember,
        report: &mut ImportReport,
    ) -> Result<(), EngineError> {
        let import_id = uuid::Uuid::now_v7().to_string();
        let mut members = {
            let mut archive = zip::ZipArchive::new(&mut file)
                .map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))?;
            let mut captured = Vec::new();
            for index in 0..archive.len() {
                let mut member = archive.by_index(index).map_err(|e| {
                    std::io::Error::other(format!("{} member {index}: {e}", path.display()))
                })?;
                if !member.is_file() {
                    continue;
                }
                let name = member.name().to_owned();
                let declared = member.size();
                let mut bytes = Captured::stream(&self.archive, &mut member)?;
                if bytes.prepared.len() != declared {
                    return Err(std::io::Error::other(
                        "ZIP member length differs from its declaration",
                    )
                    .into());
                }
                bytes.member = Some(polylogue_store::ImportMember {
                    version: 1,
                    import_id: import_id.clone(),
                    member: name.clone(),
                    format: None,
                    zip: None,
                });
                captured.push((
                    PathBuf::from(format!("{}!/{name}", path.display())),
                    index as u64,
                    bytes,
                ));
            }
            captured
        };
        std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(0))?;
        let original = self.archive.prepare_bytes(&mut file)?;
        unchanged(&file, facts, original.len())?;
        let original_id = original.hash().to_string();
        for (_, index, member) in &mut members {
            member.member.as_mut().expect("captured membership").zip =
                Some(polylogue_store::InZip {
                    input: original_id.clone(),
                    member_index: *index,
                });
        }
        let kept = polylogue_sources::origin::by_name(KEPT).expect("declared origin");
        let zip = Captured {
            prepared: original,
            head: Vec::new(),
            polylogue_export: false,
            member: Some(polylogue_store::ImportMember {
                version: 1,
                import_id,
                member: selected.member.clone(),
                format: None,
                zip: None,
            }),
        };
        let mut items = Vec::with_capacity(members.len() + 1);
        items.push((path.to_path_buf(), kept, Some(facts), zip));
        for (at, _, member) in members {
            let origin = self.recognize_captured(&at, &member)?.unwrap_or(kept);
            items.push((at, origin, None, member));
        }
        let mut metadata = Vec::with_capacity(items.len());
        let mut group = Vec::with_capacity(items.len());
        for (at, origin, facts, captured) in items {
            if origin.support != Support::Parsed {
                report.unrecognized.push(at.clone());
            }
            let mut batch = self.import_batch(
                &at,
                origin,
                captured.member,
                facts,
                captured.prepared.hash().to_string(),
                captured.prepared.len(),
            );
            batch.new_blob = Some(polylogue_store::NewBlob {
                hash: captured.prepared.hash(),
                size: captured.prepared.len(),
                stored: polylogue_store::blobs::Stored::File { stored_size: 0 },
            });
            metadata.push((at, origin, facts, batch.input.clone()));
            group.push(polylogue_store::staging::PreparedBatch {
                batch,
                files: vec![captured.prepared],
            });
        }
        let outcomes = self
            .writer
            .sender()?
            .store_group(group, Arc::clone(&self.blobs))?;
        for ((at, origin, facts, input), outcome) in metadata.into_iter().zip(outcomes) {
            let stored = match outcome {
                Ok(stored) => stored,
                Err(error) => {
                    report.pass.errors.push(error.to_string());
                    continue;
                }
            };
            if stored.already_stored {
                report.already_stored.push(at.clone());
            } else {
                report.imported.push(at.clone());
                report.pass.bytes_stored += stored.length;
            }
            if let Some(plan) = self.stored_import_plan(&at, origin, input, &stored, facts)? {
                self.execute(vec![plan], &mut report.pass)?;
            }
        }
        Ok(())
    }
}
