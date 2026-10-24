//! Replay immutable inputs by acquisition identity, apart from watched paths.
use super::*;

impl Engine {
    pub(crate) fn replay_imports(
        &mut self,
        selected: Option<&[String]>,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        let imports = self.reads.reader()?.retained_imports()?;
        for mut input in imports {
            if selected.is_some_and(|ids| !ids.contains(&input.id)) {
                continue;
            }
            if input.format == KEPT && input.legacy_context {
                let reader = self.reads.reader()?;
                let paths = reader.retained_import_paths(&input.id)?;
                let chain =
                    match reader.stored_chain(&input.id, input.state.chain, input.state.length) {
                        Ok(chain) => chain,
                        Err(polylogue_store::StoreError::InputUnfinished { .. }) => continue,
                        Err(error) => return Err(error.into()),
                    };
                drop(reader);
                let mut bytes = CaptureReader { reader: chain.reader(Arc::clone(&self.blobs)), size: 0, head: Vec::new() };
                let normalized = polylogue_sources::polylogue_export::recognize_reader(&mut bytes)?;
                let mut tail = [0; 64];
                while bytes.head.len() < 64 {
                    let remaining = 64 - bytes.head.len();
                    if std::io::Read::read(&mut bytes, &mut tail[..remaining])? == 0 { break; }
                }
                pass.bytes_read += bytes.size;
                let mut choices = std::collections::BTreeMap::new();
                for path in paths {
                    let path = PathBuf::from(path);
                    let origin = if normalized {
                        polylogue_sources::origin::by_name("polylogue-export")
                    } else {
                        recognize_import(&path, &bytes.head)
                    };
                    if let Some(origin) = origin {
                        choices.entry(origin.name.to_owned()).or_insert(path);
                    }
                }
                let candidates: Vec<_> = choices.into_iter().map(|(format,path)| LegacyImportCandidate { format,path }).collect();
                if candidates.len() == 1 {
                    input.format = candidates[0].format.clone();
                    input.path = Some(candidates[0].path.to_string_lossy().into_owned());
                } else if candidates.len() > 1 {
                    pass.errors.push(format!(
                        "legacy import {} has competing current format interpretations",
                        input.id
                    ));
                }
                pass.legacy_imports.push(LegacyImport {
                    input: input.id.clone(),
                    candidates,
                });
            }
            let Some(origin) = polylogue_sources::origin::by_name(&input.format) else {
                continue;
            };
            if origin.support != Support::Parsed {
                continue;
            }
            let state = input.state;
            if state.parser_version == Some(origin.parser_version)
                && (state.waiting || state.parsed_upto >= state.length)
            {
                continue;
            }
            let path = PathBuf::from(input.path.ok_or_else(|| {
                std::io::Error::other(format!(
                    "retained import {} has no captured parser path",
                    input.id
                ))
            })?);
            if (origin.open)(&path).is_none() {
                return Err(std::io::Error::other(format!(
                    "retained import {} has no parser for its captured path",
                    input.id
                ))
                .into());
            }
            let retained_chain =
                match self
                    .reads
                    .reader()?
                    .stored_chain(&input.id, state.chain, state.length)
                {
                    Ok(chain) => chain,
                    Err(polylogue_store::StoreError::InputUnfinished { .. }) => continue,
                    Err(error) => return Err(error.into()),
                };
            let source = pool::Bytes::Stored(Arc::new(retained_chain));
            self.execute(
                vec![Plan {
                    path: path.clone(),
                    origin,
                    input: InputKey {
                        id: input.id,
                        format: input.format,
                        route_rank: state.route_rank,
                    },
                    change: Change::Reparse { chain: state.chain },
                    observation: None,
                    facts: None,
                    size: state.length,
                    piece_pos: state.length,
                    parse_pos: 0,
                    piece_started: false,
                    carry: None,
                    tail: None,
                    span: span(origin, &path, self.config.chunk),
                    chunk: self.config.chunk,
                    priority: Priority::Bulk,
                    tracked: false,
                    failed: false,
                    waiting: false,
                    source,
                }],
                pass,
            )?;
        }
        Ok(())
    }
}
