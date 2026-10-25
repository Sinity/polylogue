//! Replay immutable inputs by acquisition identity, apart from watched paths.
use super::*;

impl Engine {
    pub(crate) fn replay_imports(
        &mut self,
        selected: Option<&[String]>,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        let imports = self.reads.reader()?.stored_imports()?;
        for input in imports {
            if selected.is_some_and(|ids| !ids.contains(&input.id)) {
                continue;
            }
            let Some(origin) = polylogue_sources::origin::by_name(&input.format) else {
                continue;
            };
            if origin.support != Support::Parsed {
                continue;
            }
            let state = input.state;
            if state.parsed && (state.waiting || state.parsed_upto >= state.length) {
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
                    Err(polylogue_store::StoreError::InputPending { .. }) => continue,
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
