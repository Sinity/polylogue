//! The earlier rows a parse step's chain-local names are settled against
//! (`polylogue_sources::chain_names`): the rows settled earlier in the same
//! run, in front of the chain's committed rows when the run continues the
//! committed reading.
use polylogue_model::{MessageId, MessageRow, SessionId};
use polylogue_sources::chain_names::{EarlierRows, Formed};
use polylogue_store::chain_names::ChainRows;

pub(crate) struct Committed(Option<ChainRows>);

pub(crate) type Names = Formed<Committed>;

/// The names for a run over `chain` of `input` that starts at `start`.
/// `None` when the run starts past the first byte but the chain's committed
/// reading does not stand there: it must be read from its start.
pub(crate) fn open(
    archive: &polylogue_store::Archive,
    input: &str,
    chain: Option<i64>,
    reading_version: &str,
    start: u64,
) -> polylogue_store::Result<Option<Names>> {
    if start == 0 {
        return Ok(Some(Formed::new(Committed(None))));
    }
    let Some(chain) = chain else {
        return Ok(None);
    };
    Ok(
        ChainRows::open(archive, input, chain, reading_version, start)?
            .map(|rows| Formed::new(Committed(Some(rows)))),
    )
}

impl EarlierRows for Committed {
    fn response(&mut self, session: &SessionId, key: &str) -> Result<Option<MessageRow>, String> {
        match &mut self.0 {
            Some(rows) => rows.response(session, key).map_err(|e| e.to_string()),
            None => Ok(None),
        }
    }
    fn alias(&mut self, name: &MessageId) -> Result<Option<Option<MessageId>>, String> {
        match &mut self.0 {
            Some(rows) => rows.alias(name).map_err(|e| e.to_string()),
            None => Ok(None),
        }
    }
    fn content_count(&mut self, name: &str) -> Result<u64, String> {
        match &mut self.0 {
            Some(rows) => rows.content_count(name).map_err(|e| e.to_string()),
            None => Ok(0),
        }
    }
    fn content_before(&mut self, name: &str, position: u64) -> Result<u64, String> {
        match &mut self.0 {
            Some(rows) => rows
                .content_before(name, position)
                .map_err(|e| e.to_string()),
            None => Ok(0),
        }
    }
    fn latest_opened(&mut self, session: &SessionId) -> Result<Option<MessageId>, String> {
        match &mut self.0 {
            Some(rows) => rows.latest_opened(session).map_err(|e| e.to_string()),
            None => Ok(None),
        }
    }
}
