//! A chain's state as the index holds it, read back as the model's enum.
use super::*;
use polylogue_model::chain::ChainState;

impl Reader<'_> {
    /// The state of chain `chain` of `input`. A tool database whose rows
    /// were imported is read through its row stream, so its own chain has
    /// no parse.
    pub fn chain_state(&self, input: &str, chain: i64) -> Result<ChainState> {
        #[cfg(feature = "faults")]
        crate::fault::check(
            "parser.checkpoint",
            &format!("{}:{input}", self.pool.archive.root.display()),
        )?;
        let conn = self.conn()?;
        let database = crate::source::database_original("?1");
        if conn.exists(&format!("SELECT {database}"), [input])? {
            return Ok(ChainState::Unparsed);
        }
        crate::parse_state::state_of_input(conn, input, chain)
    }
}

#[cfg(test)]
mod tests;
