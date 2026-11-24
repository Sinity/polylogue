//! Natural keys held in memory with their row keys, spilling to a file past
//! a budget. Candidate rows are read back and compared by the caller.
mod rows;
pub(crate) use rows::KeyRows;
