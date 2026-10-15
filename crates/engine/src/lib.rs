//! Provider interpretation runs against pinned store views outside write
//! transactions. These primitives do not certify canonical publication or
//! register live collection, querying or search.
pub mod converge;
pub mod owner;
pub mod source_turn;
pub mod state;

pub mod version_proof;

pub mod canonical_message;

mod source_rows;
