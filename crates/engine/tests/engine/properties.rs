//! Properties every archive keeps, whatever its inputs.
//!
//! - `generated`: property tests over generated Claude Code archives;
//! - `example_archive`: properties of the Morrow example archive, built
//!   through the real loop and import;
//! - `repeated_versions`: a file rewritten into versions that repeat each
//!   other, updated and rebuilt.

mod example_archive;
mod generated;
mod repeated_versions;
