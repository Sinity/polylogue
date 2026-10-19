//! Laws: properties every archive must keep, whatever its inputs. Each
//! test names the law it checks by the number the "Laws for" Beads give
//! it.
//!
//! - `generated`: property tests over generated Claude Code archives;
//! - `example_archive`: laws over the Morrow example archive, built through
//!   the real loop and import.

mod example_archive;
mod generated;
