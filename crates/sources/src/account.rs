//! Every framed record ends in exactly one disposition, and the counts must
//! add up to the records framed. The types live in `polylogue-model`, which
//! the writer shares; this module names them where parsers look.

pub use polylogue_model::{Accounting, CountMismatch, Disposition};
