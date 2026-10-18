//! The Sinex backend (architecture 7.1). Two modes: standalone (the
//! default; no Sinex code runs) and primary (Sinex holds the source
//! material and evidence). Moving to primary is a one-time cutover the
//! operator approves.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Standalone,
    Primary,
}

pub fn cutover() -> std::io::Result<()> {
    // Architecture 7.1: rebuild proof first, then the one-time move.
    todo!()
}
