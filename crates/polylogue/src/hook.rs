//! The hook command a harness calls on its events. It appends one line per
//! event to a per-process carrier file with `O_APPEND` and no fsync; the
//! daemon reads carriers through the ordinary append route. Advisory hooks
//! answer within 20 ms or fail open.

pub fn run(_event: &str) -> std::io::Result<()> {
    // Architecture 6: one O_APPEND line per event, no fsync (old k3ahm).
    todo!()
}
