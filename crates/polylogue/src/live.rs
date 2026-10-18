//! Live state, computed at read time from stored columns: which sessions
//! are running, waiting or idle. Evidence only, never coordination.

pub enum LiveState {
    Running,
    WaitingForUser,
    Idle,
    Unknown,
}
