//! Serving requests on the Unix socket and loopback HTTP.

use super::{Request, Response};

/// Answers one request. Reads use the read pool; writes go to the writer.
pub fn handle(_request: Request) -> Response {
    // Architecture 6: request lifecycle, warm caches, typed errors.
    todo!()
}

/// Runs the daemon until it is told to stop: takes the archive lock,
/// starts the loop and serves clients.
pub fn serve() -> std::io::Result<()> {
    // Architecture 6: lock, loop, socket, loopback HTTP, restart classification.
    todo!()
}
