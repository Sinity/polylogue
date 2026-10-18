//! MCP over stdio, as a proxy to the daemon: the same requests the CLI
//! sends, as tools.

pub fn serve_stdio() -> std::io::Result<()> {
    // Architecture 6, 7: MCP tools mapped onto daemon requests.
    todo!()
}
