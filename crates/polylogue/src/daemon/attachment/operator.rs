use std::io::IsTerminal;
use std::os::unix::fs::MetadataExt;

pub(crate) fn check_cli() -> Result<(), String> {
    if !std::io::stdout().is_terminal() || polylogue_sources::origin::agent_env().any(|key| std::env::var_os(key).is_some())
    {
        return Err(
            "exact attachment output requires terminal stdout outside an agent harness".into(),
        );
    }
    Ok(())
}

/// Constructed only from socket-observed process evidence, never request JSON.
pub(super) struct Operator;

impl Operator {
    pub(super) fn admit(stream: &std::os::unix::net::UnixStream) -> Result<Self, String> {
        let check = || -> std::io::Result<bool> {
            let peer = rustix::net::sockopt::socket_peercred(stream)?;
            if peer.uid != rustix::process::geteuid() {
                return Ok(false);
            }
            let process = std::path::PathBuf::from(format!("/proc/{}", peer.pid.as_raw_pid()));
            let executable = std::fs::metadata(process.join("exe"))?;
            let own = std::fs::metadata(std::env::current_exe()?)?;
            if (executable.dev(), executable.ino()) != (own.dev(), own.ino()) {
                return Ok(false);
            }
            let stdout = rustix::fs::open(
                process.join("fd/1"),
                rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::NOCTTY,
                rustix::fs::Mode::empty(),
            )?;
            if !std::fs::File::from(stdout).is_terminal() {
                return Ok(false);
            }
            let environment = std::fs::read(process.join("environ"))?;
            if environment.split(|byte| *byte == 0).any(|entry| {
                let name = entry.split(|byte| *byte == b'=').next().unwrap_or_default();
                polylogue_sources::origin::agent_env().any(|key| name == key.as_bytes())
            }) {
                return Ok(false);
            }
            let confirmed = std::fs::metadata(process.join("exe"))?;
            Ok((confirmed.dev(), confirmed.ino()) == (executable.dev(), executable.ino()))
        };
        match check() {
            Ok(true)=>Ok(Self),
            _=>Err("exact attachment bytes require a verified interactive CLI peer outside an agent harness".into()),
        }
    }
}
