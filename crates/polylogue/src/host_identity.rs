//! Explicit process machine attribution, shared by CLI and MCP adapters.
use polylogue_model::{ErrorKind, Failure};
use std::{fs::File, io::Read, path::Path};

pub(crate) fn machine() -> Result<String, Failure> {
    let path = std::env::var_os("POLYLOGUE_MACHINE_ID_FILE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "/etc/machine-id".into());
    read_machine(&path)
}

fn read_machine(path: &Path) -> Result<String, Failure> {
    if !path.is_absolute() {
        return Err(Failure::new(
            ErrorKind::InvalidQuery,
            "POLYLOGUE_MACHINE_ID_FILE must name an absolute machine-id file",
        ));
    }
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|file| file.take(129).read_to_end(&mut bytes))
        .map_err(|e| Failure::new(ErrorKind::Io, format!("machine identity file: {e}; configure POLYLOGUE_MACHINE_ID_FILE when /etc/machine-id is unavailable")))?;
    let value = std::str::from_utf8(&bytes).ok().map(str::trim);
    let value = value.filter(|value| {
        bytes.len() <= 128
            && value.len() == 32
            && value
                .bytes()
                .all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f'))
            && value.bytes().any(|c| c != b'0')
    });
    value.map(str::to_owned).ok_or_else(|| {
        Failure::new(
            ErrorKind::InvalidQuery,
            "machine identity requires a nonzero 32-character lowercase hexadecimal machine-id; no fallback identity was invented",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn configured_machine_identity_is_bounded_explicit_and_never_invented() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("machine-id");
        assert_eq!(read_machine(&path).unwrap_err().kind, ErrorKind::Io);
        assert_eq!(
            read_machine(Path::new("relative")).unwrap_err().kind,
            ErrorKind::InvalidQuery
        );
        let expected = "00112233445566778899aabbccddeeff";
        std::fs::write(&path, format!("{expected}\n")).unwrap();
        assert_eq!(read_machine(&path).unwrap(), expected);
        for invalid in [
            "".to_owned(),
            "uninitialized".into(),
            "0".repeat(32),
            "A".repeat(32),
            "a".repeat(129),
        ] {
            std::fs::write(&path, invalid).unwrap();
            assert_eq!(
                read_machine(&path).unwrap_err().kind,
                ErrorKind::InvalidQuery
            );
        }
    }
}
