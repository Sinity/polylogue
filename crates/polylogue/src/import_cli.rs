//! Client acquisition metadata is fixed once per action and reused on retry.
use polylogue_model::types::{ErrorKind, Failure};
use serde_json::{Value, json};
use std::{
    fs,
    io::Read,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
fn invalid(message: impl Into<String>) -> Failure {
    Failure::new(ErrorKind::InvalidQuery, message)
}
fn uuid_v7(millis: u64, random: [u8; 16]) -> Result<String, Failure> {
    if millis >= (1u64 << 48) {
        return Err(invalid("clock is outside UUIDv7 range"));
    }
    let mut bytes = random;
    bytes[..6].copy_from_slice(&millis.to_be_bytes()[2..]);
    bytes[6] = (bytes[6] & 15) | 0x70;
    bytes[8] = (bytes[8] & 63) | 0x80;
    let hex = bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    Ok(format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    ))
}
pub(crate) fn request(args: &[String]) -> Result<Value, Failure> {
    let mut path = None;
    let mut format = None;
    let mut id = None;
    let mut seen_at = None;
    let mut machine = None;
    let mut host = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--format" | "--request-id" | "--seen-at" | "--machine" | "--host" => {
                let flag = &args[i];
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| invalid(format!("{flag} requires a value")))?
                    .clone();
                let slot = match flag.as_str() {
                    "--format" => &mut format,
                    "--request-id" => &mut id,
                    "--seen-at" => &mut seen_at,
                    "--machine" => &mut machine,
                    _ => &mut host,
                };
                if slot.replace(value).is_some() {
                    return Err(invalid(format!("duplicate {flag}")));
                }
            }
            "--wait" | "--dry-run" => {
                return Err(Failure::new(
                    ErrorKind::NotYetAvailable,
                    "wait/dry-run require canonical finalization; no source bytes were acquired",
                ));
            }
            flag if flag.starts_with('-') => {
                return Err(invalid(format!("unknown import argument {flag}")));
            }
            value => {
                if path.replace(PathBuf::from(value)).is_some() {
                    return Err(invalid("this stage accepts one regular file per request"));
                }
            }
        }
        i += 1;
    }
    let path = path.ok_or_else(|| invalid("import requires a regular file path"))?;
    if format.as_deref() != Some("codex-rollout") {
        return Err(Failure::new(
            ErrorKind::NotYetAvailable,
            "pass --format codex-rollout; other source imports are not registered",
        ));
    }
    if id.is_some() != seen_at.is_some() {
        return Err(invalid(
            "a retry requires both --request-id and --seen-at from the original response",
        ));
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| invalid(e.to_string()))?;
    let (id, seen_at) = match (id, seen_at) {
        (Some(id), Some(time)) => (
            id,
            time.parse::<i64>()
                .map_err(|_| invalid("--seen-at requires integer epoch microseconds"))?,
        ),
        _ => {
            let mut random = [0; 16];
            fs::File::open("/dev/urandom")
                .and_then(|mut f| f.read_exact(&mut random))
                .map_err(|e| Failure::new(ErrorKind::Io, e.to_string()))?;
            (
                uuid_v7(
                    u64::try_from(now.as_millis()).map_err(|_| invalid("clock overflow"))?,
                    random,
                )?,
                i64::try_from(now.as_micros()).map_err(|_| invalid("clock overflow"))?,
            )
        }
    };
    let path = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .map_err(|e| Failure::new(ErrorKind::Io, e.to_string()))?
            .join(path)
    };
    let path = path
        .to_str()
        .ok_or_else(|| invalid("source path must be UTF-8"))?;
    let (machine, host) = match (machine, host) {
        (Some(machine), Some(host)) => (machine, host),
        (None, None) => (
            fs::read_to_string("/etc/machine-id")
                .map_err(|e| {
                    Failure::new(ErrorKind::Io, format!("collector machine identity: {e}"))
                })?
                .trim()
                .to_owned(),
            fs::read_to_string("/proc/sys/kernel/hostname")
                .map_err(|e| Failure::new(ErrorKind::Io, format!("collector host identity: {e}")))?
                .trim()
                .to_owned(),
        ),
        _ => {
            return Err(invalid(
                "observation identity requires both --machine and --host",
            ));
        }
    };
    Ok(
        json!({"command":"import","context":[],"id":id,"format":"codex-rollout","path":path,"machine":machine,"host":host,"seen_at":seen_at}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uuid_has_actual_timestamp_version_and_variant() {
        assert_eq!(
            uuid_v7(0x123456789abc, [255; 16]).unwrap(),
            "12345678-9abc-7fff-bfff-ffffffffffff"
        );
        assert!(uuid_v7(1 << 48, [0; 16]).is_err());
    }
    #[test]
    fn explicit_observation_identity_is_paired_and_preserved() {
        let base = [
            "import",
            "/synthetic/provider.jsonl",
            "--format",
            "codex-rollout",
            "--request-id",
            "synthetic-request",
            "--seen-at",
            "17",
        ];
        let mut args = base.into_iter().map(String::from).collect::<Vec<_>>();
        args.extend(
            [
                "--machine",
                "00112233445566778899aabbccddeeff",
                "--host",
                "synthetic-collector",
            ]
            .map(String::from),
        );
        let request = request(&args).unwrap();
        assert_eq!(request["machine"], "00112233445566778899aabbccddeeff");
        assert_eq!(request["host"], "synthetic-collector");
        args.truncate(args.len() - 2);
        assert!(super::request(&args).is_err());
        let mut args = base.into_iter().map(String::from).collect::<Vec<_>>();
        args.extend(["--host", "synthetic-collector"].map(String::from));
        assert!(super::request(&args).is_err());
    }
    #[test]
    fn unsupported_import_options_fail_before_acquisition() {
        for args in [
            vec!["import", "x", "--wait"],
            vec!["import", "x", "--format", "claude-code-jsonl"],
            vec![
                "import",
                "x",
                "--format",
                "codex-rollout",
                "--request-id",
                "x",
            ],
        ] {
            assert!(request(&args.into_iter().map(String::from).collect::<Vec<_>>()).is_err());
        }
    }
}
