use super::super::*;
use std::io::{self, BufReader, Seek, Write};

#[test]
#[ignore = "paired memory measurement through AgentCTL measure"]
fn observer_memory() {
    let Some(workload) = std::env::var_os("POLYLOGUE_SCHEMA_MEMORY") else {
        for workload in ["optional", "dynamic", "line", "array", "capture"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "schema::tests::memory::observer_memory",
                    "--ignored",
                    "--nocapture",
                ])
                .env("POLYLOGUE_SCHEMA_MEMORY", workload)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{workload}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            print!("{}", String::from_utf8_lossy(&output.stdout));
        }
        return;
    };
    let mut file = tempfile::tempfile().unwrap();
    let framing = match workload.to_str().unwrap() {
        "optional" => {
            for mask in 0..65536u32 {
                write!(file, "{{").unwrap();
                let mut comma = "";
                for bit in 0..16 {
                    if mask & (1 << bit) != 0 {
                        write!(file, "{comma}\"field{bit}\":0").unwrap();
                        comma = ",";
                    }
                }
                writeln!(file, "}}").unwrap();
            }
            Framing::Lines
        }
        "dynamic" => {
            for i in 0..40000 {
                writeln!(file, "{{\"mapping\":{{\"entry{i}\":{{\"text\":\"x\"}}}}}}").unwrap();
            }
            Framing::Lines
        }
        kind => {
            let mut body = tempfile::tempfile().unwrap();
            let mut hash = blake3::Hasher::new();
            let block = [b'x'; 8192];
            for bytes in std::iter::once(b"{\"text\":\"".as_slice())
                .chain(std::iter::repeat_n(block.as_slice(), 8192))
                .chain(std::iter::once(b"\"}".as_slice()))
            {
                body.write_all(bytes).unwrap();
                hash.update(bytes);
            }
            body.rewind().unwrap();
            if kind == "capture" {
                let header = polylogue_model::capture_log::Header {
                    length: body.metadata().unwrap().len(),
                    digest: hash.finalize().to_hex().to_string(),
                };
                file.write_all(polylogue_model::capture_log::MAGIC).unwrap();
                file.write_all(&header.encode()).unwrap();
            } else if kind == "array" {
                file.write_all(b"[").unwrap();
            }
            io::copy(&mut body, &mut file).unwrap();
            if kind == "array" {
                file.write_all(b"]").unwrap();
            }
            match kind {
                "capture" => Framing::CaptureLog,
                "array" => Framing::Array,
                _ => Framing::Lines,
            }
        }
    };
    file.rewind().unwrap();
    let mut observer = Observer::default();
    observer.read(BufReader::new(file), framing).unwrap();
    let schema = observer.finish("synthetic".into(), vec![]);
    match workload.to_str().unwrap() {
        "optional" => {
            assert_eq!((schema.records, schema.fields.len()), (65536, 17));
            assert_eq!(schema.fields["/field0"].present, 32768);
        }
        "dynamic" => {
            assert_eq!((schema.records, schema.fields.len()), (40000, 4));
            assert_eq!(schema.fields["/mapping/*"].present, 40000);
        }
        _ => {
            assert_eq!((schema.records, schema.fields.len()), (1, 2));
            assert_eq!(schema.fields["/text"].types["string"], 1);
        }
    }
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let peak = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .unwrap();
    println!(
        "{}: {peak}, records={}, fields={}",
        workload.to_string_lossy(),
        schema.records,
        schema.fields.len()
    );
}
