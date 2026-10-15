use std::{env, fs, io::{self, Write}, path::PathBuf, process::ExitCode, sync::{Arc, atomic::{AtomicBool, Ordering}}, thread, time::Duration};
use serde_json::json;

fn schema() -> serde_json::Value { json!({"$schema":"https://json-schema.org/draft/2020-12/schema","title":"Polylogue foundation configuration","type":"object","properties":{"archive":{"type":"string","description":"CLI --archive; default XDG_DATA_HOME/polylogue"}},"additionalProperties":false}) }

fn run() -> Result<i32, (i32,String)> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let mut archive = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--archive" {
            if i+1 == args.len() { return Err((2,"--archive needs a directory".into())); }
            archive = Some(PathBuf::from(args.remove(i+1))); args.remove(i);
        } else { i += 1; }
    }
    let mut out = io::stdout().lock();
    if args == ["--help"] || args == ["-h"] {
        writeln!(out,"Polylogue foundation\nUsage: polylogue [--archive DIR] <daemon|status|config schema>\nNo session sources, queries, export or MCP are registered.").map_err(|e|(4,e.to_string()))?; return Ok(0);
    }
    if args == ["--version"] { writeln!(out,"polylogue {}",env!("CARGO_PKG_VERSION")).map_err(|e|(4,e.to_string()))?;return Ok(0); }
    if args == ["config","schema"] { serde_json::to_writer(&mut out,&schema()).map_err(|e|(4,e.to_string()))?;writeln!(out).map_err(|e|(4,e.to_string()))?;return Ok(0); }
    let archive = archive.or_else(||env::var_os("XDG_DATA_HOME").map(|p|PathBuf::from(p).join("polylogue"))).or_else(||env::var_os("HOME").map(|p|PathBuf::from(p).join(".local/share/polylogue"))).ok_or((2,"no archive selected; pass --archive DIR".into()))?;
    let archive = if archive.is_absolute() { archive } else { env::current_dir().map_err(|e|(4,e.to_string()))?.join(archive) };
    if args != ["daemon"] && args != ["status"] && !args.is_empty() { return Err((2,format!("not_yet_available: operation is not registered; archive {}",archive.display()))); }
    let runtime = env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).ok_or((2,format!("XDG_RUNTIME_DIR is required; archive {}",archive.display())))?;
    if !runtime.is_absolute() { return Err((2,format!("XDG_RUNTIME_DIR must be absolute; archive {}",archive.display()))); }
    if args == ["daemon"] {
        let mut running = polylogue::Running::start(&archive,&runtime).map_err(|e|(4,format!("archive {}: {e}",archive.display())))?;
        let stop = Arc::new(AtomicBool::new(false));let signalled = Arc::clone(&stop);
        ctrlc::set_handler(move ||signalled.store(true,Ordering::Release)).map_err(|e|(4,e.to_string()))?;
        eprintln!("foundation listening: {} (no sources registered)",running.socket().display());
        while !stop.load(Ordering::Acquire) { thread::sleep(Duration::from_millis(10)); }
        running.stop();return Ok(0);
    }
    let resolved = fs::canonicalize(&archive).map_err(|e|(if e.kind()==io::ErrorKind::NotFound{6}else{4},format!("archive {}: daemon is not running; start polylogue daemon ({e})",archive.display())))?;
    let socket = polylogue::socket_path(&resolved,&runtime);
    let value = polylogue::request(&socket).map_err(|e|(if matches!(e.kind(),io::ErrorKind::NotFound|io::ErrorKind::ConnectionRefused){6}else{4},format!("archive {}: {e}",archive.display())))?;
    serde_json::to_writer(&mut out,&value).map_err(|e|(4,e.to_string()))?;writeln!(out).map_err(|e|(4,e.to_string()))?;
    Ok(if value["outcome"]=="ok"{0}else{4})
}
fn main() -> ExitCode { match run() { Ok(code)=>ExitCode::from(code as u8),Err((code,error))=>{eprintln!("{error}");ExitCode::from(code as u8)} } }
