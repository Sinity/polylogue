use std::{env, process::{Command, ExitCode}};

fn main() -> ExitCode {
    let args: Vec<_> = env::args().skip(1).collect();
    if args != ["check"] {
        eprintln!("usage: cargo run -p xtask -- check");
        return ExitCode::from(2);
    }
    for args in [vec!["fmt", "--all", "--check"], vec!["test", "--locked", "--workspace"]] {
        match Command::new("cargo").args(args).status() {
            Ok(status) if status.success() => (),
            Ok(status) => return ExitCode::from(status.code().unwrap_or(1) as u8),
            Err(error) => { eprintln!("cargo: {error}"); return ExitCode::FAILURE; }
        }
    }
    ExitCode::SUCCESS
}
