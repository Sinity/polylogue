use serde_json::Value;
use std::{
    env,
    path::Path,
    process::{Command, ExitCode},
};

fn boundaries(root: &Path) -> Result<(), String> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(root)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let metadata: Value = serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())?;
    for package in metadata["packages"]
        .as_array()
        .ok_or("metadata packages missing")?
    {
        let name = package["name"].as_str().ok_or("package name missing")?;
        let allowed: &[&str] = match name {
            "polylogue-model" | "hashed-blobs" | "xtask" => &[],
            "polylogue-sources" | "polylogue-query" | "polylogue-render" => &["polylogue-model"],
            "polylogue-store" => &["polylogue-model", "hashed-blobs"],
            "polylogue-engine" => &[
                "polylogue-model",
                "polylogue-sources",
                "polylogue-query",
                "polylogue-store",
            ],
            "polylogue" => &[
                "polylogue-model",
                "polylogue-store",
                "polylogue-sources",
                "polylogue-query",
                "polylogue-engine",
                "polylogue-render",
            ],
            other => return Err(format!("unassigned workspace package {other}")),
        };
        for dependency in package["dependencies"]
            .as_array()
            .ok_or("dependencies missing")?
        {
            let dependency = dependency["name"]
                .as_str()
                .ok_or("dependency name missing")?;
            let internal = metadata["packages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["name"] == dependency);
            if (internal && !allowed.contains(&dependency))
                || (dependency == "rusqlite" && name != "polylogue-store")
            {
                return Err(format!("forbidden dependency: {name} -> {dependency}"));
            }
        }
    }
    Ok(())
}
fn main() -> ExitCode {
    let args: Vec<_> = env::args().skip(1).collect();
    if args != ["check"] && args != ["boundaries"] {
        eprintln!("usage: cargo run -p xtask -- <check|boundaries>");
        return ExitCode::from(2);
    }
    if let Err(error) = boundaries(Path::new(".")) {
        eprintln!("{error}");
        return ExitCode::FAILURE;
    }
    if args == ["boundaries"] {
        return ExitCode::SUCCESS;
    }
    // The shared crate is an independently reviewed dependency; do not reformat its snapshot.
    for args in [
        vec![
            "fmt",
            "-p",
            "polylogue-model",
            "-p",
            "polylogue-store",
            "-p",
            "polylogue",
            "-p",
            "xtask",
            "--",
            "--check",
        ],
        vec!["test", "--locked", "--workspace"],
    ] {
        match Command::new("cargo").args(args).status() {
            Ok(status) if status.success() => (),
            Ok(status) => return ExitCode::from(status.code().unwrap_or(1) as u8),
            Err(error) => {
                eprintln!("cargo: {error}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_cargo_edge_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("Cargo.toml"),
            "[workspace]\nmembers=[\"query\",\"store\"]\nresolver=\"3\"\n",
        )
        .unwrap();
        for (dir, name) in [("query", "polylogue-query"), ("store", "polylogue-store")] {
            std::fs::create_dir_all(root.path().join(dir).join("src")).unwrap();
            std::fs::write(root.path().join(dir).join("src/lib.rs"), "").unwrap();
            std::fs::write(
                root.path().join(dir).join("Cargo.toml"),
                format!("[package]\nname=\"{name}\"\nversion=\"0.1.0\"\nedition=\"2024\"\n"),
            )
            .unwrap();
        }
        assert!(boundaries(root.path()).is_ok());
        use std::io::Write;
        let mut manifest = std::fs::OpenOptions::new()
            .append(true)
            .open(root.path().join("query/Cargo.toml"))
            .unwrap();
        manifest
            .write_all(b"[dependencies]\npolylogue-store={path=\"../store\"}\n")
            .unwrap();
        assert_eq!(
            boundaries(root.path()).unwrap_err(),
            "forbidden dependency: polylogue-query -> polylogue-store"
        );
    }
}
