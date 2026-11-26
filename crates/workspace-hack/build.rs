// A build script is required for cargo to consider build dependencies.
// Without a rerun-if line Cargo compares the package's newest file time with
// the one recorded at the last run, and the hosted check dates restored
// sources, so every job would rerun it and rebuild the whole workspace.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
