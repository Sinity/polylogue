fn main() {
    println!("cargo:rerun-if-changed=vendor/cksumvfs.c");
    println!("cargo:rerun-if-changed=vendor/bridge.c");
    let include = std::env::var("DEP_SQLITE3_INCLUDE").expect("bundled SQLite headers");
    cc::Build::new()
        .file("vendor/cksumvfs.c")
        .file("vendor/bridge.c")
        .include(include)
        .define("SQLITE_CKSUMVFS_STATIC", None)
        .warnings(false)
        .compile("polylogue_cksumvfs");
}
