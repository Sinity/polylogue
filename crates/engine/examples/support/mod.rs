/// Capture the measurement binary, once before constructing its Engine.
pub fn stopped_attempt() -> polylogue_model::batch::StoppedAttempt {
    #[cfg(target_os = "linux")]
    let file = std::fs::File::open("/proc/self/exe").expect("open executable");
    #[cfg(not(target_os = "linux"))]
    let file = std::fs::File::open(std::env::current_exe().expect("executable path"))
        .expect("open executable");
    let mut digest = blake3::Hasher::new();
    digest.update_reader(file).expect("read executable");
    polylogue_model::batch::StoppedAttempt {
        executable: polylogue_model::BlobHash(*digest.finalize().as_bytes()),
        parser_settings: None,
    }
}
