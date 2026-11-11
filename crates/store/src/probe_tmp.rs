//! TEMPORARY timing probe for the read-deadline investigation. Not for commit.
use std::io::Write;
use std::sync::OnceLock;
use std::time::Instant;
static START: OnceLock<Instant> = OnceLock::new();
pub fn log(label: &str, spent: std::time::Duration) {
    let start = *START.get_or_init(Instant::now);
    let line = format!(
        "{} {:?}/{} t={:.1} {} {:.2}ms\n",
        std::process::id(),
        std::thread::current().id(),
        std::thread::current().name().unwrap_or("-"),
        start.elapsed().as_secs_f64() * 1000.0,
        label,
        spent.as_secs_f64() * 1000.0
    );
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/realm/tmp/work/read-deadline/trace.log")
    {
        let _ = f.write_all(line.as_bytes());
    }
}
pub fn cpu_now() -> std::time::Duration {
    let text = std::fs::read_to_string("/proc/thread-self/schedstat").unwrap_or_default();
    std::time::Duration::from_nanos(
        text.split_whitespace()
            .next()
            .and_then(|n| n.parse().ok())
            .unwrap_or(0),
    )
}
pub struct Span(&'static str, Instant, std::time::Duration);
pub fn span(label: &'static str) -> Span {
    Span(label, Instant::now(), cpu_now())
}
impl Drop for Span {
    fn drop(&mut self) {
        let cpu = cpu_now().saturating_sub(self.2);
        log(
            &format!("{}(cpu={:.1}ms)", self.0, cpu.as_secs_f64() * 1000.0),
            self.1.elapsed(),
        );
    }
}
