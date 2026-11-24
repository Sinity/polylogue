//! A first build's own record in the log: a line each time the pass enters a
//! phase, naming what the phase before it did and how long it took, and a
//! line a minute with the rates while inputs are stored and read. The phase
//! and the workers' CPU also go to the archive's build progress, which
//! `status` reports.
use crate::{Engine, Pass};
use std::time::{Duration, Instant};

/// How often a build logs its rates while inputs are stored and read.
const EVERY: Duration = Duration::from_secs(60);

impl Engine {
    /// Logs that a first build entered `phase`, with the time the previous
    /// phase took. Outside a first build nothing is logged.
    pub(crate) fn build_phase(&mut self, phase: &'static str, detail: std::fmt::Arguments<'_>) {
        if self.archive.index_build().is_none() {
            return;
        }
        let now = Instant::now();
        if let Some(since) = polylogue_model::Timestamp::from_unix_ms(self.config.clock.now_ms()) {
            self.archive.record_build_phase(phase, since);
        }
        let previous = self
            .build_phase
            .replace((phase, now))
            .map(|(name, started)| {
                format!(
                    "; {name} took {:.0} s",
                    now.duration_since(started).as_secs_f64()
                )
            })
            .unwrap_or_default();
        polylogue_store::log!(info, "build: {phase}: {detail}{previous}");
    }

    /// Records that a batch build entered `phase`, for `status`, without a
    /// line of its own: the phase's line, when it ends, carries its time,
    /// items, bytes and peak memory ([`polylogue_store::writer::build::PhaseClock`]).
    pub(crate) fn enter_build_phase(&mut self, phase: &'static str) {
        if self.archive.index_build().is_none() {
            return;
        }
        if let Some(since) = polylogue_model::Timestamp::from_unix_ms(self.config.clock.now_ms()) {
            self.archive.record_build_phase(phase, since);
        }
        self.build_phase = Some((phase, Instant::now()));
    }
}

/// What the minute's rates are measured from.
pub(crate) struct Progress {
    at: Instant,
    stored: u64,
    parsed: u64,
    writer: f64,
    cpu: Vec<f64>,
}

impl Progress {
    pub(crate) fn start(engine: &Engine, pass: &Pass, cpu: Vec<f64>) -> Self {
        Self {
            at: Instant::now(),
            stored: pass.bytes_stored,
            parsed: engine
                .archive
                .index_build()
                .map_or(0, |build| build.bytes_read),
            writer: polylogue_store::timing::writer_seconds(),
            cpu,
        }
    }

    /// Logs the rates since the last line once a minute has passed, during
    /// a first build.
    pub(crate) fn log(&mut self, engine: &Engine, pass: &Pass, cpu: impl FnOnce() -> Vec<f64>) {
        let elapsed = self.at.elapsed();
        if elapsed < EVERY {
            return;
        }
        let Some(build) = engine.archive.index_build() else {
            return;
        };
        let seconds = elapsed.as_secs_f64();
        let cpu = cpu();
        let busy: Vec<String> = cpu
            .iter()
            .zip(self.cpu.iter().chain(std::iter::repeat(&0.0)))
            .map(|(now, before)| format!("{:.0}", now - before))
            .collect();
        let writer = polylogue_store::timing::writer_seconds();
        engine.archive.record_build_workers(
            cpu.iter()
                .zip(self.cpu.iter().chain(std::iter::repeat(&0.0)))
                .map(|(now, before)| (now - before) * 60. / seconds)
                .collect(),
        );
        polylogue_store::log!(
            info,
            "build: {:.0} of {:.0} MB parsed, {:.1} MB/s; {:.1} MB/s stored; \
             {} of {} input chains read; writer busy {:.0} s and parse workers {} s of the last {:.0} s",
            build.bytes_read as f64 / 1e6,
            build.bytes as f64 / 1e6,
            build.bytes_read.saturating_sub(self.parsed) as f64 / 1e6 / seconds,
            pass.bytes_stored.saturating_sub(self.stored) as f64 / 1e6 / seconds,
            build.chains_read,
            build.chains,
            writer - self.writer,
            busy.join(" "),
            seconds,
        );
        *self = Self {
            at: Instant::now(),
            stored: pass.bytes_stored,
            parsed: build.bytes_read,
            writer,
            cpu,
        };
    }
}
