//! An index build's own record in the log and in `status`: a line each time
//! the build starts a step, naming how long the step before took, and the
//! steps, counts and workers' CPU in the archive's build progress, which
//! `status` reports. While files are stored a line a minute gives the
//! storing rate; it goes to the journal only, since `status` shows the
//! count and rate itself.
use crate::{Engine, Pass};
use polylogue_store::archive::build_progress::{CountUnit, StepCount};
use std::time::{Duration, Instant};

/// The steps the engine runs before a rebuild, as `status`, the log and the
/// docs name them; the rebuild's stages follow
/// ([`polylogue_store::index::rebuild::RebuildStage`]).
pub(crate) const FINDING_FILES: &str = "finding files";
pub(crate) const STORING_FILES: &str = "storing files";

/// How often a build logs its storing rate.
const EVERY: Duration = Duration::from_secs(60);

impl Engine {
    /// Records the stages a rebuild runs once its files are stored.
    pub(crate) fn plan_build_steps(&self) {
        use polylogue_store::index::rebuild::RebuildStage;
        self.archive
            .record_build_plan(&RebuildStage::ALL.map(RebuildStage::name));
    }

    /// Logs that a build started `step`, with the time the previous step
    /// took. Outside a build nothing is logged.
    pub(crate) fn log_build_step(&mut self, step: &'static str, detail: std::fmt::Arguments<'_>) {
        if self.archive.index_build().is_none() {
            return;
        }
        let now = Instant::now();
        if let Some(since) = polylogue_model::Timestamp::from_unix_ms(self.config.clock.now_ms()) {
            self.archive.record_build_step(step, since);
        }
        let previous = self
            .build_step
            .replace((step, now))
            .filter(|(name, _)| *name != step)
            .map(|(name, started)| {
                format!(
                    "; {name} took {:.0} s",
                    now.duration_since(started).as_secs_f64()
                )
            })
            .unwrap_or_default();
        polylogue_store::log!(info, "index: {step}: {detail}{previous}");
    }
}

/// What the minute's storing rate is measured from, and the files the
/// storing step has finished.
pub(crate) struct Progress {
    at: Instant,
    files: u64,
    stored: u64,
    writer: f64,
    cpu: Vec<f64>,
}

impl Progress {
    /// Starts measuring from `cpu`, each parse worker's CPU seconds so far.
    pub(crate) fn start(pass: &Pass, cpu: Vec<f64>) -> Self {
        Self {
            at: Instant::now(),
            files: 0,
            stored: pass.bytes_stored,
            writer: polylogue_store::timing::writer_seconds(),
            cpu,
        }
    }

    /// During a build's storing step, records that one more file of the
    /// pass's changed inputs is stored.
    pub(crate) fn file_done(&mut self, engine: &Engine, pass: &Pass) {
        if engine.archive.index_build().is_none() {
            return;
        }
        self.files += 1;
        polylogue_store::activity::progressed();
        engine.archive.record_build_count(
            STORING_FILES,
            StepCount {
                done: self.files,
                total: Some(pass.inputs_changed.max(self.files)),
                unit: CountUnit::Files,
            },
        );
    }

    /// During a build's storing step, records the workers' CPU and logs the
    /// storing rate once a minute has passed.
    pub(crate) fn record(&mut self, engine: &Engine, pass: &Pass, cpu: impl FnOnce() -> Vec<f64>) {
        let elapsed = self.at.elapsed();
        if elapsed < EVERY || engine.archive.index_build().is_none() {
            return;
        }
        let seconds = elapsed.as_secs_f64();
        let cpu = cpu();
        engine.archive.record_build_workers(
            cpu.iter()
                .zip(self.cpu.iter().chain(std::iter::repeat(&0.0)))
                .map(|(now, before)| (now - before) * 60. / seconds)
                .collect(),
        );
        let writer = polylogue_store::timing::writer_seconds();
        polylogue_store::log!(
            progress,
            "index: {STORING_FILES}: {} of {} files stored, {:.0} MB added to the archive, \
             {:.1} MB/s over the last {:.0} s; database writer busy {:.0} s of them",
            self.files,
            pass.inputs_changed.max(self.files),
            pass.bytes_stored as f64 / 1e6,
            pass.bytes_stored.saturating_sub(self.stored) as f64 / 1e6 / seconds,
            seconds,
            writer - self.writer,
        );
        self.at = Instant::now();
        self.stored = pass.bytes_stored;
        self.writer = writer;
        self.cpu = cpu;
    }
}
