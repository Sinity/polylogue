use super::*;

#[test]
fn build_count_units_keep_their_status_words_and_wire_names() {
    let wire = r#"["bytes","files","records","indexes","sessions"]"#;
    assert_eq!(serde_json::to_string(&CountUnit::ALL).unwrap(), wire);
    assert_eq!(
        serde_json::from_str::<Vec<CountUnit>>(wire).unwrap(),
        CountUnit::ALL
    );
    assert_eq!(
        CountUnit::ALL.map(CountUnit::word),
        ["bytes", "files", "records", "indexes", "sessions"]
    );
}

const MB: u64 = 1_000_000;

fn bytes(done: u64) -> StepCount {
    StepCount {
        done,
        total: Some(10 * MB),
        unit: CountUnit::Bytes,
    }
}

fn since(seconds: i64) -> Timestamp {
    Timestamp::from_unix_ms(1_700_000_000_000 + seconds * 1000).unwrap()
}

/// A record in step `reading`, counted every ten seconds through `seconds`
/// with the bytes done by then; the writer is busy half of each second.
fn sampled(start: Instant, seconds: u64, done: impl Fn(u64) -> u64) -> Record {
    let mut record = Record::default();
    record.step("reading", since(0));
    for at in (0..=seconds).step_by(10) {
        record.count(
            start + Duration::from_secs(at),
            "reading",
            bytes(done(at)),
            || at as f64 / 2.,
        );
    }
    record
}

fn current(progress: &BuildProgress) -> &CurrentStep {
    progress.current.as_ref().expect("a current step")
}

#[test]
fn under_a_minute_of_samples_gives_a_rate_and_no_estimate() {
    let start = Instant::now();
    let record = sampled(start, 50, |at| at * 1_000);
    let progress = record.report(start + Duration::from_secs(50), 25.);
    let step = current(&progress);
    assert_eq!(step.per_second, Some(1_000.));
    assert_eq!(step.rate_seconds, Some(50.));
    assert_eq!(step.eta_seconds, None);
    assert!(!step.rate_varying);
}

#[test]
fn a_zero_rate_gives_no_estimate() {
    let start = Instant::now();
    let record = sampled(start, 120, |_| 3 * MB);
    let progress = record.report(start + Duration::from_secs(120), 60.);
    assert_eq!(current(&progress).per_second, Some(0.));
    assert_eq!(current(&progress).eta_seconds, None);
}

#[test]
fn the_rate_covers_the_last_five_minutes_and_estimates_what_is_left() {
    let start = Instant::now();
    // 1 kB/s for five minutes, then 3 kB/s.
    let done = |at: u64| {
        if at <= 300 {
            at * 1_000
        } else {
            300_000 + (at - 300) * 3_000
        }
    };
    let record = sampled(start, 600, done);
    let progress = record.report(start + Duration::from_secs(600), 300.);
    let step = current(&progress);
    assert_eq!(step.per_second, Some(3_000.));
    assert_eq!(step.rate_seconds, Some(300.));
    assert!(!step.rate_varying);
    // 10 MB less the 1.2 MB done, at 3 kB/s.
    assert_eq!(step.eta_seconds, Some(2_934));
    // No step is to come, so the step's estimate is the whole build's.
    assert_eq!(progress.eta_seconds, Some(2_934));
    // Half a writer second per second, over the last minute.
    assert_eq!(progress.writer_busy, Some(0.5));
}

#[test]
fn a_rate_that_changes_within_the_span_gives_no_estimate() {
    let start = Instant::now();
    // 1 kB/s for the first half of the five minutes, 3 kB/s for the second.
    let done = |at: u64| {
        if at <= 150 {
            at * 1_000
        } else {
            150_000 + (at - 150) * 3_000
        }
    };
    let record = sampled(start, 300, done);
    let progress = record.report(start + Duration::from_secs(300), 150.);
    let step = current(&progress);
    assert_eq!(step.per_second, Some(2_000.));
    assert!(step.rate_varying);
    assert_eq!(step.eta_seconds, None);
}

#[test]
fn a_step_without_a_total_has_a_rate_and_no_estimate() {
    let start = Instant::now();
    let mut record = Record::default();
    record.step("sorting", since(0));
    for at in (0..=120).step_by(10) {
        let count = StepCount {
            done: at * 100,
            total: None,
            unit: CountUnit::Records,
        };
        record.count(start + Duration::from_secs(at), "sorting", count, || 0.);
    }
    let progress = record.report(start + Duration::from_secs(120), 0.);
    let step = current(&progress);
    assert_eq!(step.per_second, Some(100.));
    assert_eq!(step.eta_seconds, None);
    assert!(!step.rate_varying);
}

#[test]
fn samples_are_ten_seconds_apart_and_cover_fifteen_minutes() {
    let start = Instant::now();
    let mut record = Record::default();
    record.step("reading", since(0));
    for at in 0..=1_800 {
        record.count(start + Duration::from_secs(at), "reading", bytes(at), || 0.);
    }
    assert_eq!(record.samples.len(), 91);
    assert_eq!(record.samples.front().map(|sample| sample.done), Some(900));
    assert_eq!(record.samples.back().map(|sample| sample.done), Some(1_800));
}

/// Steps finish as the next starts, with the time between and their last
/// count; the plan's steps after the current one are to come, and a
/// planned step the build passed over is not.
#[test]
fn steps_finish_in_order_and_the_plan_names_those_to_come() {
    let start = Instant::now();
    let mut record = Record::default();
    record.plan(&["reading", "sorting", "indexing", "installing"]);
    record.step("storing", since(0));
    // The writer's count of a step can come before the step is recorded.
    let indexes = StepCount {
        done: 3,
        total: Some(40),
        unit: CountUnit::Indexes,
    };
    record.count(start, "indexing", indexes.clone(), || 0.);
    let progress = record.report(start, 0.);
    assert_eq!(current(&progress).name, "storing");
    assert_eq!(
        progress.to_come,
        ["reading", "sorting", "indexing", "installing"]
    );
    record.step("reading", since(90));
    record.step("reading", since(95));
    // Nothing to sort: the build goes from reading to indexing.
    record.step("indexing", since(150));
    let progress = record.report(start, 0.);
    assert_eq!(
        progress.finished,
        [
            FinishedStep {
                name: "storing".into(),
                seconds: 90.,
                count: None,
            },
            FinishedStep {
                name: "reading".into(),
                seconds: 60.,
                count: None,
            },
        ]
    );
    assert_eq!(current(&progress).name, "indexing");
    assert_eq!(current(&progress).since, since(150));
    assert_eq!(current(&progress).count, Some(indexes));
    assert_eq!(progress.to_come, ["installing"]);
    // With a step to come there is no estimate for the whole build.
    assert_eq!(progress.eta_seconds, None);
}

#[test]
fn a_failed_attempt_starts_the_steps_again_and_stays_reported() {
    let start = Instant::now();
    let mut record = Record::default();
    record.plan(&["reading", "assembling"]);
    record.step("reading", since(0));
    record.step("assembling", since(60));
    record.failed("disk full".into());
    record.step("reading", since(70));
    let progress = record.report(start, 0.);
    assert_eq!(
        progress.failed_attempt,
        Some(FailedAttempt {
            step: Some("assembling".into()),
            error: "disk full".into(),
        })
    );
    assert!(progress.finished.is_empty());
    assert_eq!(current(&progress).name, "reading");
    assert_eq!(progress.to_come, ["assembling"]);
}

#[test]
fn the_workers_are_reported_while_recent() {
    let start = Instant::now();
    let mut record = Record::default();
    record.workers(start, vec![3., 4.5]);
    let progress = record.report(start + Duration::from_secs(30), 0.);
    assert_eq!(progress.worker_cpu_seconds, [3., 4.5]);
    assert_eq!(progress.current, None);
    // Worker figures older than two minutes are no longer reported.
    let later = record.report(start + Duration::from_secs(150), 0.);
    assert!(later.worker_cpu_seconds.is_empty());
}

/// Progress recorded once a minute, read between two records: the time
/// since the last record is not taken for slower work.
#[test]
fn progress_recorded_in_steps_is_steady_between_them() {
    let start = Instant::now();
    let mut record = Record::default();
    record.step("reading", since(0));
    for minute in 0..=5u64 {
        record.count(
            start + Duration::from_secs(minute * 60),
            "reading",
            bytes(minute * 60_000),
            || 0.,
        );
    }
    let progress = record.report(start + Duration::from_secs(330), 0.);
    let step = current(&progress);
    assert!(!step.rate_varying);
    // From the minute-one sample: 240 kB over 270 s.
    assert_eq!(step.rate_seconds, Some(270.));
    // 9.7 MB left at 240 kB per 270 s.
    assert_eq!(step.eta_seconds, Some(10_913));
}

#[test]
fn a_step_with_its_total_done_has_no_estimate() {
    let start = Instant::now();
    let record = sampled(
        start,
        120,
        |at| if at == 120 { 10 * MB } else { at * 1_000 },
    );
    let progress = record.report(start + Duration::from_secs(120), 0.);
    assert_eq!(current(&progress).eta_seconds, None);
    assert!(!current(&progress).rate_varying);
}
