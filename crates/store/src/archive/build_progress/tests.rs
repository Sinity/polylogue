use super::*;

const MB: u64 = 1_000_000;

fn build(bytes_read: u64, chains_read: u64) -> IndexBuild {
    IndexBuild {
        serving_previous: false,
        chains: 1_000,
        chains_read,
        bytes: 10 * MB,
        bytes_read,
    }
}

/// Samples every ten seconds through `seconds`, with the bytes and chains
/// read by then.
fn sampled(start: Instant, seconds: u64, read: impl Fn(u64) -> (u64, u64)) -> (Record, IndexBuild) {
    let mut record = Record::default();
    for at in (0..=seconds).step_by(10) {
        let (bytes, chains) = read(at);
        record.sample(start + Duration::from_secs(at), bytes, chains, || {
            at as f64 / 2.
        });
    }
    let (bytes, chains) = read(seconds);
    (record, build(bytes, chains))
}

#[test]
fn under_a_minute_of_samples_gives_a_rate_and_no_estimate() {
    let start = Instant::now();
    let (record, build) = sampled(start, 50, |at| (at * 1_000, at));
    let progress = record.report(start + Duration::from_secs(50), &build, 25.);
    assert_eq!(progress.bytes_per_second, Some(1_000.));
    assert_eq!(progress.rate_seconds, Some(50.));
    assert_eq!(progress.eta_seconds, None);
    assert!(!progress.rate_varying);
}

#[test]
fn a_zero_rate_gives_no_estimate() {
    let start = Instant::now();
    let (record, build) = sampled(start, 120, |_| (3 * MB, 7));
    let progress = record.report(start + Duration::from_secs(120), &build, 60.);
    assert_eq!(progress.bytes_per_second, Some(0.));
    assert_eq!(progress.chains_per_second, Some(0.));
    assert_eq!(progress.eta_seconds, None);
}

#[test]
fn the_rate_covers_the_last_five_minutes_and_estimates_what_is_left() {
    let start = Instant::now();
    // 1 kB/s for five minutes, then 3 kB/s and one chain a second.
    let read = |at: u64| {
        if at <= 300 {
            (at * 1_000, 0)
        } else {
            (300_000 + (at - 300) * 3_000, at - 300)
        }
    };
    let (record, build) = sampled(start, 600, read);
    let progress = record.report(start + Duration::from_secs(600), &build, 300.);
    assert_eq!(progress.bytes_per_second, Some(3_000.));
    assert_eq!(progress.chains_per_second, Some(1.));
    assert_eq!(progress.rate_seconds, Some(300.));
    assert!(!progress.rate_varying);
    // 10 MB less the 1.2 MB read, at 3 kB/s.
    assert_eq!(progress.eta_seconds, Some(2_934));
    // Half a writer second per second, over the last minute.
    assert_eq!(progress.writer_busy, Some(0.5));
}

#[test]
fn a_rate_that_changes_within_the_span_gives_no_estimate() {
    let start = Instant::now();
    // 1 kB/s for the first half of the five minutes, 3 kB/s for the second.
    let read = |at: u64| {
        if at <= 150 {
            (at * 1_000, 0)
        } else {
            (150_000 + (at - 150) * 3_000, 0)
        }
    };
    let (record, build) = sampled(start, 300, read);
    let progress = record.report(start + Duration::from_secs(300), &build, 150.);
    assert_eq!(progress.bytes_per_second, Some(2_000.));
    assert!(progress.rate_varying);
    assert_eq!(progress.eta_seconds, None);
}

#[test]
fn samples_are_ten_seconds_apart_and_cover_fifteen_minutes() {
    let start = Instant::now();
    let mut record = Record::default();
    for at in 0..=1_800 {
        record.sample(start + Duration::from_secs(at), at, 0, || 0.);
    }
    assert_eq!(record.samples.len(), 91);
    assert_eq!(
        record.samples.front().map(|sample| sample.bytes_read),
        Some(900)
    );
    assert_eq!(
        record.samples.back().map(|sample| sample.bytes_read),
        Some(1_800)
    );
}

#[test]
fn the_phase_and_the_workers_are_reported_as_recorded() {
    let start = Instant::now();
    let since = Timestamp::from_unix_ms(1_700_000_000_000).unwrap();
    let mut record = Record::default();
    record.phase("read chains", since);
    record.workers(start, vec![3., 4.5]);
    let progress = record.report(start + Duration::from_secs(30), &build(0, 0), 0.);
    assert_eq!(progress.phase.as_deref(), Some("read chains"));
    assert_eq!(progress.phase_since, Some(since));
    assert_eq!(progress.worker_cpu_seconds, [3., 4.5]);
    assert_eq!(progress.bytes_per_second, None, "no sample yet");
    // Worker figures older than two minutes are no longer reported.
    let later = record.report(start + Duration::from_secs(150), &build(0, 0), 0.);
    assert!(later.worker_cpu_seconds.is_empty());
}

/// Progress recorded once a minute, read between two steps: the time since
/// the last step is not taken for slower work.
#[test]
fn progress_recorded_in_steps_is_steady_between_them() {
    let start = Instant::now();
    let mut record = Record::default();
    for minute in 0..=5u64 {
        record.sample(
            start + Duration::from_secs(minute * 60),
            minute * 60_000,
            minute,
            || 0.,
        );
    }
    let progress = record.report(start + Duration::from_secs(330), &build(300_000, 5), 0.);
    assert!(!progress.rate_varying);
    // From the minute-one sample: 240 kB over 270 s.
    assert_eq!(progress.rate_seconds, Some(270.));
    // 9.7 MB left at 240 kB per 270 s.
    assert_eq!(progress.eta_seconds, Some(10_913));
}

#[test]
fn a_build_with_every_byte_read_has_no_estimate() {
    let start = Instant::now();
    let (record, _) = sampled(start, 120, |at| (at * 1_000, at));
    let mut read = build(10 * MB, 1_000);
    read.bytes = 10 * MB;
    let progress = record.report(start + Duration::from_secs(120), &read, 0.);
    assert_eq!(progress.eta_seconds, None);
    assert!(!progress.rate_varying);
}
