use super::*;
use polylogue_engine::{Engine, Lifetime, LoopMsg, LoopStatus};
use polylogue_store::Archive;
use polylogue_store::fault::{self, Action};
use polylogue_store::primary::{Confirmed, ConfirmedChange, ReplayControl};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

fn wait_for_passes(status: &Mutex<LoopStatus>, after: u64) -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while status.lock().unwrap().passes <= after {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    true
}

#[test]
fn a_fenced_pass_leaves_the_next_recovery_commit_able_to_wake_the_loop() {
    let home = polylogue_testkit::ClaudeHome::new();
    let sdk = Arc::new(client(&home.dir.path().join("producer")));
    let (archive, lock) = Archive::open_primary(&home.root, &sdk.config.store).unwrap();
    let mut config = home.config(1 << 20);
    config.reads_sources = false;
    config.live_every = Duration::from_secs(60);
    let mut engine =
        Engine::on_sinex(config, archive, sdk, polylogue_testkit::stopped_attempt()).unwrap();
    let writer = engine.writer();
    let (tx, rx) = crossbeam_channel::unbounded();
    let run_tx = tx.clone();
    let (ready, started) = mpsc::sync_channel(1);
    let (release, held) = mpsc::sync_channel(1);
    let status = Arc::new(Mutex::new(LoopStatus::default()));
    let running = Arc::clone(&status);
    let worker = std::thread::spawn(move || {
        let _lock = lock;
        engine.run(run_tx, rx, &running, Lifetime::UntilStopped, || {
            ready.send(()).unwrap();
            let _ = held.recv();
        })
    });
    started.recv_timeout(Duration::from_secs(5)).unwrap();
    let reset = || {
        vec![Confirmed {
            position: 0,
            event: String::new(),
            output_index: 0,
            sinex_context: None,
            change: ConfirmedChange::Control(Box::new(ReplayControl::Reset {
                scope: "wake-recovery".into(),
            })),
        }]
    };
    // Queue a coalesced content cue before fencing its first turn. The old
    // ordering refused the scope before consuming the subscription's token.
    writer.apply_confirmed(reset()).unwrap();
    fault::clear();
    fault::arm("primary.scope_reset_commit", "", Action::Fail, 0, 1);
    let fenced = writer
        .apply_confirmed(vec![Confirmed {
            position: 0,
            event: String::new(),
            output_index: 0,
            sinex_context: None,
            change: ConfirmedChange::Control(Box::new(ReplayControl::Reset {
                scope: "changed-wake-recovery".into(),
            })),
        }])
        .unwrap_err();
    fault::clear();
    assert!(
        fenced.to_string().contains("primary.scope_reset_commit"),
        "{fenced}"
    );
    release.send(()).unwrap();
    // One turn consumes the inbox cue; another exhausts its latched wake.
    // Both refuse the scope, then wait without a periodic tick.
    let reached_fence = wait_for_passes(&status, 1);
    let before = status.lock().unwrap().passes;
    writer.apply_confirmed(reset()).unwrap();
    let resumed = wait_for_passes(&status, before);
    tx.send(LoopMsg::Stop).unwrap();
    worker.join().unwrap().unwrap();
    assert!(
        reached_fence,
        "fenced turns did not run: {:?}",
        status.lock().unwrap()
    );
    assert!(
        resumed,
        "the successful reset's wake was lost: {:?}",
        status.lock().unwrap()
    );
    assert!(
        status.lock().unwrap().passes < 20,
        "an empty recovered loop spun"
    );
}
