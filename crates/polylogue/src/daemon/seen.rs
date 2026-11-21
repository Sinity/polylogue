//! Display acknowledgements use the ordinary authored writer, without waiting.
use super::service::Daemon;
use polylogue_model::user::{SeenMark, UserWrite};
use polylogue_reading::operation::Invocation;
use polylogue_store::writer::Queued;

pub fn record(daemon: &Daemon, invocation: &Invocation, marks: Vec<SeenMark>) {
    if marks.is_empty() {
        return;
    }
    let Some(writes) = &daemon.writes else {
        eprintln!("polylogue: seen marks: writer is unavailable");
        return;
    };
    let request = invocation.request(
        polylogue_store::user_writes::new_request_id(),
        UserWrite::MarkSeen { marks },
    );
    match writes.try_enqueue_user(request) {
        // A dropped receiver leaves commit ownership with the writer, which records a failure.
        Ok(Queued::Committing(_)) => {}
        Ok(Queued::Full(_)) => eprintln!("polylogue: seen marks: writer queue is full"),
        Err(error) => eprintln!("polylogue: seen marks: {error}"),
    }
}
