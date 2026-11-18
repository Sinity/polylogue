//! Display acknowledgements use the ordinary authored writer, without waiting.
use super::{Failure, service::Daemon};
use polylogue_model::{
    Reference,
    user::{SeenMark, UserWrite},
};
use polylogue_reading::operation::Invocation;
use polylogue_store::{Reader, writer::Queued};

#[expect(
    dead_code,
    reason = "Reading completion will submit display acknowledgements"
)]
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

#[expect(
    dead_code,
    reason = "Reading continuation will consult the last display acknowledgement"
)]
pub fn mark_of(
    reader: &Reader<'_>,
    author: &str,
    target: &Reference,
) -> Result<Option<SeenMark>, Failure> {
    let mut marks = reader.seen_marks(author, std::slice::from_ref(target))?;
    if marks.len() > 1 {
        return Err(Failure::conflict(format!(
            "multiple current seen marks for {target}"
        )));
    }
    Ok(marks.pop())
}
