//! The bounded executor that runs every request's application work.
//!
//! A fixed number of places bound how much work runs at once: `ordinary`
//! places for requests, one place reserved for interactive answers (status,
//! completion, brief) and one background place for idle-priority work
//! (schema inference, verification, imports). Queued work starts in the
//! order of its deadline, so a short interactive request is not served after
//! a long read accepted earlier; work without a deadline queues as if it had
//! the default read deadline, so it still starts in its turn. Work that
//! reads only the daemon's memory (status without its deep checks) takes no
//! place: it runs at once on the calling thread, so no queued or waiting
//! work can delay it.
//!
//! Threads are not places. Work that must wait for something outside the
//! daemon's own computation (the next commit, the writer's result) calls
//! [`outside`], which gives its place to queued work for the wait and queues
//! for a place again afterwards. Threads are started as places need them and
//! as many as there are places stay parked for the next work; parked threads
//! wake only when work arrives.

use polylogue_model::LockExt as _;
use std::collections::BinaryHeap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// How long a stop waits for running work.
pub(crate) const STOP_WAIT: Duration = Duration::from_secs(5);

/// Which places a piece of work may take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    /// Work that reads only the daemon's memory: no place, run at once on
    /// the calling thread.
    Immediate,
    /// Completion and brief: an ordinary place or the reserved one.
    Interactive,
    /// Every other request: an ordinary place.
    Ordinary,
    /// Idle-priority work: the background place only.
    Background,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    Reserved,
    Ordinary,
    Background,
}

impl Place {
    fn index(self) -> usize {
        match self {
            Self::Reserved => 0,
            Self::Ordinary => 1,
            Self::Background => 2,
        }
    }
}

impl Class {
    fn index(self) -> usize {
        match self {
            Self::Immediate => unreachable!("immediate work is never queued"),
            Self::Interactive => 0,
            Self::Ordinary => 1,
            Self::Background => 2,
        }
    }
}

type Job = Box<dyn FnOnce() + Send>;

enum Work {
    Job(Job),
    /// A thread back from [`outside`] waiting for a place.
    Resume(Arc<Resumption>),
}

#[derive(Default)]
struct Resumption {
    /// The place granted, or `None` when the executor stopped meanwhile.
    granted: Mutex<Option<Option<Place>>>,
    ready: Condvar,
}

struct Queued {
    deadline: Instant,
    sequence: u64,
    class: Class,
    work: Work,
}

impl PartialEq for Queued {
    fn eq(&self, other: &Self) -> bool {
        (self.deadline, self.sequence) == (other.deadline, other.sequence)
    }
}
impl Eq for Queued {}
impl PartialOrd for Queued {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Queued {
    /// The heap pops its greatest entry: the earliest deadline, then the first
    /// accepted.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (other.deadline, other.sequence).cmp(&(self.deadline, self.sequence))
    }
}

#[derive(Default)]
struct State {
    /// Queued work by class.
    queues: [BinaryHeap<Queued>; 3],
    /// Free places by place kind.
    free: [usize; 3],
    parked: usize,
    threads: usize,
    /// Jobs started and not finished.
    running: usize,
    sequence: u64,
    stopping: bool,
}

struct Shared {
    state: Mutex<State>,
    work: Condvar,
    finished: Condvar,
    places: usize,
}

pub(crate) struct Executor {
    shared: Arc<Shared>,
}

/// The place the work running on this thread holds.
struct Holding {
    shared: Arc<Shared>,
    place: Option<Place>,
    class: Class,
    deadline: Instant,
}

thread_local! {
    static HOLDING: std::cell::RefCell<Option<Holding>> = const { std::cell::RefCell::new(None) };
}

impl Executor {
    /// An executor with `ordinary` request places, one reserved interactive
    /// place and one background place.
    pub(crate) fn new(ordinary: usize) -> Self {
        let ordinary = ordinary.max(1);
        let mut state = State::default();
        state.free[Place::Reserved.index()] = 1;
        state.free[Place::Ordinary.index()] = ordinary;
        state.free[Place::Background.index()] = 1;
        Self {
            shared: Arc::new(Shared {
                state: Mutex::new(state),
                work: Condvar::new(),
                finished: Condvar::new(),
                places: ordinary + 2,
            }),
        }
    }

    /// Queues `job`; `false` when the executor has stopped and the job was
    /// dropped unrun.
    pub(crate) fn submit(
        &self,
        class: Class,
        deadline: Option<Instant>,
        job: impl FnOnce() + Send + 'static,
    ) -> bool {
        if class == Class::Immediate {
            if self.shared.state.locked().stopping {
                return false;
            }
            job();
            return true;
        }
        let deadline = effective_deadline(deadline);
        let mut state = self.shared.state.locked();
        if state.stopping {
            return false;
        }
        state.sequence += 1;
        let sequence = state.sequence;
        state.queues[class.index()].push(Queued {
            deadline,
            sequence,
            class,
            work: Work::Job(Box::new(job)),
        });
        self.shared.dispatch(&mut state);
        true
    }

    /// Runs `job` and returns its result; `None` when the executor stopped
    /// before running it.
    pub(crate) async fn run<T: From<polylogue_model::Panicked> + Send + 'static>(
        &self,
        class: Class,
        deadline: Option<Instant>,
        job: impl FnOnce() -> T + Send + 'static,
    ) -> Option<T> {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        if !self.submit(class, deadline, move || {
            let answer = polylogue_model::caught(job).unwrap_or_else(T::from);
            let _ = sender.send(answer);
        }) {
            return None;
        }
        receiver.await.ok()
    }

    /// Refuses new work, drops queued work unrun and waits for running work
    /// to end, at most `wait`. Returns how many jobs were still running then:
    /// work that never checks its cancellation does not hold up a stop.
    /// Call it from outside the executor's own threads.
    pub(crate) fn stop(&self, wait: Duration) -> usize {
        let until = Instant::now() + wait;
        self.refuse();
        let mut state = self.shared.state.locked();
        while state.threads > 0 {
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return state.running;
            }
            state = polylogue_model::unpoisoned(self.shared.finished.wait_timeout(state, left)).0;
        }
        0
    }

    /// Every place: the ordinary ones, the reserved one and the background one.
    #[cfg(test)]
    pub(crate) fn places(&self) -> usize {
        self.shared.places
    }

    #[cfg(test)]
    pub(crate) fn threads(&self) -> usize {
        self.shared.state.locked().threads
    }

    /// Work started and not yet finished, holding a place or waiting
    /// outside one.
    #[cfg(test)]
    pub(crate) fn running(&self) -> usize {
        self.shared.state.locked().running
    }

    #[cfg(test)]
    pub(crate) fn queued(&self) -> usize {
        self.shared
            .state
            .locked()
            .queues
            .iter()
            .map(BinaryHeap::len)
            .sum()
    }
}

impl Executor {
    /// Refuses new jobs and releases queued callers without waiting for work
    /// that already owns its execution. Transport drain can then precede join.
    pub(crate) fn refuse(&self) {
        let mut state = self.shared.state.locked();
        state.stopping = true;
        let queued: Vec<_> = state.queues.iter_mut().flat_map(std::mem::take).collect();
        for queued in queued {
            match queued.work {
                Work::Resume(resumption) => resumption.grant(None),
                // Dropping a job drops its answer channel: its caller hears
                // that the daemon is stopping.
                Work::Job(job) => drop(job),
            }
        }
        self.shared.work.notify_all();
    }
}

impl Drop for Executor {
    /// The last holder may be a job on one of the executor's threads, so
    /// dropping refuses work without waiting for it.
    fn drop(&mut self) {
        self.refuse();
    }
}

/// Work without a deadline queues as if it had the default read deadline.
fn effective_deadline(deadline: Option<Instant>) -> Instant {
    deadline.unwrap_or_else(|| Instant::now() + super::execution::default_read_deadline())
}

impl Resumption {
    fn grant(&self, place: Option<Place>) {
        *self.granted.locked() = Some(place);
        self.ready.notify_one();
    }
}

impl Shared {
    /// The queued work to start now and the place it takes.
    fn take(state: &mut State) -> Option<(Queued, Place)> {
        let (class, place) = Self::next_start(state)?;
        let queued = state.queues[class.index()].pop()?;
        state.free[place.index()] -= 1;
        Some((queued, place))
    }

    /// The class of the earliest-deadline queued work that has a free place
    /// it may use, with that place.
    fn next_start(state: &State) -> Option<(Class, Place)> {
        let free = |place: Place| state.free[place.index()] > 0;
        let interactive = if free(Place::Reserved) {
            Some(Place::Reserved)
        } else if free(Place::Ordinary) {
            Some(Place::Ordinary)
        } else {
            None
        };
        let candidates = [
            (Class::Interactive, interactive),
            (
                Class::Ordinary,
                free(Place::Ordinary).then_some(Place::Ordinary),
            ),
            (
                Class::Background,
                free(Place::Background).then_some(Place::Background),
            ),
        ];
        candidates
            .into_iter()
            .filter_map(|(class, place)| {
                let head = state.queues[class.index()].peek()?;
                Some(((head.deadline, head.sequence), class, place?))
            })
            .min_by_key(|(order, _, _)| *order)
            .map(|(_, class, place)| (class, place))
    }

    fn runnable(state: &State) -> bool {
        Self::next_start(state).is_some()
    }

    /// Wakes a parked thread, or starts one, when queued work can start.
    fn dispatch(self: &Arc<Self>, state: &mut State) {
        if !Self::runnable(state) {
            return;
        }
        if state.parked > 0 {
            self.work.notify_one();
            return;
        }
        let shared = Arc::clone(self);
        match polylogue_model::spawn_named("polylogue-request", move || shared.serve()) {
            Ok(_) => state.threads += 1,
            Err(error) => polylogue_store::log!(error, "start a request thread: {error}"),
        }
    }

    fn release(self: &Arc<Self>, place: Place) {
        let mut state = self.state.locked();
        state.free[place.index()] += 1;
        self.dispatch(&mut state);
    }

    fn serve(self: Arc<Self>) {
        loop {
            let mut state = self.state.locked();
            let (queued, place) = loop {
                if let Some(taken) = Self::take(&mut state) {
                    break taken;
                }
                if state.stopping || state.parked >= self.places {
                    state.threads -= 1;
                    self.finished.notify_all();
                    return;
                }
                state.parked += 1;
                state = polylogue_model::unpoisoned(self.work.wait(state));
                state.parked -= 1;
            };
            // More work may be startable than this thread can take.
            self.dispatch(&mut state);
            if matches!(queued.work, Work::Job(_)) {
                state.running += 1;
            }
            drop(state);
            let job = match queued.work {
                Work::Resume(resumption) => {
                    resumption.grant(Some(place));
                    continue;
                }
                Work::Job(job) => job,
            };
            HOLDING.with(|holding| {
                *holding.borrow_mut() = Some(Holding {
                    shared: Arc::clone(&self),
                    place: Some(place),
                    class: queued.class,
                    deadline: queued.deadline,
                })
            });
            if let Err(panic) = polylogue_model::caught(job) {
                polylogue_store::log!(error, "a request job {panic}");
            }
            let held = HOLDING.with(|holding| holding.borrow_mut().take());
            let mut state = self.state.locked();
            state.running -= 1;
            if let Some(place) = held.and_then(|held| held.place) {
                state.free[place.index()] += 1;
                self.dispatch(&mut state);
            }
        }
    }
}

/// Runs `wait` without holding an executor place: the place of the work
/// running on this thread goes to queued work, and the work queues again at
/// its original deadline when `wait` returns. On a thread outside the executor
/// it just runs `wait`.
pub(crate) fn outside<T>(wait: impl FnOnce() -> T) -> T {
    let Some(holding) = HOLDING.with(|holding| holding.borrow_mut().take()) else {
        return wait();
    };
    if let Some(place) = holding.place {
        holding.shared.release(place);
    }
    let result = wait();
    let resumption = Arc::new(Resumption::default());
    {
        let mut state = holding.shared.state.locked();
        if state.stopping {
            drop(state);
            HOLDING.with(|current| {
                *current.borrow_mut() = Some(Holding {
                    place: None,
                    ..holding
                })
            });
            return result;
        }
        state.sequence += 1;
        let sequence = state.sequence;
        state.queues[holding.class.index()].push(Queued {
            deadline: holding.deadline,
            sequence,
            class: holding.class,
            work: Work::Resume(Arc::clone(&resumption)),
        });
        holding.shared.dispatch(&mut state);
    }
    let mut granted = resumption.granted.locked();
    let place = loop {
        if let Some(place) = *granted {
            break place;
        }
        granted = polylogue_model::unpoisoned(resumption.ready.wait(granted));
    };
    drop(granted);
    HOLDING.with(|current| *current.borrow_mut() = Some(Holding { place, ..holding }));
    result
}

#[cfg(test)]
mod tests;
