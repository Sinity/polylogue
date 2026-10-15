//! Finite caller-owned lifetime witness, not a Sinex unlock service or a blob-store policy.
use hashed_blobs::{DataKey, Hash, Store};
use std::io::Cursor;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Principal { Operator, Agent }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Ticket { session: u64, key_revision: u64, principal: Principal }
struct State {
    session: u64,
    deadline: Option<u64>,
    now: u64,
    key_revision: u64,
    key_live: bool,
    cache: Option<(Ticket, Vec<u8>)>,
    // A bounded transport handoff. Already handed-off bytes are outside revocation.
    handed_off: Vec<Vec<u8>>,
}
struct Owner(Mutex<State>);
impl Owner {
    fn new() -> Self {
        Self(Mutex::new(State { session: 0, deadline: None, now: 0,
            key_revision: 0, key_live: true, cache: None, handed_off: vec![] }))
    }
    fn revoke(s: &mut State) { s.session += 1; s.deadline = None; s.cache = None; }
    fn valid(s: &State, t: Ticket) -> bool {
        t.principal == Principal::Operator && s.session == t.session
            && s.key_revision == t.key_revision && s.key_live
            && s.deadline.is_some_and(|d| s.now < d)
    }
    fn unlock(&self, until: u64) {
        let mut s = self.0.lock().unwrap();
        Self::revoke(&mut s);
        assert!(until > s.now);
        s.deadline = Some(until);
    }
    fn lock(&self) { Self::revoke(&mut self.0.lock().unwrap()); }
    fn advance_clock(&self, now: u64) {
        let mut s = self.0.lock().unwrap();
        assert!(now >= s.now);
        s.now = now;
        if s.deadline.is_some_and(|d| now >= d) { Self::revoke(&mut s); }
    }
    fn purge(&self) {
        let mut s = self.0.lock().unwrap();
        s.key_live = false; s.key_revision += 1; s.cache = None;
    }
    fn begin(&self, principal: Principal) -> Result<Ticket, String> {
        let s = self.0.lock().unwrap();
        let t = Ticket { session: s.session, key_revision: s.key_revision, principal };
        if Self::valid(&s, t) { Ok(t) } else { Err("sealed or unauthorized".into()) }
    }
    // Fixed synthetic key provider stands in for authorized age unwrapping. No key backup policy here.
    fn unwrap(&self, t: Ticket, key_ref: &str) -> Result<[u8; 32], String> {
        let s = self.0.lock().unwrap();
        if key_ref == "material-one" && Self::valid(&s, t) { Ok([7; 32]) }
        else { Err("revoked, purged or unauthorized".into()) }
    }
    fn finish(&self, t: Ticket, bytes: Vec<u8>, fence: bool, in_guard: impl FnOnce()) -> bool {
        let mut s = self.0.lock().unwrap();
        // The deliberately disabled branch is used only by the negative control below.
        if fence && !Self::valid(&s, t) { return false; }
        in_guard();
        s.cache = Some((t, bytes.clone()));
        s.handed_off.push(bytes);
        true
    }
    fn cached(&self, principal: Principal) -> Option<Vec<u8>> {
        let s = self.0.lock().unwrap();
        let (t, value) = s.cache.as_ref()?;
        if t.principal == principal && Self::valid(&s, *t) { Some(value.clone()) } else { None }
    }
    fn counts(&self) -> (usize, usize) {
        let s = self.0.lock().unwrap();
        (usize::from(s.cache.is_some()), s.handed_off.len())
    }
}

fn fixture() -> (tempfile::TempDir, Arc<Store>, Hash, Arc<Owner>) {
    let d = tempfile::tempdir().unwrap();
    let store = Arc::new(Store::open(d.path().join("blobs"), vec![], 0o600).unwrap()
        .with_frame_size(64));
    let key = DataKey { key_ref: "material-one".into(), key: [7; 32] };
    let blob = store.put_sealed(Cursor::new(b"synthetic private value"), 3, &key).unwrap();
    (d, store, blob, Arc::new(Owner::new()))
}

#[derive(Clone, Copy)]
enum Pause { AfterUnwrap, BeforePublication }
#[derive(Clone, Copy)]
enum Transition { Lock, Expire, RelockUnlock, Purge }

fn revoked_schedule(pause: Pause, transition: Transition, fence: bool) -> (bool, usize, usize) {
    let (_d, store, blob, owner) = fixture();
    owner.unlock(10);
    let t = owner.begin(Principal::Operator).unwrap();
    let entered = Arc::new(Barrier::new(2));
    let resume = Arc::new(Barrier::new(2));
    let (o, e, r) = (owner.clone(), entered.clone(), resume.clone());
    let worker = thread::spawn(move || {
        let unwrap = |key_ref: &str| {
            let key = o.unwrap(t, key_ref)?;
            if matches!(pause, Pause::AfterUnwrap) { e.wait(); r.wait(); }
            Ok(key)
        };
        // Actual authenticated sealed range decode; no parallel mock decoder.
        let bytes = store.read_range_with(&blob, 0, 23, &unwrap).unwrap();
        if matches!(pause, Pause::BeforePublication) { e.wait(); r.wait(); }
        o.finish(t, bytes, fence, || {})
    });
    entered.wait();
    match transition {
        Transition::Lock => owner.lock(),
        Transition::Expire => owner.advance_clock(10),
        Transition::RelockUnlock => { owner.lock(); owner.unlock(20); }
        Transition::Purge => owner.purge(),
    }
    // Revocation has acknowledged and owned cache is empty before old work resumes.
    assert_eq!(owner.counts(), (0, 0));
    resume.wait();
    let published = worker.join().unwrap();
    let (cache, outputs) = owner.counts();
    (published, cache, outputs)
}

#[test]
fn late_publication_negative_control_and_guarded_schedules() {
    assert_eq!(revoked_schedule(Pause::BeforePublication, Transition::Lock, false), (true, 1, 1),
        "disabled fence must reproduce a late plaintext cache refill and output handoff");
    for pause in [Pause::AfterUnwrap, Pause::BeforePublication] {
        for transition in [Transition::Lock, Transition::Expire, Transition::RelockUnlock, Transition::Purge] {
            assert_eq!(revoked_schedule(pause, transition, true), (false, 0, 0));
        }
    }
    println!("unsafe late refill reproduced; fenced lock/expiry/ABA/purge refuse after unwrap and before publication");
}

#[test]
fn cache_authorization_reunlock_and_previously_disclosed_bytes() {
    let (_d, store, blob, owner) = fixture();
    owner.unlock(10);
    let t = owner.begin(Principal::Operator).unwrap();
    let bytes = store.read_range_with(&blob, 0, 23, &|r| owner.unwrap(t, r)).unwrap();
    assert!(owner.finish(t, bytes.clone(), true, || {}));
    assert_eq!(owner.cached(Principal::Operator), Some(bytes.clone()));
    assert!(owner.begin(Principal::Agent).is_err());
    assert_eq!(owner.cached(Principal::Agent), None);
    owner.lock();
    assert_eq!(owner.counts(), (0, 1));
    assert_eq!(bytes, b"synthetic private value", "already disclosed/captured plaintext is not revoked");
    owner.unlock(20);
    let new_ticket = owner.begin(Principal::Operator).unwrap();
    assert_ne!(t.session, new_ticket.session);
    assert_eq!(owner.cached(Principal::Operator), None);
    assert!(!owner.finish(t, bytes, true, || {}));
    let fresh = store.read_range_with(&blob, 0, 23, &|r| owner.unwrap(new_ticket, r)).unwrap();
    assert!(owner.finish(new_ticket, fresh, true, || {}));
    assert_eq!(owner.counts(), (1, 2));
    owner.purge();
    assert_eq!(owner.counts(), (0, 2));
    assert!(owner.unwrap(new_ticket, "material-one").is_err());
}

#[test]
fn publication_handoff_precedes_lock_acknowledgement_when_it_wins_guard() {
    let (_d, store, blob, owner) = fixture();
    owner.unlock(10);
    let t = owner.begin(Principal::Operator).unwrap();
    let bytes = store.read_range_with(&blob, 0, 23, &|r| owner.unwrap(t, r)).unwrap();
    let entered = Arc::new(Barrier::new(2));
    let resume = Arc::new(Barrier::new(2));
    let (o, e, r) = (owner.clone(), entered.clone(), resume.clone());
    let publisher = thread::spawn(move || o.finish(t, bytes, true, || { e.wait(); r.wait(); }));
    entered.wait(); // The publisher holds the same guard required by lock.
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (ack_tx, ack_rx) = std::sync::mpsc::channel();
    let o = owner.clone();
    let locker = thread::spawn(move || { started_tx.send(()).unwrap(); o.lock(); ack_tx.send(()).unwrap(); });
    started_rx.recv().unwrap();
    assert!(matches!(ack_rx.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)));
    resume.wait();
    assert!(publisher.join().unwrap());
    ack_rx.recv().unwrap();
    locker.join().unwrap();
    assert_eq!(owner.counts(), (0, 1), "prior handoff survives, owned plaintext cache is removed");
}
