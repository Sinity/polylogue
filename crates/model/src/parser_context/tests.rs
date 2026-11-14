use super::*;
use std::collections::VecDeque;
struct Backend(VecDeque<Error>);
impl Context for Backend {
    fn get(&mut self, _: &Key) -> Result<Option<Value>, Error> {
        Err(self.0.pop_front().unwrap())
    }
    fn put(&mut self, _: Key, _: Value) -> Result<(), Error> {
        Ok(())
    }
}
fn key() -> Key {
    Key::Alias {
        namespace: "fixture".into(),
        uuid: "one".into(),
    }
}
#[test]
fn a_prior_backend_failure_is_not_reclassified_when_cancellation_arrives() {
    let handle = Handle::new(Backend(VecDeque::from([
        Error::Failed("ordinary backend failure".into()),
        Error::Cancelled,
    ])));
    assert!(handle.get(&key()).is_none());
    assert!(handle.get(&key()).is_none());
    assert!(!handle.is_cancelled());
    assert_eq!(handle.check().unwrap_err(), "ordinary backend failure");
}
#[test]
fn backend_cancellation_keeps_its_kind_across_the_provider_boundary() {
    let handle = Handle::new(Backend(VecDeque::from([Error::Cancelled])));
    assert!(handle.get(&key()).is_none());
    assert!(handle.is_cancelled());
    assert_eq!(handle.check().unwrap_err(), "read cancelled");
}
