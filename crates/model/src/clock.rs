//! An explicitly supplied wall clock. The binary owns the system-clock sampler;
//! libraries receive this handle and tests supply their own sequence of instants.
use std::sync::Arc;

#[derive(Clone)]
pub struct WallClock {
    sample: Arc<dyn Fn() -> jiff::Timestamp + Send + Sync>,
}
impl WallClock {
    pub fn new(sample: impl Fn() -> jiff::Timestamp + Send + Sync + 'static) -> Self {
        Self {
            sample: Arc::new(sample),
        }
    }
    pub fn fixed(instant: jiff::Timestamp) -> Self {
        Self::new(move || instant)
    }
    pub fn now(&self) -> jiff::Timestamp {
        (self.sample)()
    }
}
impl std::fmt::Debug for WallClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WallClock").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
