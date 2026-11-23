use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard};

/// Recover the value returned by a lock or condition-variable wait after a panic.
pub fn unpoisoned<T>(result: std::sync::LockResult<T>) -> T {
    result.unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Access shared state even when an earlier owner panicked.
pub trait LockExt {
    type Guard<'a>
    where
        Self: 'a;
    fn locked(&self) -> Self::Guard<'_>;
}
impl<T: ?Sized> LockExt for Mutex<T> {
    type Guard<'a>
        = MutexGuard<'a, T>
    where
        T: 'a;
    fn locked(&self) -> Self::Guard<'_> {
        unpoisoned(self.lock())
    }
}
impl<T: ?Sized> LockExt for RwLock<T> {
    type Guard<'a>
        = RwLockReadGuard<'a, T>
    where
        T: 'a;
    fn locked(&self) -> Self::Guard<'_> {
        unpoisoned(self.read())
    }
}
/// Typed fields of a JSON object. Absent fields and other types return None.
pub trait JsonExt {
    fn str(&self, key: &str) -> Option<&str>;
    fn u64(&self, key: &str) -> Option<u64>;
    fn i64(&self, key: &str) -> Option<i64>;
    fn bool(&self, key: &str) -> Option<bool>;
    fn array(&self, key: &str) -> Option<&Vec<serde_json::Value>>;
    fn object(&self, key: &str) -> Option<&serde_json::Map<String, serde_json::Value>>;
}
impl JsonExt for serde_json::Value {
    fn str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }
    fn u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }
    fn i64(&self, key: &str) -> Option<i64> {
        self.get(key)?.as_i64()
    }
    fn bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }
    fn array(&self, key: &str) -> Option<&Vec<serde_json::Value>> {
        self.get(key)?.as_array()
    }
    fn object(&self, key: &str) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.get(key)?.as_object()
    }
}

/// Lowercase hexadecimal without a separator.
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        text.push(DIGITS[usize::from(byte >> 4)] as char);
        text.push(DIGITS[usize::from(byte & 15)] as char);
    }
    text
}

/// Start a thread whose name appears in panics, debuggers and `top -H`.
pub fn spawn_named<F, T>(
    name: impl Into<String>,
    f: F,
) -> std::io::Result<std::thread::JoinHandle<T>>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    std::thread::Builder::new().name(name.into()).spawn(f)
}

#[cfg(test)]
mod tests;
