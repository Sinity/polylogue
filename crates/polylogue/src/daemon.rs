//! The daemon: the only path to the archive. It holds the archive lock,
//! runs the loop, keeps a pool of read connections and its warm state
//! ([`warm`]: embedding vectors, query vectors, recent answers, completion
//! data), and serves every client over one request shape.

pub(crate) mod attachment;
pub mod calendar;
pub mod measure;
pub mod query;
pub mod request;
pub mod server;
pub mod service;
pub mod warm;

pub use request::{PageSelection, ReadTarget, Request, Response};
