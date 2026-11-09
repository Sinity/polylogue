//! The daemon: the only path to the archive. It holds the archive lock,
//! runs the loop, keeps a pool of read connections and its warm state
//! ([`warm`]: embedding vectors, query vectors, completion
//! data), and serves every client over one request format: an operation, its typed
//! arguments and the request's identity and deadline.

pub(crate) mod attachment;
pub mod calendar;
pub(crate) mod disconnect;
pub mod execution;
mod failure;
pub(crate) mod handler;
pub(crate) mod items;
pub(crate) mod lifecycle;
pub mod measure;
pub mod query;
pub(crate) mod queued_deliveries;
pub(crate) mod schema;
pub mod scope;
pub mod server;
pub mod service;
mod verify;
pub mod warm;

pub use failure::Failure;
