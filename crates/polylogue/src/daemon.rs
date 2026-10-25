//! The daemon (architecture 6): the only path. It holds the archive lock,
//! runs the convergence loop, keeps a pool of read connections and warm
//! caches (compiled queries, recent results and pages, embedding vectors,
//! completion data), and serves every client over one request shape.

pub(crate) mod attachment;
pub mod measure;
pub mod query;
pub mod request;
pub mod server;
pub mod service;

pub use request::{ReadTarget, Request, Response};
