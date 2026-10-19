//! The daemon (architecture 6): the only path. It holds the archive lock,
//! runs the convergence loop, keeps a pool of read connections and warm
//! caches (compiled queries, recent results and pages, embedding vectors,
//! completion data), and serves every client over one request shape.

pub mod measure;
pub mod query;
pub mod request;
pub mod server;

pub use request::{Request, Response};
