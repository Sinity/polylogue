//! The engine's integration tests, one binary so the engine, store and
//! testkit are linked once:
//!
//! - `passes`: the loop over files a test writes;
//! - `properties`: properties every archive keeps;
//! - `faults`: properties that need a failure injected at a chosen point.

mod faults;
mod passes;
mod properties;
