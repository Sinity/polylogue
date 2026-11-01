//! Canonical custody of derived products in the shared SQL outbox.
use super::{Product, ProductWrite, Sinex};
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension, Transaction, params};

mod outbox;
pub(crate) use outbox::confirmed;
pub use outbox::stage_products;
pub(crate) use outbox::{SCHEMA, hold, publish, restore_custody, revoke};

fn pack(value: &impl serde::Serialize) -> Result<Vec<u8>> {
    Ok(crate::encoding::pack_bytes(&serde_json::to_vec(value)?)?)
}
fn product(bytes: &[u8]) -> Result<Product> {
    crate::encoding::unpack_json(bytes)
}

#[cfg(test)]
mod tests;
