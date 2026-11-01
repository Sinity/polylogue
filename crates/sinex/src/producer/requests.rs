//! Authored requests waiting in the spool, read back for the stale-base check
//! of the next authored write on this host.
use crate::spool::{Prepared, Spool};
use crate::{Error, Result};
use polylogue_model::authored::{PendingAuthored, PreparedRequest};
use polylogue_store::primary::{Product, ProductWrite};
use sinex_protocol::Ingress;

pub(super) fn visit(
    queue: &Spool,
    entry: &Prepared,
    visit: &mut dyn FnMut(&PendingAuthored) -> polylogue_store::Result<()>,
) -> Result<()> {
    let Ingress::Material { material, .. } = queue.decode_frame(&entry.identity)? else {
        return Err(Error::Invalid("authored intent has no material".into()));
    };
    match material.parser.as_str() {
        "polylogue-request" => {
            let prepared: PreparedRequest =
                crate::spool::decode_json(queue.body(&entry.identity)?)?;
            visit(&PendingAuthored::Live { prepared }).map_err(Error::Store)
        }
        "polylogue-request-restore" => {
            let writes: Vec<ProductWrite> =
                crate::spool::decode_json(queue.body(&entry.identity)?)?;
            for write in writes {
                let pending = match write.product {
                    Product::RestoredRequest {
                        prepared,
                        claims,
                        current,
                    } => PendingAuthored::Restore {
                        prepared,
                        claims,
                        current,
                    },
                    Product::RestoredHistory {
                        evidence,
                        claims,
                        current,
                    } => PendingAuthored::RestoreHistory {
                        evidence,
                        claims,
                        current,
                    },
                    Product::RestoredAssertion {
                        fact,
                        claims,
                        current,
                    } => PendingAuthored::RestoreAssertion {
                        fact,
                        claims,
                        current,
                    },
                    _ => {
                        return Err(Error::Invalid(
                            "non-restoration product in authored restore frame".into(),
                        ));
                    }
                };
                visit(&pending).map_err(Error::Store)?;
            }
            Ok(())
        }
        _ => Err(Error::Invalid("unknown authored material parser".into())),
    }
}
