//! Native reference membership is shared by lowering, pinned reads and metadata fingerprints.
use polylogue_model::Reference;
use polylogue_query::ast::Unit;

pub(crate) fn membership(
    unit: Unit,
    reference: &Reference,
    bind: &mut dyn FnMut(String) -> String,
) -> Option<String> {
    if let Reference::Span { container, .. } = reference {
        return membership(unit, container, bind);
    }
    let messages = match reference {
        Reference::Message(id) => Some(format!(
            "SELECT m.pk,m.session_pk FROM message m WHERE m.id={}",
            bind(id.to_string())
        )),
        Reference::Block { message, n } => {
            let n = i64::try_from(*n).ok()?;
            Some(format!(
                "SELECT m.pk,m.session_pk FROM message m WHERE m.id={} AND EXISTS(SELECT 1 FROM block b WHERE b.message_pk=m.pk AND b.idx={n})",
                bind(message.to_string())
            ))
        }
        Reference::MessageRange { first, last } => {
            let first = bind(first.to_string());
            let last = bind(last.to_string());
            Some(format!(
                "SELECT m.pk,m.session_pk FROM message m JOIN message f ON f.id={first} JOIN message l ON l.id={last} WHERE m.session_pk=f.session_pk AND f.session_pk=l.session_pk AND (m.order_input_id,m.seq,m.record_position,m.id)>=(f.order_input_id,f.seq,f.record_position,f.id) AND (m.order_input_id,m.seq,m.record_position,m.id)<=(l.order_input_id,l.seq,l.record_position,l.id)"
            ))
        }
        _ => None,
    };
    Some(match unit {
        Unit::Sessions => match reference {
            Reference::Session(id) => {
                format!("SELECT pk FROM session WHERE id={}", bind(id.to_string()))
            }
            _ => {
                let messages = messages?;
                format!(
                    "SELECT q.session_pk AS pk FROM ({messages}) q JOIN message m ON m.pk=q.pk WHERE {}",
                    crate::session_sql::own_messages("m", None)
                )
            }
        },
        Unit::Messages => match reference {
            Reference::Session(id) => format!(
                "SELECT m.pk FROM message m JOIN session s ON s.pk=m.session_pk WHERE s.id={} AND {}",
                bind(id.to_string()),
                crate::session_sql::own_messages("m", None)
            ),
            _ => {
                let messages = messages?;
                format!("SELECT pk FROM ({messages})")
            }
        },
        Unit::Blocks | Unit::ToolCalls => {
            let messages = messages?;
            let extra = if let Reference::Block { n, .. } = reference {
                let Ok(n) = i64::try_from(*n) else {
                    return None;
                };
                format!(" AND b.idx={n}")
            } else {
                String::new()
            };
            if unit == Unit::Blocks {
                format!(
                    "SELECT b.pk FROM block b WHERE b.message_pk IN (SELECT pk FROM ({messages})){extra}"
                )
            } else {
                format!(
                    "SELECT t.pk FROM tool_call t JOIN block b ON b.pk=t.call_block_pk WHERE b.message_pk IN (SELECT pk FROM ({messages})){extra}"
                )
            }
        }
        Unit::Documents => {
            let Reference::Document(id) = reference else {
                return None;
            };
            format!("SELECT pk FROM document WHERE id={}", bind(id.to_string()))
        }
        Unit::Events => {
            let Reference::Event(id) = reference else {
                return None;
            };
            format!("SELECT pk FROM event WHERE id={}", bind(id.clone()))
        }
        Unit::Assertions => {
            let Reference::Assertion(revision) = reference else {
                return None;
            };
            let root = bind(revision.root.clone());
            let exact = revision
                .revision
                .as_ref()
                .map(|id| format!(" AND n.id={}", bind(id.clone())))
                .unwrap_or_default();
            format!("SELECT n.rowid AS pk FROM source_assertion n WHERE n.root={root}{exact}")
        }
        Unit::Deliveries => {
            let Reference::Delivery(id) = reference else {
                return None;
            };
            format!(
                "SELECT rowid AS pk FROM source_delivery WHERE id={}",
                bind(id.clone())
            )
        }
        _ => return None,
    })
}

/// Recorded conversation memberships include every contributing native source,
/// rather than only the session selected for the deduplicated shown message.
pub(crate) fn message_sessions(message: &str) -> (&'static str, &str) {
    (
        "SELECT DISTINCT session.id FROM message_source source JOIN message ON message.pk=source.message_pk JOIN session ON session.pk=source.session_pk WHERE message.id=?1 ORDER BY session.id",
        message,
    )
}
