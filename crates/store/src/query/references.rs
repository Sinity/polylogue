//! Reference membership in a unit's own rows is shared by lowering, pinned reads and metadata fingerprints.
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
            crate::chain_names::resolved_message_sql(id, bind)
        )),
        Reference::Block { message, n } => {
            let n = i64::try_from(*n).ok()?;
            Some(format!(
                "SELECT m.pk,m.session_pk FROM message m WHERE m.id={} AND EXISTS(SELECT 1 FROM block b WHERE b.message_pk=m.pk AND b.position={n})",
                crate::chain_names::resolved_message_sql(message, bind)
            ))
        }
        Reference::MessageRange { first, last } => {
            let first = crate::chain_names::resolved_message_sql(first, bind);
            let last = crate::chain_names::resolved_message_sql(last, bind);
            Some(format!(
                "SELECT m.pk,m.session_pk FROM message m JOIN message f ON f.id={first} JOIN message l ON l.id={last} WHERE m.session_pk=f.session_pk AND f.session_pk=l.session_pk AND (m.session_order,m.id)>=(f.session_order,f.id) AND (m.session_order,m.id)<=(l.session_order,l.id)"
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
        Unit::ToolCalls if matches!(reference, Reference::ToolCall(..)) => {
            let Reference::ToolCall(session, call) = reference else {
                return None;
            };
            format!(
                "SELECT t.pk FROM tool_call t JOIN session s ON s.pk=t.session_pk WHERE s.id={} AND t.call_id={}",
                bind(session.to_string()),
                bind(call.clone())
            )
        }
        Unit::Blocks | Unit::ToolCalls => {
            let messages = messages?;
            let extra = if let Reference::Block { n, .. } = reference {
                let Ok(n) = i64::try_from(*n) else {
                    return None;
                };
                format!(" AND b.position={n}")
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
        Unit::Annotations => {
            let Reference::Annotation(revision) = reference else {
                return None;
            };
            let root = bind(revision.root.clone());
            let exact = revision
                .revision
                .as_ref()
                .map(|id| format!(" AND n.id={}", bind(id.clone())))
                .unwrap_or_default();
            format!("SELECT n.rowid AS pk FROM source_annotation n WHERE n.root={root}{exact}")
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

/// Recorded conversation memberships include every contributing source,
/// rather than only the session selected for the deduplicated shown message.
pub(crate) fn message_sessions(message: &str) -> (&'static str, &str) {
    (
        "SELECT DISTINCT session.id FROM message_parse source JOIN session_stated stated ON stated.stated_id=source.session_id JOIN session ON session.pk=stated.session_pk WHERE source.message_id=?1 ORDER BY session.id",
        message,
    )
}
