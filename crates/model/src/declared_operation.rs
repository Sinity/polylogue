//! A declared legacy call identity. This is not a modern native turn-scoped key
//! or a correspondence proof. The source mapper must establish the original
//! Session scope and supplied invocation ID before invoking this constructor.
use crate::{IdError, SessionId};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeclaredOperationId {
    reference: String,
    session: SessionId,
}
impl DeclaredOperationId {
    pub fn new(source_session: &SessionId, supplied_id: &str) -> Result<Self, IdError> {
        if supplied_id.is_empty() {
            return Err(IdError::Empty);
        }
        let mut hash = blake3::Hasher::new();
        hash.update(b"polylogue-declared-operation-v1\0");
        for component in [source_session.as_str(), supplied_id] {
            hash.update(&(component.len() as u64).to_le_bytes());
            hash.update(component.as_bytes());
        }
        Ok(Self {
            reference: format!("declared-v1:{}", hash.finalize().to_hex()),
            session: source_session.clone(),
        })
    }
    pub fn as_str(&self) -> &str {
        &self.reference
    }
    pub fn session(&self) -> &SessionId {
        &self.session
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MessageId;
    #[test]
    fn declared_call_scope_is_independent_of_message_and_import_order() {
        let session = SessionId::from_stored("claude-code:source:with:colons").unwrap();
        let a = MessageId::new(session.clone(), "a").unwrap();
        let b = MessageId::new(session.clone(), "b").unwrap();
        let first = DeclaredOperationId::new(a.session(), "tool:call").unwrap();
        let repeated = DeclaredOperationId::new(b.session(), "tool:call").unwrap();
        assert_eq!(first, repeated);
        assert_eq!(
            [first.clone(), repeated.clone()]
                .into_iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            1
        );
        assert_eq!(
            [repeated, first.clone()]
                .into_iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            1
        );
        assert_eq!(first.session(), &session);
        assert_ne!(
            first,
            DeclaredOperationId::new(
                &SessionId::from_stored("claude-code:another").unwrap(),
                "tool:call"
            )
            .unwrap()
        );
        assert_eq!(DeclaredOperationId::new(&session, ""), Err(IdError::Empty));
    }
    #[test]
    fn framing_has_an_independent_byte_oracle_and_no_delimiter_ambiguity() {
        let session = SessionId::from_stored("s:a").unwrap();
        let result = DeclaredOperationId::new(&session, "b:c").unwrap();
        let mut bytes = b"polylogue-declared-operation-v1\0".to_vec();
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(b"s:a");
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(b"b:c");
        assert_eq!(
            result.as_str(),
            format!("declared-v1:{}", blake3::hash(&bytes).to_hex())
        );
        let other =
            DeclaredOperationId::new(&SessionId::from_stored("s:a:b").unwrap(), "c").unwrap();
        assert_ne!(result, other);
        assert_ne!(
            DeclaredOperationId::new(&session, "é").unwrap(),
            DeclaredOperationId::new(&session, "e\u{301}").unwrap()
        );
    }
}
