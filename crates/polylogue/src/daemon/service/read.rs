//! The daemon's services to the reading library: ranking, freshness and the
//! ready calls of continuations and attachments.
use super::Daemon;
use command_surface::{Deadline, RunnableRequest};
use polylogue_model::{freshness::ReadFreshness, read::*};
use polylogue_reading::{
    AttachmentPlace, Compile, Error, ReadRequest, ReadServices, operation::Context,
};
use polylogue_store::{Reader, WriterSender};
use std::sync::Arc;

impl ReadServices for Daemon {
    fn requests(&self) -> &polylogue_model::user::RequestMetadata {
        &self.config.requests
    }
    fn freshness(&self, reader: &Reader<'_>) -> polylogue_store::Result<ReadFreshness> {
        self.freshness(reader)
    }
    fn compiled<T>(
        &self,
        reader: &Reader<'_>,
        operation: &Arc<Context>,
        target: &str,
        run: impl FnOnce(Compile<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        crate::daemon::query::with_ranking(
            reader,
            &self.warm,
            &self.config,
            |embeddings, ranking_identity, near| {
                let ask = crate::daemon::query::Ask {
                    text: target,
                    cursor: None,
                    page_size: None,
                    why: false,
                    session: None,
                    context: Arc::clone(operation),
                    embeddings,
                    ranking_identity: ranking_identity.clone(),
                    near: Some(near),
                    fields: &[],
                };
                let options = crate::daemon::query::execution_options(
                    reader,
                    &ask,
                    polylogue_reading::seed(target),
                )
                .map_err(|failure| Error::Operation(failure.error))?;
                run(Compile {
                    options,
                    ranking_identity,
                    near,
                })
            },
        )
    }
    fn continuation(
        &self,
        request: &ReadRequest,
        cursor: Option<String>,
        deadline: Deadline,
    ) -> Result<ReadyCall, Error> {
        crate::daemon::verbs::read::LineReads {
            daemon: self,
            accepted: None,
        }
        .continuation(request, cursor, deadline)
    }
    fn attachment_call(&self, at: &AttachmentPlace) -> Result<ReadyCall, Error> {
        use crate::protocol::line::{Rows, Step};
        let reference = polylogue_model::Reference::Attachment {
            message: at.message.clone(),
            block: at.block,
            attachment: at.attachment,
        };
        let arguments = serde_json::to_value(Rows {
            selection: Some(reference.to_string()),
            steps: vec![Step::Read(Box::new(crate::protocol::steps::ReadStep {
                input: at.input.clone(),
                ..Default::default()
            }))],
            ..Rows::default()
        })
        .map_err(|error| Error::Internal(error.to_string()))?;
        let identity = canonical_json::canonical_hash("polylogue-attachment-call", &arguments)
            .map_err(|error| Error::Internal(error.to_string()))?;
        self.ready_call(&RunnableRequest {
            operation: "rows".into(),
            arguments,
            request_id: format!("read-attachment-{}", identity.to_hex()),
            deadline: Deadline::Default,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        })
        .map_err(Error::Operation)
    }
    fn writer(&self) -> Option<&WriterSender> {
        self.writes.as_ref()
    }
}
