//! The daemon adapts read arguments, ranking and ready calls to the reading library.
use super::{Daemon, Failure};
use command_surface::{Answer, Deadline, RunnableRequest, Suggestion};
use polylogue_model::{freshness::ReadFreshness, read::*};
use polylogue_reading::{
    AttachmentPlace, Compile, Error, ReadRequest, ReadServices, operation::Context,
};
use polylogue_store::{Reader, WriterSender};
use std::{io::Write, sync::Arc};

pub(crate) fn request_of(
    arguments: crate::protocol::read::Arguments,
    cursor: Option<String>,
) -> Result<ReadRequest, String> {
    let crate::protocol::read::Arguments {
        target,
        from,
        displayed: _,
        view,
        format,
        delivery,
        page_tokens,
        show,
        hide,
        output_lines,
        for_others,
        alternate,
        with_blobs,
        at,
        diff,
        around,
        latest,
        physical,
    } = arguments;
    let mut options = ReadOptions {
        view,
        output_lines,
        for_others,
        alternates: alternate,
        at: at.map(ReadAt::try_from).transpose()?,
        diff,
        around,
        latest,
        physical,
        ..ReadOptions::default()
    };
    options
        .apply_show_hide(&show, &hide)
        .map_err(|error| error.to_string())?;
    options.validate().map_err(|error| error.to_string())?;
    Ok(ReadRequest {
        start: match from.as_deref() {
            None => polylogue_reading::ReadStart::Beginning,
            Some("seen") => polylogue_reading::ReadStart::Seen,
            Some(_) => return Err("only --from seen is supported".into()),
        },
        references: None,
        target,
        options,
        delivery,
        format: format.unwrap_or(Format::Json).name().into(),
        budget: page_tokens,
        with_blobs,
        cursor,
    })
}

pub(crate) fn arguments_of(
    request: &ReadRequest,
) -> Result<crate::protocol::read::Arguments, String> {
    let ReadRequest {
        target,
        options,
        delivery,
        format,
        budget,
        with_blobs,
        ..
    } = request;
    let (show, hide) = options.show_hide_words();
    Ok(crate::protocol::read::Arguments {
        target: target.clone(),
        from: request.start.uses_seen().then(|| "seen".to_owned()),
        displayed: false,
        view: options.view,
        format: Some(Format::parse(format).ok_or_else(|| format!("unknown format {format:?}"))?),
        delivery: *delivery,
        page_tokens: *budget,
        show,
        hide,
        output_lines: options.output_lines,
        for_others: options.for_others,
        alternate: options.alternates.clone(),
        with_blobs: *with_blobs,
        at: options.at.as_ref().map(ToString::to_string),
        diff: options.diff.clone(),
        around: options.around,
        latest: options.latest,
        physical: options.physical,
    })
}

/// One page of a read, with ready calls bound to its accepted request.
pub(crate) fn page(
    arguments: crate::protocol::read::Arguments,
    call: &crate::daemon::handler::Call<'_>,
    references: Option<&[polylogue_model::Reference]>,
) -> Result<Answer<ReadPage>, Failure> {
    let displayed = match call.caller.channel() {
        polylogue_model::authorship::Channel::Mcp => true,
        channel => crate::operation::command_line(channel) && arguments.displayed,
    };
    let mut request =
        request_of(arguments, call.cursor.map(str::to_owned)).map_err(Failure::bad_request)?;
    request.references = references.map(<[_]>::to_vec);
    let history = if request.cursor.is_none() && !polylogue_reading::is_object_read(&request) {
        call.daemon
            .writes
            .as_ref()
            .map(|writer| writer.archive().reserve_history())
            .transpose()?
    } else {
        None
    };
    let reader = call.reader()?;
    let context = call.context()?;
    polylogue_reading::resolve_start(&reader, &context, &mut request)
        .map_err(|error| failure(error, &request))?;
    let services = &crate::daemon::verbs::read::LineReads {
        daemon: call.daemon,
        accepted: context.accepted_request(),
    };
    let request = polylogue_reading::with_alternates(&reader, services, request.clone())
        .map_err(|error| failure(error, &request))?;
    let (page, marks) = polylogue_reading::page(&reader, services, &request, &context, history)
        .map_err(|error| failure(error, &request))?;
    if displayed {
        crate::daemon::seen::record(call.daemon, call.invocation, marks);
    }
    let freshness = page.snapshot.freshness.clone();
    Ok(crate::protocol::read(page, &freshness))
}
/// A read as a stream of events; `services` make its ready calls.
pub(crate) fn stream(
    arguments: crate::protocol::read::Arguments,
    call: &crate::daemon::handler::Call<'_>,
    context: &Arc<Context>,
    output: &mut dyn Write,
    services: &impl ReadServices,
    references: Option<&[polylogue_model::Reference]>,
) -> Result<Answer<ReadComplete>, Failure> {
    let mut request =
        request_of(arguments, call.cursor.map(str::to_owned)).map_err(Failure::bad_request)?;
    request.references = references.map(<[_]>::to_vec);
    super::accepting(call.daemon)?;
    let reader = call.reader()?;
    polylogue_reading::resolve_start(&reader, context, &mut request)
        .map_err(|error| failure(error, &request))?;
    let request = polylogue_reading::with_alternates(&reader, services, request.clone())
        .map_err(|error| failure(error, &request))?;
    let freshness = services.freshness(&reader)?;
    let complete = polylogue_reading::stream(&reader, services, &request, context, output)
        .map_err(|error| failure(error, &request))?;
    if complete.counts == polylogue_model::bundle::Counts::default()
        && complete.footer.coverage.is_empty()
        && freshness.up_to_date()
    {
        Ok(Answer::empty(complete))
    } else {
        Ok(crate::protocol::read(complete, &freshness))
    }
}

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

fn failure(error: Error, request: &ReadRequest) -> Failure {
    let stale = error.code() == polylogue_model::Code::StaleCursor;
    let mut failure = Failure::from(error);
    if stale {
        failure.error.hint = None;
        if let Some(changed) = failure
            .error
            .message
            .strip_suffix("; run the query again without its cursor")
        {
            failure.error.message = changed.to_owned();
        }
        let mut repeated = request.clone();
        repeated.cursor = None;
        if let polylogue_reading::ReadStart::Bound(bound) = &repeated.start
            && bound.from_seen
        {
            repeated.target = if bound.logical {
                format!("root:{}", bound.session)
            } else {
                polylogue_model::Reference::Session(bound.session.clone()).to_string()
            };
        }
        // "Start over" means the beginning, as the sentence below says, not
        // the seen mark the stale cursor read from.
        repeated.start = polylogue_reading::ReadStart::Beginning;
        let target = polylogue_render::shell_quote(&repeated.target);
        let view = if request.options.view == View::Transcript {
            String::new()
        } else {
            format!(" --view {}", request.options.view)
        };
        failure.error.message.push_str(&format!(
            "; run `polylogue {target} read{view}` again to start over"
        ));
        let arguments = arguments_of(&repeated)
            .map_err(Error::Internal)
            .and_then(|arguments| {
                let rows = crate::protocol::line::Rows {
                    selection: Some(repeated.target.clone()),
                    steps: vec![crate::protocol::line::Step::Read(Box::new(
                        crate::daemon::verbs::read::step_of(arguments, None)
                            .map_err(|failure| Error::Operation(failure.error))?,
                    ))],
                    ..Default::default()
                };
                serde_json::to_value(rows).map_err(|error| Error::Internal(error.to_string()))
            });
        match arguments.map(|arguments| {
            crate::client::request(
                "rows",
                arguments,
                polylogue_store::user_writes::new_request_id(),
            )
        }) {
            Ok(request) => {
                failure.error = failure.error.with_suggestion(Suggestion::Request(request))
            }
            Err(error) => failure
                .error
                .message
                .push_str(&format!("; fresh request options: {error}")),
        }
    }
    failure
}
