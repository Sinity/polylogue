//! The daemon adapts read arguments, ranking and ready calls to the reading library.
use super::{Daemon, Failure};
use command_surface::{Answer, Deadline, RunnableRequest, Suggestion};
use polylogue_model::{freshness::ReadFreshness, read::*};
use polylogue_reading::{
    AttachmentPlace, Compile, Error, ReadRequest, ReadServices, operation::Context,
};
use polylogue_store::{Reader, WriterSender};
use std::{io::Write, num::NonZeroU64, sync::Arc};

pub(crate) fn request_of(
    arguments: crate::protocol::read::Arguments,
    cursor: Option<String>,
) -> Result<ReadRequest, String> {
    let crate::protocol::read::Arguments {
        target,
        view,
        format,
        delivery,
        page_tokens,
        show,
        hide,
        output_lines,
        for_others,
        variant,
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
        variants: variant,
        at: at
            .map(|at| polylogue_model::MessageId::parse(&at))
            .transpose()
            .map_err(|error| error.to_string())?,
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
        target,
        options,
        delivery,
        format: format.unwrap_or(Format::Json).name().into(),
        budget: page_tokens
            .map(|tokens| usize::try_from(tokens.get()))
            .transpose()
            .map_err(|_| "page_tokens exceeds address space")?,
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
        cursor: _,
    } = request;
    let (show, hide) = options.show_hide_words();
    Ok(crate::protocol::read::Arguments {
        target: target.clone(),
        view: options.view,
        format: Some(Format::parse(format).ok_or_else(|| format!("unknown format {format:?}"))?),
        delivery: *delivery,
        page_tokens: budget
            .map(|value| {
                NonZeroU64::new(value as u64).ok_or_else(|| "budget must be positive".to_string())
            })
            .transpose()?,
        show,
        hide,
        output_lines: options.output_lines,
        for_others: options.for_others,
        variant: options.variants.clone(),
        with_blobs: *with_blobs,
        at: options.at.as_ref().map(ToString::to_string),
        diff: options.diff.clone(),
        around: options.around,
        latest: options.latest,
        physical: options.physical,
    })
}

pub(crate) struct Read;
impl crate::daemon::handler::Handler for Read {
    type Arguments = crate::protocol::read::Arguments;
    type Answer = crate::protocol::read::Answer;
    const PAGES: bool = true;
    fn ranking_query(
        arguments: &Self::Arguments,
        _row: &crate::commands::Command,
    ) -> Option<String> {
        Some(arguments.target.clone())
    }
    fn default_deadline(arguments: &Self::Arguments) -> bool {
        arguments.delivery != ReadDelivery::Stream
    }
    const PRINT: fn(&Self::Answer, command_surface::TextStyle) -> String = crate::cli::print::read;
    fn run(
        arguments: Self::Arguments,
        call: &crate::daemon::handler::Call<'_>,
    ) -> Result<Answer<Self::Answer>, Failure> {
        let request =
            request_of(arguments, call.cursor.map(str::to_owned)).map_err(Failure::bad_request)?;
        let history = if request.cursor.is_none() && !polylogue_reading::is_object_read(&request) {
            call.daemon
                .writes
                .as_ref()
                .map(|writer| writer.reserve_history())
                .transpose()?
        } else {
            None
        };
        let reader = call.reader()?;
        let context = call.context()?;
        let request = polylogue_reading::with_variants(&reader, call.daemon, request.clone())
            .map_err(|error| failure(error, &request))?;
        let page = polylogue_reading::page(&reader, call.daemon, &request, &context, history)
            .map_err(|error| failure(error, &request))?;
        let freshness = page.snapshot.freshness.clone();
        Ok(crate::protocol::read(page, &freshness))
    }
}
impl crate::daemon::handler::Streams for Read {
    fn streaming(arguments: &Self::Arguments) -> bool {
        arguments.delivery == ReadDelivery::Stream
    }
    fn produce(
        arguments: Self::Arguments,
        call: &crate::daemon::handler::Call<'_>,
        context: &Arc<Context>,
        output: &mut dyn Write,
    ) -> Result<(), Failure> {
        let request = request_of(arguments, None).map_err(Failure::bad_request)?;
        super::accepting(call.daemon)?;
        let reader = call.reader()?;
        let request = polylogue_reading::with_variants(&reader, call.daemon, request.clone())
            .map_err(|error| failure(error, &request))?;
        polylogue_reading::stream(&reader, call.daemon, &request, context, output)
            .map_err(|error| failure(error, &request))
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
        seed: u64,
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
                let options = crate::daemon::query::options(reader, &ask, seed)
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
        let mut request = request.clone();
        request.cursor = cursor;
        let ready = ready_request(&request, deadline)?;
        self.ready_call(&ready).map_err(Error::Operation)
    }
    fn attachment_call(&self, at: &AttachmentPlace) -> Result<ReadyCall, Error> {
        let arguments = serde_json::to_value(crate::protocol::attachment::Arguments {
            message: at.message.to_string(),
            block: at.block,
            attachment: at.attachment,
            input: at.input.clone(),
            length: crate::daemon::attachment::PAGE_BYTES as u64,
        })
        .map_err(|error| Error::Internal(error.to_string()))?;
        let identity = canonical_json::canonical_hash("polylogue-attachment-call", &arguments)
            .map_err(|error| Error::Internal(error.to_string()))?;
        self.ready_call(&RunnableRequest {
            operation: "attachment".into(),
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

fn ready_request(request: &ReadRequest, deadline: Deadline) -> Result<RunnableRequest, Error> {
    let arguments = arguments_of(request).map_err(Error::Internal)?;
    let arguments =
        serde_json::to_value(arguments).map_err(|error| Error::Internal(error.to_string()))?;
    let cursor = request.cursor.clone();
    let identity = canonical_json::canonical_hash("polylogue-read-call", &(&arguments, &cursor))
        .map_err(|error| Error::Internal(error.to_string()))?;
    Ok(RunnableRequest {
        operation: "read".into(),
        arguments,
        request_id: format!("read-page-{}", identity.to_hex()),
        deadline,
        budget_tokens: Default::default(),
        cursor,
        fresh: false,
        format: Default::default(),
    })
}

fn failure(error: Error, request: &ReadRequest) -> Failure {
    let stale = error.code() == "stale_cursor";
    let mut failure = Failure::from(error);
    if stale {
        if let Some(changed) = failure
            .error
            .message
            .strip_suffix("; run the query again without its cursor")
        {
            failure.error.message = changed.to_owned();
        }
        let mut repeated = request.clone();
        repeated.cursor = None;
        let target = polylogue_render::shell_quote(&request.target);
        let view = if request.options.view == View::Transcript {
            String::new()
        } else {
            format!(" --view {}", request.options.view)
        };
        failure.error.message.push_str(&format!(
            "; run `polylogue read {target}{view}` again to start over"
        ));
        match ready_request(&repeated, Deadline::Default) {
            Ok(request) => {
                failure.error = failure.error.with_suggestion(Suggestion::Request(request))
            }
            Err(error) => return Failure::from(error),
        }
    }
    failure
}
