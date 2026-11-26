//! The reader routes: a line given as query values (`/rows?selection=…&steps=read`)
//! or a stable object URL (`/s/ID`, `/d/ID`, `/m/ID`, `/a/REFERENCE`) is
//! answered as a typed line and rendered as HTML: a read page in the
//! reader, any other line as its text in the frame.
use super::*;
use crate::daemon::service::line::Line;
use crate::protocol::line::{LineAnswer, Rows, StepAnswer, StepOutcome};

/// The row a reader route reads through.
fn row() -> &'static crate::commands::Command {
    crate::daemon::handler::row_of::<Line>().expect("the line has a row")
}

pub(super) async fn rows(
    State(web): State<Arc<Web>>,
    Extension(lifetime): Extension<Lifetime>,
    Query(parameters): Query<Vec<(String, String)>>,
) -> HttpResponse {
    render(&web, &lifetime, &parameters).await
}

async fn render(
    web: &Arc<Web>,
    lifetime: &Lifetime,
    parameters: &[(String, String)],
) -> HttpResponse {
    let row = row();
    let request = match super::api::query_request(row, parameters) {
        Ok(request) => request,
        Err(reason) => return failed(StatusCode::BAD_REQUEST, "bad_request", &reason),
    };
    // A reader route is a GET: it runs only a line that reads.
    match crate::daemon::handler::policy(
        row,
        &request.arguments,
        polylogue_model::authorship::Channel::Web,
    ) {
        Ok(policy) if policy.takes_get() => {}
        Ok(_) => {
            return failed(
                StatusCode::BAD_REQUEST,
                "bad_request",
                "a reader page only reads; write through POST /api/v1/rows",
            );
        }
        Err(failure) => return failed_with(&failure.error),
    }
    let arguments: Rows = match serde_json::from_value(request.arguments.clone()) {
        Ok(arguments) => arguments,
        Err(error) => return failed(StatusCode::BAD_REQUEST, "bad_request", &error.to_string()),
    };
    let daemon = Arc::clone(&web.daemon);
    let caller = lifetime.caller(&Default::default());
    let mut cancel = polylogue_store::cancellation::CancelOnDrop::new(caller.stop.clone());
    let cursor = request.cursor.clone();
    let answer = tokio::task::spawn_blocking(move || {
        crate::daemon::handler::ask::<Line>(&daemon, arguments, cursor, &caller)
    })
    .await
    .unwrap_or_else(|_| {
        crate::daemon::Failure::internal("the reader's task ended without an answer").answer()
    });
    cancel.disarm();
    let declarations = crate::commands::declarations();
    let page = match &answer.outcome {
        command_surface::Outcome::Error(failure) => return failed_with(failure),
        command_surface::Outcome::Ok(LineAnswer::Steps(steps))
        | command_surface::Outcome::Empty(LineAnswer::Steps(steps))
        | command_surface::Outcome::Degraded {
            value: LineAnswer::Steps(steps),
            ..
        } if steps.steps.len() == 1 => match &steps.steps[0].outcome {
            StepOutcome::Failed { error } => return failed_with(error),
            StepOutcome::Done {
                reply: StepAnswer::Read(page),
                ..
            } => Some(page.clone()),
            _ => None,
        },
        _ => None,
    };
    if let Some(page) = page {
        return match polylogue_render::web::read(&page, declarations, &request) {
            Ok(html) => Html(html).into_response(),
            Err(failure) => failed_with(&failure),
        };
    }
    let answer = browser_response(
        answer.map(|line| serde_json::to_value(line).unwrap_or_default()),
        &web.daemon.archive.root,
    );
    let rendered = declarations
        .render_text(row.name, &answer, command_surface::TextStyle::Plain)
        .and_then(|text| declarations.cli_command(&request).map(|cli| (text, cli)));
    let (mut text, cli) = match rendered {
        Ok(rendered) => rendered,
        Err(failure) => return failed_with(&failure),
    };
    text.push_str(&declarations.render_footer(command_surface::Via::Web, &answer));
    let status = answer
        .http_status()
        .and_then(|status| StatusCode::from_u16(status).ok())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (
        status,
        Html(pages::layout(
            "Polylogue",
            "",
            &cli,
            &format!("<pre>{}</pre>", polylogue_render::html::escape(&text)),
        )),
    )
        .into_response()
}

/// The reader for one object: its reference as the selection, read with
/// the options the URL gives.
async fn object(
    web: &Arc<Web>,
    lifetime: &Lifetime,
    target: String,
    parameters: Vec<(String, String)>,
) -> HttpResponse {
    if parameters
        .iter()
        .any(|(name, _)| matches!(name.as_str(), "target" | "selection" | "steps"))
    {
        return failed(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "the object URL already supplies selection and its read step",
        );
    }
    let mut line = vec![
        ("selection".to_owned(), target),
        ("steps".to_owned(), "read".to_owned()),
    ];
    line.extend(parameters);
    render(web, lifetime, &line).await
}

pub(super) async fn session(
    State(web): State<Arc<Web>>,
    Extension(lifetime): Extension<Lifetime>,
    UrlPath(id): UrlPath<String>,
    Query(parameters): Query<Vec<(String, String)>>,
) -> HttpResponse {
    object(&web, &lifetime, format!("session:{id}"), parameters).await
}
pub(super) async fn document(
    State(web): State<Arc<Web>>,
    Extension(lifetime): Extension<Lifetime>,
    UrlPath(id): UrlPath<String>,
    Query(parameters): Query<Vec<(String, String)>>,
) -> HttpResponse {
    object(&web, &lifetime, format!("document:{id}"), parameters).await
}
pub(super) async fn message(
    State(web): State<Arc<Web>>,
    Extension(lifetime): Extension<Lifetime>,
    UrlPath(id): UrlPath<String>,
    Query(parameters): Query<Vec<(String, String)>>,
) -> HttpResponse {
    object(&web, &lifetime, format!("message:{id}"), parameters).await
}

pub(super) async fn annotation(
    State(web): State<Arc<Web>>,
    Extension(lifetime): Extension<Lifetime>,
    UrlPath(reference): UrlPath<String>,
    Query(parameters): Query<Vec<(String, String)>>,
) -> HttpResponse {
    match polylogue_model::Reference::parse(&reference) {
        Ok(reference @ polylogue_model::Reference::Annotation(_)) => {
            object(&web, &lifetime, reference.to_string(), parameters).await
        }
        Ok(_) => failed(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "an annotation URL requires its natural reference",
        ),
        Err(error) => failed(StatusCode::BAD_REQUEST, "bad_request", &error.to_string()),
    }
}

#[cfg(test)]
mod tests;
