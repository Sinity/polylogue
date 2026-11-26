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
    let mut answer = browser_response(
        answer.map(|line| serde_json::to_value(line).unwrap_or_default()),
        &web.daemon.archive.root,
    );
    // A single failed step is the failure of this view, with the line's details.
    if let Some(value) = answer.value() {
        let line: LineAnswer =
            serde_json::from_value(value.clone()).expect("line reply deserializes");
        if let LineAnswer::Steps(steps) = line
            && steps.steps.len() == 1
            && let StepOutcome::Failed { error } = &steps.steps[0].outcome
        {
            answer.outcome = command_surface::Outcome::Error(error.clone());
        }
    }
    let cli = match declarations.cli_command(&request) {
        Ok(cli) => cli,
        Err(failure) => return failed_with(&failure),
    };
    let rendered = answer.value().map(|value| {
        let line: LineAnswer =
            serde_json::from_value(value.clone()).expect("line reply deserializes");
        if let LineAnswer::Steps(steps) = &line
            && steps.steps.len() == 1
            && let StepOutcome::Done {
                reply: StepAnswer::Read(page),
                ..
            } = &steps.steps[0].outcome
        {
            return polylogue_render::web::read(page, declarations, &request);
        }
        declarations
            .render_text(row.name, &answer, command_surface::TextStyle::Plain)
            .map(|text| {
                pages::layout(
                    "Polylogue",
                    "",
                    &cli,
                    &format!("<pre>{}</pre>", polylogue_render::html::escape(&text)),
                )
            })
    });
    match rendered {
        Some(Err(failure)) => failed_with(&failure),
        rendered => answer_page(&[(&answer, &cli)], || rendered.unwrap().unwrap()),
    }
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
