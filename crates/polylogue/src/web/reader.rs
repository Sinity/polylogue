//! The reader routes: a line given as query values (`/rows?selection=…&steps=read`)
//! or a stable object URL (`/s/ID`, `/d/ID`, `/m/ID`, `/a/REFERENCE`) is
//! answered as a typed line and rendered as HTML: a read page in the
//! reader, any other line as its text in the frame.
use super::*;
use crate::daemon::service::line::Line;
use crate::protocol::line::{LineAnswer, Rows, Step, StepAnswer, StepOutcome};
use command_surface::{Failure, FailureKind, OperationDeclarations, RunnableRequest};
use polylogue_model::read::View;
use serde::Deserialize;

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
    let request = match super::api::get_request(row, parameters) {
        Ok(request) => request,
        Err(failure) => return failed_with(&failure),
    };
    let caller = lifetime.caller(&Default::default());
    let answer = match super::api::execute(web, request.clone(), caller).await {
        super::api::HttpReply::Answer(answer) => answer.0,
        super::api::HttpReply::Stream(response) => return response,
    };
    let declarations = crate::commands::declarations();
    let mut answer = browser_response(answer, &web.daemon.archive.root);
    let line = match answer.value().map(LineAnswer::deserialize).transpose() {
        Ok(line) => line,
        Err(error) => return failed_with(&Failure::new(FailureKind::Internal, error.to_string())),
    };
    // A single failed step is the failure of this view, with the line's details.
    if let Some(LineAnswer::Steps(steps)) = &line
        && steps.steps.len() == 1
        && let StepOutcome::Failed { error } = &steps.steps[0].outcome
    {
        answer.outcome = command_surface::Outcome::Error(error.clone());
    }
    let cli = match declarations.cli_command(&request) {
        Ok(cli) => cli,
        Err(failure) => return failed_with(&failure),
    };
    if let Some(LineAnswer::Steps(steps)) = &line
        && steps.steps.len() == 1
        && let StepOutcome::Done {
            reply: StepAnswer::Read(page),
            ..
        } = &steps.steps[0].outcome
    {
        let rows: Rows = match Rows::deserialize(&request.arguments) {
            Ok(rows) => rows,
            Err(error) => {
                return failed_with(&Failure::new(FailureKind::Internal, error.to_string()));
            }
        };
        return match polylogue_render::web::read(page, declarations, &request, |view| {
            view_url(declarations, &request, rows.clone(), view)
        }) {
            Ok(html) => answer_page(&[(&answer, &cli)], || html),
            Err(failure) => {
                answer.outcome = command_surface::Outcome::Error(failure);
                answer_page(&[(&answer, &cli)], String::new)
            }
        };
    }
    let rendered = declarations.render_text(row.name, &answer, command_surface::TextStyle::Plain);
    match rendered {
        Ok(text) => answer_page(&[(&answer, &cli)], || {
            pages::layout(
                "Polylogue",
                "",
                &cli,
                &format!("<pre>{}</pre>", polylogue_render::html::escape(&text)),
            )
        }),
        Err(failure) => {
            answer.outcome = command_surface::Outcome::Error(failure);
            answer_page(&[(&answer, &cli)], String::new)
        }
    }
}

/// Change only the read step, with a fresh identity and no continuation.
fn view_url(
    declarations: &OperationDeclarations,
    request: &RunnableRequest,
    mut rows: Rows,
    view: View,
) -> Result<String, Failure> {
    let [Step::Read(read)] = rows.steps.as_mut_slice() else {
        return Err(Failure::new(
            FailureKind::Internal,
            "a read page needs a read step",
        ));
    };
    read.options
        .set_view(view)
        .map_err(|error| Failure::new(FailureKind::BadRequest, error.to_string()))?;
    read.cursor = None;
    let arguments = serde_json::to_value(rows)
        .map_err(|error| Failure::new(FailureKind::Internal, error.to_string()))?;
    polylogue_render::ready::reader_url(
        declarations,
        &RunnableRequest {
            arguments,
            cursor: None,
            request_id: polylogue_store::user_writes::new_request_id(),
            ..request.clone()
        },
    )
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
