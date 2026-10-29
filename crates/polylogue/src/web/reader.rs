//! Stable object URLs forward to the same read operation as every caller.
use super::*;

fn failure(web: &Web, kind: &str, message: &str) -> HttpResponse {
    failed(
        error_status(kind),
        kind,
        &access::scrub(message, &web.daemon.archive.root),
    )
}

fn target_url(target: &str, parameters: &[(String, String)]) -> Result<String, String> {
    if parameters.iter().any(|(name, _)| name == "target") {
        return Err("the object URL already supplies target".into());
    }
    let encode = polylogue_render::html::encode_path_segment;
    let mut url = format!("/api/v1/read?target={}", encode(target));
    for (name, value) in parameters {
        url.push('&');
        url.push_str(&encode(name));
        url.push('=');
        url.push_str(&encode(value));
    }
    Ok(url)
}
fn redirect(kind: &str, id: &str, parameters: &[(String, String)]) -> HttpResponse {
    match target_url(&format!("{kind}:{id}"), parameters) {
        Ok(url) => Redirect::temporary(&url).into_response(),
        Err(reason) => failed(StatusCode::BAD_REQUEST, "bad_request", &reason),
    }
}
pub(super) async fn session(
    UrlPath(id): UrlPath<String>,
    Query(parameters): Query<Vec<(String, String)>>,
) -> HttpResponse {
    redirect("session", &id, &parameters)
}
pub(super) async fn document(
    UrlPath(id): UrlPath<String>,
    Query(parameters): Query<Vec<(String, String)>>,
) -> HttpResponse {
    redirect("document", &id, &parameters)
}
pub(super) async fn message(
    UrlPath(id): UrlPath<String>,
    Query(parameters): Query<Vec<(String, String)>>,
) -> HttpResponse {
    redirect("message", &id, &parameters)
}

/// Resolve an exact note revision through the shared application operation.
pub(super) async fn note(
    State(web): State<Arc<Web>>,
    UrlPath(id): UrlPath<String>,
) -> HttpResponse {
    match ask(&web, Request::Note { id: id.clone() }).await {
        Response::Ok { data, .. } => page(pages::layout(
            "Note",
            "",
            &format!("polylogue note read {}", polylogue_render::shell_quote(&id)),
            &format!(
                "<h1>Note</h1><pre>{}</pre>",
                polylogue_render::html::escape(&serde_json::to_string_pretty(&data).expect("JSON"))
            ),
        )),
        Response::Error { kind, message, .. } => failure(&web, &kind, &message),
    }
}

#[cfg(test)]
mod tests;
