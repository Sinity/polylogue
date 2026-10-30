//! Stable object URLs forward to the same read operation as every caller.
use super::*;

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

pub(super) async fn assertion(
    UrlPath(reference): UrlPath<String>,
    Query(parameters): Query<Vec<(String, String)>>,
) -> HttpResponse {
    match polylogue_model::Reference::parse(&reference) {
        Ok(reference @ polylogue_model::Reference::Assertion(_)) => {
            match target_url(&reference.to_string(), &parameters) {
                Ok(url) => Redirect::temporary(&url).into_response(),
                Err(reason) => failed(StatusCode::BAD_REQUEST, "bad_request", &reason),
            }
        }
        _ => failed(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "an assertion URL requires its natural reference",
        ),
    }
}

#[cfg(test)]
mod tests;
