//! Live state is computed by the daemon, including changes driven by time.
use super::*;

pub(super) const LIVE: &str = "sessions where -live:ended | sort by live_rank, last_event_at desc";
pub(super) const RECENT: &str = "sessions | sort by last_active desc";

#[derive(Default, Deserialize)]
pub(super) struct List {
    #[serde(default)]
    recent: bool,
    cursor: Option<String>,
}

pub(super) async fn home(State(web): State<Arc<Web>>, Query(p): Query<List>) -> HttpResponse {
    let text = if p.recent { RECENT } else { LIVE };
    let mut request = search::query(text.into(), p.cursor.clone());
    if let Request::Query { fields, .. } = &mut request {
        *fields = [
            "id",
            "live",
            "title",
            "repo",
            "cwd",
            "harness",
            "last_event_at",
            "last_active",
            "waiting_on",
            "last_event",
            "live_until",
        ]
        .map(str::to_owned)
        .to_vec();
    }
    match ask(&web, request).await {
        Response::Ok { data } => page(pages::live::home(
            &scrubbed(&web, data),
            p.recent,
            p.cursor.as_deref(),
        )),
        Response::Error { kind, message, .. } => failed(
            StatusCode::INTERNAL_SERVER_ERROR,
            &kind,
            &access::scrub(&message, &web.daemon.archive.root),
        ),
    }
}

pub(super) async fn script() -> HttpResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        include_str!("assets/sessions.js"),
    )
        .into_response()
}

pub(super) async fn style() -> HttpResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("assets/sessions.css"),
    )
        .into_response()
}
