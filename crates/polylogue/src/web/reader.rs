//! HTTP selections and navigation for the shared paged reader.
use super::*;
use crate::daemon::{PageSelection, ReadTarget};
use polylogue_render::{View, html::encode_path_segment as encode};

const DEFAULT_BUDGET: usize = 8192;

#[derive(Deserialize, Default, Clone)]
pub(crate) struct ReadQuery {
    pub view: Option<String>,
    pub cursor: Option<String>,
    pub message: Option<String>,
    pub latest: Option<bool>,
    pub with_parents: Option<bool>,
    pub budget: Option<usize>,
    pub outline: Option<bool>,
    pub outline_cursor: Option<String>,
    pub links_cursor: Option<String>,
}

impl ReadQuery {
    pub fn view(&self) -> &str {
        self.view.as_deref().unwrap_or("full")
    }
    pub fn parents(&self) -> bool {
        self.with_parents.unwrap_or(true)
    }
    pub fn budget(&self) -> usize {
        self.budget.unwrap_or(DEFAULT_BUDGET)
    }
    pub fn cli(&self, target: &str, parents: bool) -> String {
        let mut cli = format!(
            "polylogue read {}{} --view {} --budget {}",
            polylogue_render::shell_quote(target),
            if parents { " --with-parents" } else { "" },
            self.view(),
            self.budget()
        );
        if let Some(cursor) = &self.cursor {
            cli.push_str(&format!(
                " --cursor {}",
                polylogue_render::shell_quote(cursor)
            ));
        } else if self.latest.unwrap_or(false) {
            cli.push_str(" --latest");
        } else if let Some(message) = &self.message {
            cli.push_str(&format!(
                " --message {}",
                polylogue_render::shell_quote(message)
            ));
        }
        cli
    }
    pub fn url(&self, path: &str, cursor: Option<&str>) -> String {
        let mut url = format!(
            "{path}?view={}&with_parents={}&budget={}",
            encode(self.view()),
            self.parents(),
            self.budget.unwrap_or(DEFAULT_BUDGET)
        );
        for (name, value) in [
            ("cursor", cursor),
            ("message", self.message.as_deref()),
            ("outline_cursor", self.outline_cursor.as_deref()),
            ("links_cursor", self.links_cursor.as_deref()),
        ] {
            if let Some(value) = value {
                url.push_str(&format!("&{name}={}", encode(value)));
            }
        }
        if self.outline == Some(true) {
            url.push_str("&outline=true");
        }
        url
    }
}

fn failure(web: &Web, kind: &str, message: &str) -> HttpResponse {
    failed(
        error_status(kind),
        kind,
        &access::scrub(message, &web.daemon.archive.root),
    )
}

pub(super) async fn session(
    State(web): State<Arc<Web>>,
    UrlPath(id): UrlPath<String>,
    Query(mut p): Query<ReadQuery>,
) -> HttpResponse {
    let Ok(session) = polylogue_model::SessionId::parse(&id) else {
        return failed(StatusCode::BAD_REQUEST, "bad_request", "not a session ID");
    };
    if View::parse(p.view()).is_none() {
        return failed(StatusCode::BAD_REQUEST, "bad_request", "unknown view");
    }
    let message = match p
        .message
        .as_deref()
        .map(polylogue_model::MessageId::parse)
        .transpose()
    {
        Ok(message) => message,
        Err(_) => return failed(StatusCode::BAD_REQUEST, "bad_request", "not a message ID"),
    };
    let start = if p.cursor.is_some() {
        None
    } else if let Some(message) = message {
        Some(PageSelection::Message(message))
    } else {
        p.latest.unwrap_or(false).then_some(PageSelection::Latest)
    };
    let response = ask(
        &web,
        Request::Read {
            slices: true,
            target: ReadTarget::Session(session.clone()),
            view: p.view().into(),
            format: "json".into(),
            with_parents: p.parents(),
            budget: Some(p.budget.unwrap_or(DEFAULT_BUDGET)),
            cursor: p.cursor.clone(),
            selection: start,
        },
    )
    .await;
    let data = match response {
        Response::Ok { data, .. } => data,
        Response::Error { kind, message, .. } => {
            if kind == "stale_cursor" {
                let refresh = p.url(&format!("/s/{}", encode(&id)), None);
                return (StatusCode::CONFLICT,Html(pages::layout("Read changed","","polylogue web",&format!("<h1>Read changed</h1><p>{}</p><p><a href=\"{}\">Refresh from this message</a></p>",polylogue_render::html::escape(&access::scrub(&message,&web.daemon.archive.root)),polylogue_render::html::escape(&refresh))))).into_response();
            }
            return failure(&web, &kind, &message);
        }
    };
    // An anchor in each navigation URL allows a stale page to be reopened.
    if let Some(message) = data["slices"]
        .as_array()
        .and_then(|s| s.first())
        .and_then(|s| s["message"]["id"].as_str())
    {
        p.message = Some(message.into());
    }
    if let Some(cursor) = data["cursor"].as_str() {
        p.cursor = Some(cursor.into());
    }
    let outline = if p.outline == Some(true) || p.outline_cursor.is_some() {
        ask(
            &web,
            Request::Outline {
                session,
                with_parents: p.parents(),
                page_size: 50,
                cursor: p.outline_cursor.clone(),
            },
        )
        .await
    } else {
        Response::Ok {
            data: serde_json::Value::Null,
            references: Default::default(),
        }
    };
    let links = ask(
        &web,
        Request::Query {
            text: "self | links".into(),
            cursor: p.links_cursor.clone(),
            page_size: Some(50),
            why: false,
            session: Some(id.clone()),
            fields: [
                "parent",
                "child",
                "kind",
                "how",
                "evidence",
                "pending",
                "branch_point",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        },
    )
    .await;
    let optional = |response| match response {
        Response::Ok { data, .. } => data,
        Response::Error { kind, message, .. } => {
            serde_json::json!({"error":kind,"message":access::scrub(&message,&web.daemon.archive.root)})
        }
    };
    let shown_id = data["session"]["id"].as_str().unwrap_or("Session unknown");
    page(pages::reader::session(
        shown_id,
        &data,
        &p,
        &optional(outline),
        &optional(links),
    ))
}

pub(super) async fn document(
    State(web): State<Arc<Web>>,
    UrlPath(id): UrlPath<String>,
    Query(mut p): Query<ReadQuery>,
) -> HttpResponse {
    if p.message.is_some() || p.latest.unwrap_or(false) || p.with_parents == Some(true) {
        return failed(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "documents have no message starting point or conversation parents",
        );
    }
    p.with_parents = Some(false);
    let Ok(document) = polylogue_model::DocumentId::parse(&id) else {
        return failed(StatusCode::BAD_REQUEST, "bad_request", "not a document ID");
    };
    if !matches!(View::parse(p.view()), Some(View::Full | View::Dialogue)) {
        return failed(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "documents support the full and dialogue views",
        );
    }
    match ask(
        &web,
        Request::Read {
            slices: true,
            target: ReadTarget::Document(document),
            view: p.view().into(),
            format: "json".into(),
            with_parents: false,
            budget: Some(p.budget.unwrap_or(DEFAULT_BUDGET)),
            cursor: p.cursor.clone(),
            selection: None,
        },
    )
    .await
    {
        Response::Ok { data, .. } => {
            let shown_id = data["document"]["id"]
                .as_str()
                .unwrap_or("Document unknown");
            page(pages::reader::document(shown_id, &data, &p))
        }
        Response::Error { kind, message, .. } => failure(&web, &kind, &message),
    }
}

/// Resolve the opaque message reference through the public query operation.
pub(super) async fn message(
    State(web): State<Arc<Web>>,
    UrlPath(id): UrlPath<String>,
) -> HttpResponse {
    let Ok(message_id) = polylogue_model::MessageId::parse(&id) else {
        return failed(StatusCode::BAD_REQUEST, "bad_request", "not a message ID");
    };
    let text = format!(
        "messages where message:{}",
        polylogue_query::quoted(&message_id.to_string())
    );
    match ask(
        &web,
        Request::Query {
            text,
            cursor: None,
            page_size: Some(1),
            why: false,
            session: None,
            fields: vec!["id".into(), "session".into()],
        },
    )
    .await
    {
        Response::Ok { data, .. } => match (
            data["rows"][0]["session"].as_str(),
            data["rows"][0]["id"].as_str(),
        ) {
            (Some(session), Some(message)) => {
                if polylogue_model::SessionId::parse(session).is_err()
                    || polylogue_model::MessageId::parse(message).is_err()
                {
                    return failure(&web, "internal", "invalid message reference");
                }
                Redirect::to(&format!(
                    "/s/{}?message={}",
                    encode(session),
                    encode(message)
                ))
                .into_response()
            }
            _ => failed(
                StatusCode::NOT_FOUND,
                "not_found",
                "the message is not available",
            ),
        },
        Response::Error { kind, message, .. } => failure(&web, &kind, &message),
    }
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
            &format!(
                "polylogue note read {}",
                polylogue_render::shell_quote(&polylogue_render::mask(&access::scrub(
                    &id,
                    &web.daemon.archive.root
                )))
            ),
            &format!(
                "<h1>Note</h1><pre>{}</pre>",
                polylogue_render::html::escape(&serde_json::to_string_pretty(&data).expect("JSON"))
            ),
        )),
        Response::Error { kind, message, .. } => failure(&web, &kind, &message),
    }
}
