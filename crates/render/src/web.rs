//! The typed read page in the shared frame. Selection and ready calls are
//! already resolved by the operation; this module formats no cursors.
use command_surface::{Failure, FailureKind, OperationTable, RunnableRequest};
use maud::{Markup, html};
use polylogue_model::{
    currency::{CurrencyState, IndexCurrency},
    read::{HttpCall, ReadError, ReadOptions, ReadPage, ReadyCall, View},
};
use serde_json::Value;

pub fn read(
    page: &ReadPage,
    table: &OperationTable,
    request: &RunnableRequest,
) -> Result<String, Failure> {
    let title = page
        .document
        .header
        .title
        .value()
        .map_or(page.document.header.target.as_str(), String::as_str);
    let mut frame = web_shell::Frame::for_request(
        web_shell::Product::Polylogue,
        title,
        "/search",
        table,
        request,
    )?;
    // JSON Unicode escapes preserve the invocation when displayed controls
    // become visible in HTML; the operation table still owns its spelling.
    let raw = command_surface::quote_argument(&request.arguments.to_string());
    let safe = command_surface::quote_argument(&crate::terminal::json(&request.arguments, false));
    if raw != safe {
        frame.command = std::borrow::Cow::Owned(frame.command.replacen(
            &format!("--request {raw}"),
            &format!("--request {safe}"),
            1,
        ));
    }
    frame.query = &page.snapshot.query_text;
    frame.navigation = navigation_entries();
    frame.status = currency(&page.snapshot.currency);
    let route = table
        .descriptions()
        .into_iter()
        .find(|description| description.spec.name == request.operation)
        .ok_or_else(|| Failure::new(FailureKind::BadRequest, "unknown read operation"))?
        .spec
        .http_route;
    let body = html! {
        @if page.snapshot.currency.degraded() { (web_shell::values::stale_banner("the primary read position is behind, unknown or unavailable; see the recorded position below")) }
        @for note in &page.snapshot.notes { p class="read-note" { (note) } }
        @for reason in &page.snapshot.incomplete { p class="read-incomplete" role="status" { (reason) } }
        nav class="read-views" aria-label="Read view" { @for view in View::ALL { a href=(view_url(request,&route,view)?) aria-current=[(view==page.document.header.view).then_some("page")] { (view) } } }
        (navigation(page))
        (polylogue_render::html::document(&page.document))
        (navigation(page))
        details class="read-position" { summary { "Recorded read position" } dl {
            dt { "Generation" } dd { (&page.snapshot.read_at.generation) }
            dt { "Index" } dd { (page.snapshot.read_at.index) }
            dt { "Archive" } dd { (page.snapshot.read_at.archive) }
            dt { "Ran at" } dd { (page.snapshot.ran_at) }
            dt { "Caught up" } dd { (page.snapshot.caught_up) }
        } }
    };
    Ok(polylogue_render::terminal::escape(
        &frame
            .render(&body, &|body: &Markup| body.clone())
            .into_string(),
    ))
}
fn navigation(page: &ReadPage) -> Markup {
    html! { nav class="paging" aria-label="Content pages" {
        @if let Some(call)=&page.document.footer.previous { (follow(call,"Previous",false)) }
        @if let Some(call)=&page.document.footer.next { (follow(call,"Next",true)) }
    } }
}
fn follow(call: &ReadyCall, label: &str, next: bool) -> Markup {
    match http_url(&call.http) {
        Ok(url) => {
            html! { a href=(url) rel=(if next {"next"}else{"prev"}) data-next[next] data-previous[!next] { (label) } }
        }
        Err(_) => html! { span class="view-state" { (label) ": " code { (&call.cli) } } },
    }
}
/// Encode the operation's complete HTTP form; no cursor parsing, target
/// selection or option defaults occur here. Complex fields use JSON.
pub(crate) fn http_url(call: &HttpCall) -> Result<String, Failure> {
    if call.method != "GET"
        || !call.path.starts_with('/')
        || call.path.starts_with("//")
        || call.path.contains(['?', '#'])
        || !polylogue_render::html::safe_url(&call.path)
    {
        return Err(Failure::new(
            FailureKind::BadRequest,
            "read navigation requires a local GET operation",
        ));
    }
    let fields = call.arguments.as_object().ok_or_else(|| {
        Failure::new(
            FailureKind::BadRequest,
            "read HTTP arguments must be an object",
        )
    })?;
    let mut url = call.path.clone();
    let mut first = true;
    for (name, value) in fields {
        if value.is_null() {
            continue;
        }
        let values = match value {
            Value::Array(values) => values.as_slice(),
            _ => std::slice::from_ref(value),
        };
        for value in values {
            let value = match value {
                Value::String(value) => value.clone(),
                _ => value.to_string(),
            };
            url.push(if first { '?' } else { '&' });
            first = false;
            url.push_str(&polylogue_render::html::encode_path_segment(name));
            url.push('=');
            url.push_str(&polylogue_render::html::encode_path_segment(&value));
        }
    }
    Ok(url)
}
fn currency(currency: &IndexCurrency) -> Markup {
    html! { @match currency {
        IndexCurrency::Standalone=>{span data-mode="standalone" {"Standalone archive"}},
        IndexCurrency::Primary {applied_position,owner_position,owner_position_seconds_ago,reachable,lag_positions,state,error}=>{
            span data-mode="primary" data-state=(match state {CurrencyState::Current=>"current",CurrencyState::Behind=>"behind",CurrencyState::Unknown=>"unknown",CurrencyState::Unavailable=>"unavailable"}) {
                "Primary: " (format!("{state:?}")) " · applied " (applied_position.map_or_else(||"unknown".into(),|n|n.to_string())) " · owner " (owner_position.map_or_else(||"unknown".into(),|n|n.to_string())) " · observed " (owner_position_seconds_ago.map_or_else(||"age unknown".into(),|age|format!("{age} seconds ago"))) " · reachable " (reachable.map_or_else(||"unknown".into(),|reachable|reachable.to_string())) " · lag " (lag_positions.map_or_else(||"unknown".into(),|lag|lag.to_string())) @if let Some(error)=error {" · " (error)}
            }
        }
    } }
}
#[cfg(test)]
mod tests;

/// Product destinations shared by every frame, independent of archived data.
pub fn navigation_entries() -> &'static [web_shell::Navigation<'static>] {
    const ENTRIES: [web_shell::Navigation<'static>; 3] = [
        web_shell::Navigation {
            label: "Home",
            href: "/",
            key: 'h',
            current: false,
        },
        web_shell::Navigation {
            label: "Search",
            href: "/search",
            key: 's',
            current: false,
        },
        web_shell::Navigation {
            label: "Cost and usage",
            href: "/cost",
            key: 'c',
            current: false,
        },
    ];
    &ENTRIES
}

/// A view switch is a new invocation. The model's own validation decides
/// which view options must return to their defaults; cursor state is reset.
fn view_url(request: &RunnableRequest, route: &str, view: View) -> Result<String, Failure> {
    let mut args =
        request.arguments.as_object().cloned().ok_or_else(|| {
            Failure::new(FailureKind::BadRequest, "read arguments must be an object")
        })?;
    args.remove("cursor");
    args.insert("view".into(), Value::String(view.name().into()));
    let schema = serde_json::to_value(schemars::schema_for!(ReadOptions))
        .expect("read options schema serializes");
    loop {
        let options = args
            .iter()
            .filter(|(name, value)| {
                schema["properties"].get(name.as_str()).is_some() && !value.is_null()
            })
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        let options: ReadOptions = serde_json::from_value(Value::Object(options))
            .map_err(|error| Failure::new(FailureKind::BadRequest, error.to_string()))?;
        match options.validate() {
            Ok(()) => break,
            Err(ReadError::UnsupportedOption { option, .. }) => {
                args.remove(&option);
            }
            Err(error) => return Err(Failure::new(FailureKind::BadRequest, error.to_string())),
        }
    }
    http_url(&HttpCall {
        method: "GET".into(),
        path: route.into(),
        arguments: Value::Object(args),
    })
}
