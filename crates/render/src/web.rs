//! The typed read page in the shared frame. Selection and ready calls are
//! already resolved by the operation; this module formats no cursors.
use crate::ready::reader_url;
use command_surface::{Failure, OperationDeclarations, RunnableRequest};
use maud::{Markup, html};
use polylogue_model::JsonExt;
use polylogue_model::{
    freshness::IndexFreshness,
    read::{ReadPage, ReadyCall, View},
};
use serde_json::Value;

pub fn read(
    page: &ReadPage,
    table: &OperationDeclarations,
    request: &RunnableRequest,
    view_url: impl Fn(View) -> Result<String, Failure>,
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
    let selection = request
        .arguments
        .str("selection")
        .unwrap_or(&page.snapshot.query);
    // JSON escapes keep controls visible as `\uXXXX` in the query field.
    let selection_display = crate::terminal::json(&Value::String(selection.into()), false);
    frame.query = &selection_display[1..selection_display.len() - 1];
    frame.navigation = navigation_entries();
    frame.status = freshness(&page.snapshot.freshness.sinex);
    let body = html! {
        (freshness_notice(&page.snapshot.freshness, &frame.command))
        nav class="read-views" aria-label="Read view" { @for view in View::ALL { a href=(view_url(view)?) aria-current=[(view==page.document.header.view).then_some("page")] { (view) } } }
        (navigation(page, table)?)
        (crate::html::document(&page.document, Some(table))?)
        (navigation(page, table)?)
        details class="read-position" { summary { "Recorded read position" } dl {
            dt { "Generation" } dd { (&page.snapshot.read_position.index_instance) }
            dt { "Index" } dd { (page.snapshot.read_position.index) }
            dt { "Archive" } dd { (page.snapshot.read_position.archive) }
            dt { "Ran at" } dd { (page.snapshot.ran_at) }
            dt { "Up to date" } dd { (page.snapshot.freshness.up_to_date()) }
        } }
    };
    Ok(crate::terminal::escape(
        &frame
            .render(&body, &|body: &Markup| body.clone())
            .into_string(),
    ))
}
fn navigation(page: &ReadPage, table: &OperationDeclarations) -> Result<Markup, Failure> {
    Ok(html! { nav class="paging" aria-label="Content pages" {
        @if let Some(call)=&page.document.footer.previous { (paging_link(call,"Previous",false,table)?) }
        @if let Some(call)=&page.document.footer.next { (paging_link(call,"Next",true,table)?) }
    } })
}
fn paging_link(
    call: &ReadyCall,
    label: &str,
    next: bool,
    table: &OperationDeclarations,
) -> Result<Markup, Failure> {
    Ok(match reader_url(table, &call.request) {
        Ok(url) => {
            html! { a href=(url) rel=(if next {"next"}else{"prev"}) { (label) } }
        }
        Err(_) => {
            html! { span class="view-state" { (label) ": " code { (table.cli_command(&call.request)?) } } }
        }
    })
}
fn freshness(freshness: &IndexFreshness) -> Markup {
    html! { @match freshness {
        IndexFreshness::Standalone=>{span data-mode="standalone" {"Standalone archive"}},
        IndexFreshness::Primary {scope,applied_position,sinex_position,sinex_position_seconds_ago,reachable,lag_positions,state,error}=>{
            @let state = state.as_str();
            span data-mode="primary" data-state=(state) {
                "Primary: " @if let Some(scope)=scope { (scope) " · " } (state) " · applied " (applied_position.map_or_else(||"unknown".into(),|n|n.to_string())) " · Sinex " (sinex_position.map_or_else(||"unknown".into(),|n|n.to_string())) " · observed " (sinex_position_seconds_ago.map_or_else(||"age unknown".into(),|age|format!("{age} seconds ago"))) " · reachable " (reachable.map_or_else(||"unknown".into(),|reachable|reachable.to_string())) " · lag " (lag_positions.map_or_else(||"unknown".into(),|lag|lag.to_string())) @if let Some(error)=error {" · " (error)}
            }
        }
    } }
}
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

fn freshness_notice(
    freshness: &polylogue_model::freshness::ReadFreshness,
    command: &str,
) -> Markup {
    let sinex = freshness.sinex_note();
    html! { @for note in freshness.notes() {
        // Local notes describe an index still catching up; the Sinex note
        // carries its own state.
        @let state = match sinex {
            Some((state, text)) if text == note => state.as_str(),
            _ => "behind",
        };
        (web_shell::values::view_state(state, &note, command))
    } }
}

#[cfg(test)]
mod tests;
