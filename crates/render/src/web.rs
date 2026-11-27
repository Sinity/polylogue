//! The typed read page in the shared frame. Selection and ready calls are
//! already resolved by the operation; this module formats no cursors.
use crate::ready::reader_url;
use command_surface::{Failure, FailureKind, OperationDeclarations, RunnableRequest};
use maud::{Markup, html};
use polylogue_model::JsonExt;
use polylogue_model::{
    freshness::IndexFreshness,
    read::{ReadError, ReadOptions, ReadPage, ReadyCall, View},
};
use serde_json::Value;

pub fn read(
    page: &ReadPage,
    table: &OperationDeclarations,
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
    let selection = request
        .arguments
        .str("target")
        .unwrap_or(&page.snapshot.query);
    // JSON escapes keep controls visible as `\uXXXX` in the query field.
    let selection_display = crate::terminal::json(&Value::String(selection.into()), false);
    frame.query = &selection_display[1..selection_display.len() - 1];
    frame.navigation = navigation_entries();
    frame.status = freshness(&page.snapshot.freshness.sinex);
    let body = html! {
        (freshness_notice(&page.snapshot.freshness, &frame.command))
        nav class="read-views" aria-label="Read view" { @for view in View::ALL { a href=(view_url(table,request,view)?) aria-current=[(view==page.document.header.view).then_some("page")] { (view) } } }
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

/// A view switch is a new invocation. The model's own validation decides
/// which view options must return to their defaults; cursor state is reset.
fn view_url(
    table: &OperationDeclarations,
    request: &RunnableRequest,
    view: View,
) -> Result<String, Failure> {
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
                // A refused item kind leaves the show and hide words that
                // name it; any other refused option leaves the arguments.
                let kind = if option == "context_meter" {
                    "context"
                } else {
                    option.as_str()
                };
                let mut removed = args.remove(&option).is_some();
                for words in ["show", "hide"] {
                    if let Some(Value::Array(list)) = args.get_mut(words) {
                        let before = list.len();
                        list.retain(|word| {
                            word.as_str()
                                .is_none_or(|word| word.split(':').next() != Some(kind))
                        });
                        removed |= list.len() != before;
                    }
                }
                if !removed {
                    return Err(Failure::new(
                        FailureKind::BadRequest,
                        format!("view {view} does not accept {option}"),
                    ));
                }
            }
            Err(error) => return Err(Failure::new(FailureKind::BadRequest, error.to_string())),
        }
    }
    reader_url(
        table,
        &RunnableRequest {
            arguments: Value::Object(args),
            cursor: None,
            ..request.clone()
        },
    )
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
        (web_shell::values::view_state(state, note, command))
    } }
}

#[cfg(test)]
mod tests;
