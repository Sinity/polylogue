//! Drive consent and revocation use the collector's one credential file.
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
pub(crate) use polylogue_sources::ai_studio::drive::SCOPE;
use polylogue_sources::ai_studio::drive::{AuthorizedUser, token_path};
use reqwest::Url;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs, io,
    net::{Ipv4Addr, TcpListener},
    path::{Path, PathBuf},
    sync::mpsc,
    thread::JoinHandle,
    time::Duration,
};

const HTTP_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) struct Drive {
    pub authorization: String,
    pub token: String,
    pub revoke: String,
    pub browser: fn(&str) -> io::Result<()>,
}
impl Default for Drive {
    fn default() -> Self {
        Self {
            authorization: "https://accounts.google.com/o/oauth2/v2/auth".into(),
            token: "https://oauth2.googleapis.com/token".into(),
            revoke: "https://oauth2.googleapis.com/revoke".into(),
            browser: open_browser,
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum Installation {
    Authorized { path: PathBuf },
    AuthorizationDryRun { url: String, scope: &'static str },
    Removed { path: PathBuf },
    RemovalDryRun { path: PathBuf },
    NotAuthorized { path: PathBuf },
}

impl Drive {
    pub fn install(
        &self,
        archive: &Path,
        client_id: Option<&str>,
        remove: bool,
        dry_run: bool,
        wait: Option<Duration>,
    ) -> io::Result<Installation> {
        let path = token_path(archive);
        let agent = ureq::Agent::new_with_config(
            ureq::Agent::config_builder()
                // The deadline bounds the consent wait; each HTTP call has its own bound.
                .timeout_global(Some(
                    wait.map_or(HTTP_TIMEOUT, |duration| duration.min(HTTP_TIMEOUT)),
                ))
                .http_status_as_error(false)
                .max_redirects(0)
                .build(),
        );
        let existing = match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(io::Error::other(
                    "Drive credentials are managed by a symlink; change their source instead",
                ));
            }
            Ok(_) => Some(fs::read(&path)?),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error),
        };
        if let Some(bytes) = existing {
            let user: AuthorizedUser = serde_json::from_slice(&bytes).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Drive credentials are not an authorized-user record: {error}"),
                )
            })?;
            if user.client_id.is_empty() || user.refresh_token.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Drive credentials have no client ID or refresh token",
                ));
            }
            if !remove {
                return Ok(Installation::Authorized { path });
            }
            if dry_run {
                return Ok(Installation::RemovalDryRun { path });
            }
            let mut response = agent
                .post(&self.revoke)
                .send_form([("token", user.refresh_token.as_str())])
                .map_err(|error| io::Error::other(format!("Drive revocation failed: {error}")))?;
            let status = response.status();
            let failure = if status.is_success() {
                serde_json::Value::Null
            } else {
                response
                    .body_mut()
                    .read_json::<serde_json::Value>()
                    .unwrap_or_default()
            };
            let invalid_token = status == 400
                && failure.get("error").and_then(serde_json::Value::as_str)
                    == Some("invalid_token");
            if !status.is_success() && !invalid_token {
                return Err(io::Error::other(format!(
                    "Drive revocation returned HTTP {status}{}; credentials were kept",
                    provider_reason(&failure)
                )));
            }
            // Keep a replacement observed after the revocation request.
            if fs::read(&path)? != bytes {
                return Err(io::Error::other(
                    "Drive credentials changed during revocation; the replacement was kept",
                ));
            }
            fs::remove_file(&path)?;
            // Make the unlink itself durable, not only the file contents.
            fs::File::open(archive)?.sync_all()?;
            return Ok(Installation::Removed { path });
        }
        if remove {
            return Ok(Installation::NotAuthorized { path });
        }
        let client_id = client_id.filter(|id| !id.trim().is_empty()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Register a Google Desktop OAuth client, then run `polylogue set drive.client_id CLIENT_ID` before `polylogue install drive`")
        })?;
        let verifier = random_secret();
        let state = random_secret();
        let callback = Callback::start(state.clone())?;
        let redirect = format!("http://{}/", callback.address);
        let url = consent_url(&self.authorization, client_id, &redirect, &state, &verifier)?;
        if dry_run {
            return Ok(Installation::AuthorizationDryRun {
                url: url.into(),
                scope: SCOPE,
            });
        }
        (self.browser)(url.as_str())?;
        let code = callback.code(wait)?;
        let mut response = agent
            .post(&self.token)
            .send_form([
                ("client_id", client_id),
                ("code", code.as_str()),
                ("code_verifier", verifier.as_str()),
                ("grant_type", "authorization_code"),
                ("redirect_uri", redirect.as_str()),
            ])
            .map_err(|error| io::Error::other(format!("Drive token exchange failed: {error}")))?;
        let status = response.status();
        if !status.is_success() {
            let failure = response
                .body_mut()
                .read_json::<serde_json::Value>()
                .unwrap_or_default();
            return Err(io::Error::other(format!(
                "Drive token exchange returned HTTP {status}{}",
                provider_reason(&failure)
            )));
        }
        let value: serde_json::Value = response.body_mut().read_json().map_err(io::Error::other)?;
        if let Some(scope) = value.get("scope").and_then(serde_json::Value::as_str)
            && !scope.split_whitespace().any(|scope| scope == SCOPE)
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Drive read access was not granted",
            ));
        }
        let refresh = value
            .get("refresh_token")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Drive did not return a refresh token; authorize offline access again",
                )
            })?;
        let user = AuthorizedUser::new(client_id.into(), refresh.into());
        let bytes = serde_json::to_vec_pretty(&user).map_err(io::Error::other)?;
        fs::create_dir_all(archive)?;
        polylogue_model::atomic_write::write_atomic_if_absent(&path, &bytes)?;
        Ok(Installation::Authorized { path })
    }
}

pub(crate) fn consent_url(
    endpoint: &str,
    client_id: &str,
    redirect: &str,
    state: &str,
    verifier: &str,
) -> io::Result<Url> {
    let mut url = Url::parse(endpoint).map_err(io::Error::other)?;
    url.query_pairs_mut().extend_pairs([
        ("client_id", client_id),
        ("redirect_uri", redirect),
        ("response_type", "code"),
        ("scope", SCOPE),
        ("access_type", "offline"),
        ("prompt", "consent"),
        ("state", state),
        ("code_challenge", &pkce_challenge(verifier)),
        ("code_challenge_method", "S256"),
    ]);
    Ok(url)
}

fn random_secret() -> String {
    let mut bytes = [0; 32];
    bytes[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    bytes[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    URL_SAFE_NO_PAD.encode(bytes)
}
/// The S256 code challenge of a PKCE verifier.
pub(crate) fn pkce_challenge(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

fn open_browser(url: &str) -> io::Result<()> {
    eprintln!("Authorize read-only Google Drive access in your browser:\n{url}");
    let opened = std::process::Command::new("xdg-open")
        .arg(url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if !opened {
        eprintln!("Open the address above to continue.");
    }
    Ok(())
}

struct Callback {
    address: std::net::SocketAddr,
    reply: mpsc::Receiver<io::Result<String>>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    worker: Option<JoinHandle<()>>,
}
impl Callback {
    fn start(state: String) -> io::Result<Self> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        // The first result wins; later requests find the one slot full.
        let (send, reply) = mpsc::sync_channel(1);
        let report_failure = send.clone();
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let worker = polylogue_model::spawn_named("drive-consent", move || {
            let serve = || -> io::Result<()> {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                runtime.block_on(async {
                    let listener = tokio::net::TcpListener::from_std(listener)?;
                    let router = axum::Router::new().route("/", axum::routing::get(move |axum::extract::RawQuery(query): axum::extract::RawQuery| {
                        let state = state.clone();
                        let send = send.clone();
                        async move {
                            let result = callback_code(query.as_deref().unwrap_or(""), &state);
                            let good = result.is_ok();
                            let _ = send.try_send(result);
                            (if good { axum::http::StatusCode::OK } else { axum::http::StatusCode::BAD_REQUEST },
                                if good { "Drive authorization received. You may close this window." } else { "Drive authorization was refused. Return to your terminal." })
                        }
                    }));
                    tokio::select! {
                        result = async { axum::serve(listener, router).await } => result,
                        _ = stopped => Ok(()),
                    }
                })
            };
            if let Err(error) = serve() {
                let _ = report_failure.try_send(Err(error));
            }
        })?;
        Ok(Self {
            address,
            reply,
            stop: Some(stop),
            worker: Some(worker),
        })
    }
    fn code(&self, wait: Option<Duration>) -> io::Result<String> {
        match wait {
            Some(wait) => self.reply.recv_timeout(wait).map_err(|error| match error {
                mpsc::RecvTimeoutError::Timeout => io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Drive consent did not finish before the deadline; repeat with --deadline none",
                ),
                mpsc::RecvTimeoutError::Disconnected => {
                    io::Error::other("Drive consent listener stopped")
                }
            })?,
            None => self
                .reply
                .recv()
                .map_err(|_| io::Error::other("Drive consent listener stopped"))?,
        }
    }
}
impl Drop for Callback {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
/// The provider's error code and description from a failed token or
/// revocation reply, as a message suffix; empty when the body had neither.
fn provider_reason(body: &serde_json::Value) -> String {
    let text = |key| {
        body.get(key)
            .and_then(serde_json::Value::as_str)
            .map(|value| {
                value
                    .chars()
                    .filter(|c| !c.is_control())
                    .collect::<String>()
            })
            .filter(|value| !value.is_empty())
    };
    match (text("error"), text("error_description")) {
        (Some(error), Some(description)) => format!(": {error} ({description})"),
        (Some(error), None) => format!(": {error}"),
        (None, Some(description)) => format!(": {description}"),
        (None, None) => String::new(),
    }
}
fn callback_code(query: &str, state: &str) -> io::Result<String> {
    let url = Url::parse(&format!("http://127.0.0.1/?{query}")).map_err(io::Error::other)?;
    let pairs: Vec<_> = url.query_pairs().collect();
    let field = |key| {
        let values: Vec<_> = pairs
            .iter()
            .filter(|(name, _)| name == key)
            .map(|(_, value)| value.as_ref())
            .collect();
        if values.len() == 1 {
            Some(values[0])
        } else {
            None
        }
    };
    if field("state") != Some(state) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "Drive authorization state did not match",
        ));
    }
    if let Some((_, reason)) = pairs.iter().find(|(name, _)| name == "error") {
        let reason: String = reason
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("Drive authorization failed: {reason}"),
        ));
    }
    field("code")
        .filter(|code| !code.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Drive authorization returned no single code",
            )
        })
}

#[cfg(test)]
mod tests;
