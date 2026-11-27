//! The origin table: one declaration per origin, naming where its files
//! live, how a file is recognized and which parser reads it.

use crate::parse::Parse;
use polylogue_model::{QuotaLimit, Route};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// Parsed into sessions and messages.
    Parsed,
    /// Collected as original bytes or committed row dumps, with parse state
    /// `no_parser_yet` until the origin has a parser.
    NoParser,
    /// Bytes kept with no records in them: the images, audio and HTML an
    /// export carries beside its conversations. Messages refer to them.
    Kept,
}

/// How an input reaches Polylogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arrival {
    /// A file the tool keeps writing at a fixed location.
    Live,
    /// An export the operator brings in with `polylogue import`.
    Import,
    /// A spool Polylogue's own hook, process watcher or browser extension
    /// writes.
    Spool,
}

/// What names an input, which decides whether a move, a recompression or a
/// second machine continues it or starts another one.
#[derive(Debug, Clone, Copy)]
pub enum InputIdentity {
    /// The machine and the path: `blake3(machine \0 path bytes)`, the first
    /// 32 hex digits. A file the tool keeps at one place.
    Location,
    /// An ID the provider writes into the file, with the machine
    /// (`codex:rollout:<session id>@<machine>`), for a tool that moves or
    /// compresses its own files. The file is not stored until that ID is
    /// readable.
    ProviderId(ProviderId),
    /// The bytes: the full BLAKE3 hex of an imported file.
    Content,
}

impl InputIdentity {
    /// The input a provider's own ID names on `machine`: no two machines
    /// write chains of one input.
    pub fn provider_named(id: &str, machine: &str) -> String {
        format!("{id}@{machine}")
    }
}

/// How a file names itself by the ID its provider writes into it.
#[derive(Debug, Clone, Copy)]
pub struct ProviderId {
    /// Whether the file at a path is one named by the ID it states; the
    /// origin's other files are named by their location.
    pub named: fn(&Path) -> bool,
    /// The input ID the file states, read from its first record (decoded);
    /// `None` until that record is complete.
    pub read: fn(&mut dyn std::io::BufRead) -> std::io::Result<Option<String>>,
}

/// Visits the local file paths written in a source record.
pub type VisitLocalReferences =
    fn(&polylogue_model::Json, &mut dyn FnMut(&str) -> std::io::Result<()>) -> std::io::Result<()>;

/// Local files an origin's records refer to (images attached by path),
/// captured when the record is read.
#[derive(Clone, Copy)]
pub struct LocalReferences {
    /// The working directory a record states for the records after it,
    /// against which a relative reference resolves.
    pub directory: fn(&polylogue_model::Json) -> Option<&str>,
    /// Each local file a record refers to, as the record writes it.
    pub each: VisitLocalReferences,
}

/// The origin of a tool's context files no parser reads.
pub const TOOL_CONTEXT: &str = "tool-context";
/// The origin of a tool's databases no parser reads.
pub const TOOL_DATABASE: &str = "tool-database";

/// How `polylogue import` recognizes an origin's files (docs/capture.md,
/// Import). One rule per origin: its `recognize`, given the whole document.
pub struct Import {
    /// A file on its own, or a member of a package that is no provider's
    /// export.
    pub alone: Alone,
    /// The provider export this origin's files arrive in, and which of its
    /// members are this origin's.
    pub member: Option<Member>,
    /// Whether a file this origin recognizes on its own makes the files
    /// beside it members of its package, kept with it.
    pub holds_package: fn(&Path) -> bool,
}

/// How a file on its own is recognized as an origin's.
pub enum Alone {
    /// Never on its own.
    No,
    /// Wherever it lies, by the origin's `recognize`: a declared structure,
    /// named among the structures a refusal lists as tried.
    Structure(&'static str),
    /// When its path keeps one of the origin's locations and `recognize`
    /// accepts it: a tool's file outside the tool's root.
    Layout,
    /// Only with `--origin NAME`: the shape has no reliable structure.
    Named,
}

/// Which members of a provider export are an origin's.
pub struct Member {
    pub package: Package,
    /// From the member's path inside the import (`export.zip!/a/b.json`),
    /// its file name, and whether this origin recognizes it on its own.
    pub is: fn(&str, &str, bool) -> bool,
}

/// The provider exports import knows by their structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Package {
    ChatGpt,
    ClaudeAi,
    Takeout,
    AiderPair,
}

impl Package {
    /// The structure's name among the structures a refusal lists as tried.
    pub fn structure(self) -> &'static str {
        match self {
            Package::ChatGpt => "ChatGPT export",
            Package::ClaudeAi => "claude.ai export",
            Package::Takeout => "Google Takeout",
            Package::AiderPair => "Aider paired histories",
        }
    }

    pub const ALL: [Package; 4] = [
        Package::ChatGpt,
        Package::ClaudeAi,
        Package::Takeout,
        Package::AiderPair,
    ];

    /// Whether a member named `name` makes its directory the root of a
    /// package, whose deeper members (assets, sidecars) join it.
    pub fn root_name(name: &str) -> bool {
        conversations_name(name)
            || name == crate::grok::EXPORT
            || name == crate::codex_cloud::EXPORT
            || name == crate::aider::HISTORY
    }

    /// The directory name that roots a Google Takeout part.
    pub const TAKEOUT: &str = "Takeout";

    /// The package the members under one root form, from each member's file
    /// name and the origin that recognizes it on its own; an error when the
    /// members declare two exports at once.
    pub fn of(
        root: &Path,
        members: &[(&str, Option<&OriginSpec>)],
    ) -> Result<Option<Package>, &'static str> {
        if root.file_name().is_some_and(|name| name == Self::TAKEOUT) {
            return Ok(Some(Package::Takeout));
        }
        let has = |name: &str| members.iter().any(|(member, _)| *member == name);
        // An export's conversations, recognized as its provider's, or named
        // beside the export's account file.
        let conversations = |package: Package, account: &str| {
            members.iter().any(|(name, origin)| {
                conversations_name(name)
                    && (origin.is_some_and(|origin| {
                        origin
                            .import
                            .member
                            .as_ref()
                            .is_some_and(|member| member.package == package)
                    }) || has(account))
            })
        };
        let chatgpt = conversations(Package::ChatGpt, "user.json");
        let claude = conversations(Package::ClaudeAi, "users.json");
        match (chatgpt, claude) {
            (true, true) => {
                Err("export package declares conflicting ChatGPT and claude.ai structures")
            }
            (false, true) => Ok(Some(Package::ClaudeAi)),
            (true, false) => Ok(Some(Package::ChatGpt)),
            (false, false) if has(crate::aider::HISTORY) && has(crate::aider::COMPANION) => {
                Ok(Some(Package::AiderPair))
            }
            (false, false) => Ok(None),
        }
    }

    /// What becomes of a member of this package that no origin's member
    /// rule takes.
    pub fn others(self, path: &str) -> Others {
        match self {
            Package::ChatGpt | Package::ClaudeAi => Others::Kept,
            // Other Takeout products are refused members.
            Package::Takeout if TAKEOUT_KEPT.iter().any(|product| path.contains(product)) => {
                Others::Kept
            }
            Package::Takeout => Others::Refused,
            Package::AiderPair => Others::Alone,
        }
    }
}

/// The Takeout products whose files are kept: Gemini Apps activity, AI
/// Studio prompts, Gems and NotebookLM.
const TAKEOUT_KEPT: &[&str] = &[
    "My Activity/Gemini Apps/",
    "Drive/Google AI Studio/",
    "Gemini/",
    "NotebookLM/",
];

/// `conversations.json`, or a shard `conversations-NNN.json`.
pub(crate) fn conversations_name(name: &str) -> bool {
    name.starts_with("conversations") && name.ends_with(".json")
}

/// What becomes of a package member no origin's member rule takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Others {
    /// Stored as a member of the export, with no parser.
    Kept,
    /// Refused: counted and not stored.
    Refused,
    /// Recognized as if it were on its own.
    Alone,
}

fn always(_: &Path) -> bool {
    true
}

impl Import {
    /// A file import never takes on its own; no export holds it.
    pub const NOT_IMPORTED: Import = Import {
        alone: Alone::No,
        member: None,
        holds_package: always,
    };

    /// A tool's file, recognized outside its root by its layout.
    pub const LAYOUT: Import = Import {
        alone: Alone::Layout,
        member: None,
        holds_package: always,
    };
}

/// The directory an origin's locations are relative to: the tool's own
/// override variable when it is set, otherwise a directory under home.
/// Exactly where the tool itself writes: Claude Code
/// honours `CLAUDE_CONFIG_DIR`, Codex `CODEX_HOME`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Root {
    pub env: Option<&'static str>,
    /// Relative to home; empty for home itself.
    pub default: &'static str,
}

impl Root {
    pub const HOME: Root = Root {
        env: None,
        default: "",
    };
    /// The root of Polylogue's own state directory ([`STATE_DIR`] under
    /// it): the spools its hook, browser host and collectors write.
    pub const STATE: Root = Root {
        env: Some("XDG_STATE_HOME"),
        default: ".local/state",
    };
}

/// Polylogue's own directory under [`Root::STATE`]. Every spool origin's
/// locations lie under it.
pub const STATE_DIR: &str = "polylogue";

/// Every root an override variable can move, once each, from the origins and
/// the tools' own homes: the variables a deployment reads to find inputs.
pub fn variable_roots() -> Vec<(&'static str, Root)> {
    let mut roots: Vec<(&'static str, Root)> = ORIGINS
        .iter()
        .map(|origin| origin.root)
        .chain(crate::locations::TOOLS.iter().map(|tool| tool.root))
        .filter_map(|root| Some((root.env?, root)))
        .collect();
    roots.sort_by_key(|(variable, _)| *variable);
    roots.dedup_by_key(|(variable, _)| *variable);
    roots
}

/// The environment of a run over a synthetic home: `HOME` and every root
/// variable set to where that root lies under `home`, so discovery reads
/// nothing outside it whatever the calling environment holds.
pub fn synthetic_home_environment(home: &Path) -> Vec<(&'static str, PathBuf)> {
    std::iter::once(("HOME", home.to_path_buf()))
        .chain(
            variable_roots()
                .into_iter()
                .map(|(variable, root)| (variable, home.join(root.default))),
        )
        .collect()
}

/// What is stored for a file found at an origin's location.
#[derive(Debug, Clone, Copy)]
pub enum Capture {
    /// The file's bytes.
    Bytes,
    /// A tool's SQLite database: a typed row dump of its rows,
    /// written by the store and parsed in place of the database, never
    /// page images. `is_input` says which found files are databases; the
    /// origin's other files are stored as bytes.
    RowDump { is_input: fn(&Path) -> bool },
}

impl Capture {
    pub fn is_row_dump(&self) -> bool {
        matches!(self, Capture::RowDump { .. })
    }
}

/// A tool's resume command line (program first) from a session's native
/// ID, or `None` when that session cannot be resumed on its own.
pub type Resume = fn(&str) -> Option<Vec<String>>;

/// One module contributing to an origin's reading.
#[derive(Debug, Clone, Copy)]
pub struct ParserModule {
    pub name: &'static str,
    pub version: u32,
}

/// Harness prefixes in the declarations affect message rows.
pub const VERSION: u32 = 1;

parser_modules! {}

macro_rules! modules {
    (@entry $name:expr, $module:path) => {
        ParserModule {
            name: $name,
            version: { use $module as parser; parser::VERSION },
        }
    };
    (@children source $module:path) => {{ use $module as parser; parser::MODULES }};
    (@children external $module:path) => { &[] };
    (@list $($kind:ident $name:expr => $module:path),* $(,)?) => {
        parser_modules!(@join
            &[$(modules!(@entry $name, $module)),*],
            $(modules!(@children $kind $module)),*
        )
    };
    (@base $($kind:ident $name:expr => $module:path),* $(,)?) => { modules!(@list
        source "parse" => crate::parse,
        source "lines" => crate::lines,
        source "records" => crate::records,
        source "fields" => crate::fields,
        source "document" => crate::document,
        source "time" => crate::time,
        external "command" => polylogue_model::command,
        $($kind $name => $module),*
    ) };
    (@no_parser $module:path) => { modules!(@base
        source concat!(stringify!($module), "::VERSION") => $module
    ) };
    ($($module:path),* $(,)?) => { modules!($($module),*;) };
    ($($module:path),*; $($external:path),* $(,)?) => { modules!(@base
        source "content_ids" => crate::content_ids,
        source "chain_names" => crate::chain_names,
        source "export" => crate::export,
        source "origin" => crate::origin,
        $(source concat!(stringify!($module), "::VERSION") => $module,)*
        $(external concat!(stringify!($external), "::VERSION") => $external),*
    ) };

}

/// One origin's whole declaration: where its files live, how they are
/// recognized, parsed, resumed and collected.
pub struct OriginSpec {
    pub modules: &'static [ParserModule],
    pub name: &'static str,
    pub support: Support,
    pub arrival: Arrival,
    /// Which route its readings of a conversation are, for the route
    /// precedence that composes a message seen through several routes
    /// (docs/sessions.md).
    pub route: Route,
    /// What names this origin's inputs.
    pub input_identity: InputIdentity,
    pub root: Root,
    /// Glob patterns relative to `root`, exactly the layout the tool
    /// writes. Empty for imports.
    pub locations: &'static [&'static str],
    pub capture: Capture,
    /// Whether a file is this origin's, from its path and first bytes. Runs
    /// after the location pattern matched and before the file is tracked; a
    /// suffix alone never decides it.
    pub recognize: fn(&Path, &crate::document::Head<'_>) -> bool,
    /// The parser for one input, which also says whether the input is read
    /// line by line or whole; `None` when the path is not one this origin
    /// parses.
    pub open: fn(&Path) -> Option<Parse>,
    /// The tool's own command line that reopens a session, from its
    /// native ID; `None` inside when that session cannot be resumed on its
    /// own.
    pub resume: Option<Resume>,
    /// Whether a bare UUID the operator pastes can name one of this
    /// origin's sessions.
    pub bare_uuid_ids: bool,
    /// When the tool runs under a subscription quota: the limits it always
    /// has, whether or not a response reported them (others come with the
    /// readings its parser stores).
    pub quota: Option<&'static [QuotaLimit]>,
    /// The web chat whose conversation pages hold this origin's sessions.
    pub web: Option<WebChat>,
    /// How files beside this origin's export name the assets its messages
    /// attach.
    pub assets: Option<ExportAssets>,
    /// What fetches this origin's files into its spool.
    pub collector: Option<Collector>,
    /// How `polylogue import` recognizes this origin's files.
    pub import: Import,
    /// The local files this origin's records refer to, captured beside them.
    pub local_references: Option<LocalReferences>,
    /// The origin whose parser reads this origin's files once decoded, when
    /// they are its files compressed (a `.zst` Codex rollout).
    pub compressed_form_of: Option<&'static str>,
}

/// Fetches an origin's files into its spool before the loop reads it.
pub struct Collector {
    /// None runs every pass; Some runs only when this setting is `on`.
    pub setting: Option<&'static str>,
    /// Run outside the loop thread when requests can wait on another process.
    pub background: bool,
    /// Advances past inputs the archive has accepted before fetching more.
    /// The callback verifies bytes, not parser completion or provider presence.
    pub advance: Option<AdvanceCollector>,
    /// Writes new inputs into the spool and returns the errors for the pass.
    /// Background collectors check the stop flag between files.
    pub sync: fn(&CollectorPaths<'_>, &std::sync::atomic::AtomicBool) -> Vec<String>,
}

/// Whether the complete input bytes have reached the archive. In primary
/// mode this means Sinex confirmed them, not only that the producer prepared them.
pub type CollectorAcceptance<'a> =
    dyn Fn(&Path, polylogue_model::BlobHash, u64) -> Result<bool, String> + 'a;

pub type AdvanceCollector = fn(&CollectorPaths<'_>, &CollectorAcceptance<'_>) -> Result<(), String>;

/// Configured directories a collector reads from and writes to.
pub struct CollectorPaths<'a> {
    pub home: &'a Path,
    pub archive: &'a Path,
    pub spool: &'a Path,
}

/// An export's asset files: the member files that hold attachment bytes,
/// named by the provider's asset ID, and the sidecars that describe them.
pub struct ExportAssets {
    /// The asset ID a member file holds, from its path in the export.
    pub member_id: fn(&str) -> Option<&str>,
    /// The kind of sidecar a member is, from its path in the export.
    pub sidecar_kind: fn(&str) -> Option<&str>,
    /// Reads a sidecar, one asset's metadata at a time.
    pub metadata: ReadSidecar,
}

/// Reads one sidecar member, handing each asset's metadata to the callback.
pub type ReadSidecar = fn(
    &str,
    &mut dyn std::io::Read,
    &mut dyn FnMut(AssetMetadata) -> Result<(), String>,
) -> Result<(), crate::export::ScanError>;

/// What a sidecar says about one asset.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssetMetadata {
    pub source_id: String,
    pub name: Option<String>,
    pub media_type: Option<String>,
    pub size: Option<u64>,
}

/// A web chat: the page that shows one conversation is
/// `https://<host>/<path>/<conversation id>`, with an optional declared prefix.
pub struct WebChat {
    /// The names the browser extension gives this provider.
    pub extension_names: &'static [&'static str],
    /// The hosts that serve conversation pages, the current one first.
    pub hosts: &'static [&'static str],
    /// The page the extension opens to work in when no tab of the provider
    /// is open.
    pub home: &'static str,
    pub path: &'static str,
    /// Optional two-segment prefixes before the conversation path.
    pub prefixes: &'static [LinkPrefix],
    /// Whether the extension can draft and submit turns on its pages.
    pub actions: bool,
    /// Whether the extension can page through the provider's conversation
    /// list to capture older conversations.
    pub import_job: bool,
}

/// A provider's optional account or project prefix on a conversation link.
pub struct LinkPrefix {
    pub path: &'static str,
    pub numeric: bool,
}

impl OriginSpec {
    /// Stable, ordered tuple of module names and versions.
    pub fn parse_version(&self) -> String {
        let modules = if self.support == Support::NoParser && self.capture.is_row_dump() {
            modules!(@no_parser crate::row_dump)
        } else {
            self.modules
        };
        modules
            .iter()
            .map(|m| format!("{}={}", m.name, m.version))
            .collect::<Vec<_>>()
            .join(",")
    }

    pub fn is_row_dump(&self, path: &Path) -> bool {
        match self.capture {
            Capture::RowDump { is_input } => is_input(path),
            Capture::Bytes => false,
        }
    }

    /// Where Polylogue's own writer puts a spool origin's files, relative
    /// to its root: the directory every location lies under, up to the
    /// first wildcard (`polylogue/hooks`). `None` for other arrivals.
    pub fn spool_dir(&self) -> Option<PathBuf> {
        if self.arrival != Arrival::Spool {
            return None;
        }
        fn fixed(l: &'static str) -> Vec<&'static str> {
            let segs: Vec<&str> = l.split('/').collect();
            // The last segment names files, never the directory.
            segs[..segs.len() - 1]
                .iter()
                .take_while(|s| !s.contains('*'))
                .copied()
                .collect()
        }
        let mut shared = fixed(self.locations.first()?);
        for l in &self.locations[1..] {
            let mine = fixed(l);
            let n = shared.iter().zip(&mine).take_while(|(a, b)| a == b).count();
            shared.truncate(n);
        }
        Some(shared.iter().collect())
    }
}

fn recognizes_nothing(_: &Path, _: &crate::document::Head<'_>) -> bool {
    false
}

fn opens_nothing(_: &Path) -> Option<Parse> {
    None
}

const CLAUDE_CODE: OriginSpec = OriginSpec {
    import: Import::LAYOUT,
    compressed_form_of: None,
    local_references: None,
    name: "claude-code",
    modules: modules!(crate::claude_code),
    support: Support::Parsed,
    arrival: Arrival::Live,
    route: Route::Live,
    input_identity: InputIdentity::Location,
    root: crate::claude_code::ROOT,
    locations: crate::claude_code::LOCATIONS,
    capture: Capture::Bytes,
    recognize: crate::claude_code::recognize,
    open: crate::claude_code::open,
    resume: Some(crate::claude_code::resume),
    bare_uuid_ids: true,
    quota: Some(crate::claude_code::QUOTA_LIMITS),
    web: None,
    assets: None,
    collector: None,
};

pub static ORIGINS: &[OriginSpec] = &[
    CLAUDE_CODE,
    OriginSpec {
        import: Import::LAYOUT,
        compressed_form_of: None,
        local_references: None,
        name: "claude-code-side",
        modules: modules!(crate::claude_code, crate::claude_code::side),
        support: Support::Parsed,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::Location,
        root: crate::claude_code::side::SIDE_ROOT,
        locations: crate::claude_code::side::SIDE_LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::claude_code::side::recognize_side,
        open: crate::claude_code::side::open_side,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::LAYOUT,
        compressed_form_of: None,
        local_references: None,
        name: "claude-code-history",
        modules: modules!(crate::claude_code, crate::claude_code::side),
        support: Support::Parsed,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::Location,
        root: crate::claude_code::ROOT,
        locations: crate::claude_code::side::HISTORY_LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::claude_code::side::recognize_history,
        open: crate::claude_code::side::open_history,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::LAYOUT,
        compressed_form_of: None,
        local_references: Some(crate::codex::LOCAL_REFERENCES),
        name: "codex",
        modules: modules!(crate::codex),
        support: Support::Parsed,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::ProviderId(crate::codex::ROLLOUT_ID),
        root: crate::codex::ROOT,
        locations: crate::codex::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::codex::recognize,
        open: crate::codex::open,
        resume: Some(crate::codex::resume),
        bare_uuid_ids: true,
        quota: Some(&[]),
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::LAYOUT,
        compressed_form_of: None,
        local_references: None,
        name: "codex-database",
        modules: modules!(crate::codex, crate::codex::database, crate::row_dump),
        support: Support::Parsed,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::Location,
        root: crate::codex::database::ROOT,
        locations: crate::codex::database::LOCATIONS,
        capture: Capture::RowDump {
            is_input: crate::codex::database::is_row_dump,
        },
        recognize: crate::codex::database::recognize,
        open: crate::codex::database::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::LAYOUT,
        compressed_form_of: None,
        local_references: None,
        name: "gemini-cli",
        modules: modules!(crate::gemini_cli),
        support: Support::Parsed,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::Location,
        root: Root::HOME,
        locations: crate::gemini_cli::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::gemini_cli::recognize,
        open: crate::gemini_cli::open,
        resume: Some(crate::gemini_cli::resume),
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::LAYOUT,
        compressed_form_of: None,
        local_references: None,
        name: "antigravity",
        modules: modules!(crate::antigravity, crate::antigravity::cli),
        support: Support::Parsed,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::Location,
        root: Root::HOME,
        locations: crate::antigravity::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::antigravity::recognize,
        open: crate::antigravity::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::LAYOUT,
        compressed_form_of: None,
        local_references: None,
        name: "antigravity-database",
        modules: modules!(crate::antigravity::database),
        support: Support::Parsed,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::Location,
        root: Root::HOME,
        locations: crate::antigravity::database::LOCATIONS,
        capture: Capture::RowDump {
            is_input: crate::antigravity::database::is_row_dump,
        },
        recognize: crate::antigravity::database::recognize,
        open: crate::antigravity::database::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::LAYOUT,
        compressed_form_of: None,
        local_references: None,
        name: "antigravity-exports",
        modules: modules!(crate::antigravity, crate::antigravity::cli),
        support: Support::Parsed,
        arrival: Arrival::Spool,
        route: Route::Capture,
        input_identity: InputIdentity::Location,
        root: Root::STATE,
        locations: &["polylogue/antigravity/*.json"],
        capture: Capture::Bytes,
        recognize: crate::antigravity::recognize,
        open: crate::antigravity::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: Some(crate::antigravity::server::COLLECTOR),
    },
    OriginSpec {
        import: Import {
            alone: Alone::Structure("Hermes session files"),
            member: None,
            holds_package: always,
        },
        compressed_form_of: None,
        local_references: None,
        name: "hermes",
        modules: modules!(crate::hermes),
        support: Support::Parsed,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::Location,
        root: crate::locations::HERMES_ROOT,
        locations: crate::hermes::LOCATIONS,
        capture: Capture::RowDump {
            is_input: crate::hermes::is_database,
        },
        recognize: crate::hermes::recognize,
        open: crate::hermes::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::LAYOUT,
        compressed_form_of: None,
        local_references: None,
        name: "grok-cli",
        modules: modules!(crate::grok_cli),
        support: Support::Parsed,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::Location,
        root: Root::HOME,
        locations: crate::grok_cli::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::grok_cli::recognize,
        open: crate::grok_cli::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::LAYOUT,
        compressed_form_of: None,
        local_references: None,
        name: "polylogue-repo-lookup",
        modules: modules!(crate::repo_lookup),
        support: Support::Parsed,
        arrival: Arrival::Spool,
        route: Route::Recorded,
        input_identity: InputIdentity::Location,
        root: Root::STATE,
        locations: crate::repo_lookup::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::repo_lookup::recognize,
        open: crate::repo_lookup::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::LAYOUT,
        compressed_form_of: None,
        local_references: None,
        name: crate::hook_spool::ORIGIN,
        modules: modules!(crate::hook_spool),
        support: Support::Parsed,
        arrival: Arrival::Spool,
        route: Route::Recorded,
        input_identity: InputIdentity::Location,
        root: Root::STATE,
        locations: crate::hook_spool::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::hook_spool::recognize,
        open: crate::hook_spool::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::LAYOUT,
        compressed_form_of: None,
        local_references: None,
        name: crate::browser_capture::ORIGIN,
        modules: modules!(
            crate::browser_capture,
            crate::chatgpt,
            crate::claude_ai,
            crate::codex_cloud,
            crate::grok
        ),
        support: Support::Parsed,
        arrival: Arrival::Spool,
        route: Route::Capture,
        input_identity: InputIdentity::Location,
        root: Root::STATE,
        locations: crate::browser_capture::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::browser_capture::recognize,
        open: crate::browser_capture::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::LAYOUT,
        compressed_form_of: None,
        local_references: None,
        name: "pi",
        modules: &[],
        support: Support::NoParser,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::Location,
        root: Root {
            env: None,
            default: ".pi/agent",
        },
        locations: &["sessions/**/*.jsonl"],
        capture: Capture::Bytes,
        recognize: |_, _| true,
        open: opens_nothing,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::LAYOUT,
        compressed_form_of: None,
        local_references: None,
        name: "llm",
        modules: &[],
        support: Support::NoParser,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::Location,
        root: Root {
            env: None,
            default: ".config/io.datasette.llm",
        },
        locations: &["logs.db"],
        capture: Capture::RowDump { is_input: |_| true },
        recognize: |_, _| true,
        open: opens_nothing,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::NOT_IMPORTED,
        compressed_form_of: None,
        local_references: None,
        name: TOOL_CONTEXT,
        modules: &[],
        support: Support::NoParser,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::Location,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: |_, _| true,
        open: opens_nothing,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::NOT_IMPORTED,
        compressed_form_of: None,
        local_references: None,
        name: TOOL_DATABASE,
        modules: &[],
        support: Support::NoParser,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::Location,
        root: Root::HOME,
        locations: &[],
        capture: Capture::RowDump { is_input: |_| true },
        recognize: |_, _| true,
        open: opens_nothing,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::NOT_IMPORTED,
        compressed_form_of: None,
        local_references: None,
        name: "codex-local-file",
        modules: &[],
        support: Support::Kept,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::Location,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: |_, _| true,
        open: opens_nothing,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::NOT_IMPORTED,
        compressed_form_of: Some("codex"),
        local_references: None,
        name: "codex-compressed",
        modules: &[],
        support: Support::Kept,
        arrival: Arrival::Live,
        route: Route::Live,
        input_identity: InputIdentity::ProviderId(crate::codex::ROLLOUT_ID),
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: |_, _| true,
        open: opens_nothing,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import {
            alone: Alone::Structure("Codex Cloud tasks"),
            member: Some(Member {
                package: Package::ChatGpt,
                is: |_, name, _| name == crate::codex_cloud::EXPORT,
            }),
            holds_package: always,
        },
        compressed_form_of: None,
        local_references: None,
        name: crate::codex_cloud::ORIGIN,
        modules: modules!(crate::codex_cloud),
        support: Support::Parsed,
        arrival: Arrival::Import,
        route: Route::Export,
        input_identity: InputIdentity::Content,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::codex_cloud::recognize,
        open: crate::codex_cloud::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import {
            alone: Alone::No,
            member: Some(Member {
                package: Package::AiderPair,
                is: |_, name, _| name == crate::aider::HISTORY || name == crate::aider::COMPANION,
            }),
            holds_package: always,
        },
        compressed_form_of: None,
        local_references: None,
        name: "aider",
        modules: modules!(crate::aider),
        support: Support::Parsed,
        arrival: Arrival::Import,
        route: Route::Export,
        input_identity: InputIdentity::Content,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::aider::recognize,
        open: crate::aider::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::NOT_IMPORTED,
        compressed_form_of: None,
        local_references: None,
        name: "export-file",
        modules: &[],
        support: Support::Kept,
        arrival: Arrival::Import,
        route: Route::Copy,
        input_identity: InputIdentity::Content,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: recognizes_nothing,
        open: opens_nothing,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import::NOT_IMPORTED,
        compressed_form_of: None,
        local_references: None,
        name: crate::polylogue_bundle::ORIGIN,
        modules: modules!(crate::polylogue_bundle; polylogue_model::bundle),
        support: Support::Parsed,
        arrival: Arrival::Import,
        route: Route::Copy,
        input_identity: InputIdentity::Content,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::polylogue_bundle::recognize,
        open: crate::polylogue_bundle::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import {
            alone: Alone::Structure("ChatGPT conversation"),
            member: Some(Member {
                package: Package::ChatGpt,
                is: |_, name, _| conversations_name(name),
            }),
            holds_package: always,
        },
        compressed_form_of: None,
        local_references: None,
        name: "chatgpt",
        modules: modules!(crate::chatgpt),
        support: Support::Parsed,
        arrival: Arrival::Import,
        route: Route::Export,
        input_identity: InputIdentity::Content,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::chatgpt::recognize,
        open: crate::chatgpt::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: Some(WebChat {
            extension_names: &["chatgpt"],
            hosts: &["chatgpt.com", "chat.openai.com"],
            home: "https://chatgpt.com/",
            path: "c",
            prefixes: &[LinkPrefix {
                path: "g",
                numeric: false,
            }],
            actions: true,
            import_job: true,
        }),
        assets: Some(crate::chatgpt::assets::RULES),
        collector: None,
    },
    OriginSpec {
        import: Import {
            alone: Alone::Structure("claude.ai conversation"),
            member: Some(Member {
                package: Package::ClaudeAi,
                is: |path, name, _| {
                    matches!(
                        name,
                        "conversations.json" | "memories.json" | "projects.json"
                    ) || path.contains("/projects/")
                },
            }),
            holds_package: |path| !crate::claude_ai::project_path(path),
        },
        compressed_form_of: None,
        local_references: None,
        name: "claude-ai",
        modules: modules!(crate::claude_ai),
        support: Support::Parsed,
        arrival: Arrival::Import,
        route: Route::Export,
        input_identity: InputIdentity::Content,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::claude_ai::recognize,
        open: crate::claude_ai::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: Some(WebChat {
            extension_names: &["claude-ai", "claude"],
            hosts: &["claude.ai"],
            home: "https://claude.ai/recents",
            path: "chat",
            prefixes: &[],
            actions: false,
            import_job: true,
        }),
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import {
            alone: Alone::Structure("Claude Design chat"),
            member: Some(Member {
                package: Package::ClaudeAi,
                is: |path, _, alone| alone && path.contains("/design_chats/"),
            }),
            holds_package: always,
        },
        compressed_form_of: None,
        local_references: None,
        name: "claude-design",
        modules: modules!(crate::claude_ai),
        support: Support::Parsed,
        arrival: Arrival::Import,
        route: Route::Export,
        input_identity: InputIdentity::Content,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::claude_ai::recognize_design,
        open: crate::claude_ai::open_design,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import {
            alone: Alone::Structure("AI Studio prompt"),
            member: Some(Member {
                package: Package::Takeout,
                is: |path, _, alone| alone && path.contains("Drive/Google AI Studio/"),
            }),
            holds_package: always,
        },
        compressed_form_of: None,
        local_references: None,
        name: crate::ai_studio::ORIGIN,
        modules: modules!(crate::ai_studio),
        support: Support::Parsed,
        arrival: Arrival::Spool,
        route: Route::Capture,
        input_identity: InputIdentity::Location,
        root: Root::STATE,
        locations: crate::ai_studio::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::ai_studio::recognize,
        open: crate::ai_studio::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: Some(WebChat {
            extension_names: &[],
            hosts: &["aistudio.google.com"],
            home: "https://aistudio.google.com/",
            path: "prompts",
            prefixes: &[LinkPrefix {
                path: "u",
                numeric: true,
            }],
            actions: false,
            import_job: false,
        }),
        assets: None,
        collector: Some(crate::ai_studio::drive::COLLECTOR),
    },
    OriginSpec {
        import: Import {
            alone: Alone::No,
            member: Some(Member {
                package: Package::Takeout,
                is: |path, name, _| {
                    path.contains("My Activity/Gemini Apps/") && name == "MyActivity.html"
                },
            }),
            holds_package: always,
        },
        compressed_form_of: None,
        local_references: None,
        name: crate::gemini_app::ORIGIN,
        modules: modules!(crate::gemini_app),
        support: Support::Parsed,
        arrival: Arrival::Import,
        route: Route::Export,
        input_identity: InputIdentity::Content,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::gemini_app::recognize,
        open: crate::gemini_app::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: Some(WebChat {
            extension_names: &["gemini"],
            hosts: &["gemini.google.com"],
            home: "https://gemini.google.com/app",
            path: "app",
            prefixes: &[LinkPrefix {
                path: "u",
                numeric: true,
            }],
            actions: false,
            import_job: false,
        }),
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import {
            alone: Alone::Structure("Grok account export"),
            member: None,
            holds_package: always,
        },
        compressed_form_of: None,
        local_references: None,
        name: "grok",
        modules: modules!(crate::grok),
        support: Support::Parsed,
        arrival: Arrival::Import,
        route: Route::Export,
        input_identity: InputIdentity::Content,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::grok::recognize,
        open: crate::grok::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: Some(WebChat {
            extension_names: &["grok"],
            hosts: &["grok.com"],
            home: "https://grok.com/",
            path: "c",
            prefixes: &[],
            actions: false,
            import_job: true,
        }),
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import {
            alone: Alone::Structure("OpenTelemetry GenAI"),
            member: None,
            holds_package: always,
        },
        compressed_form_of: None,
        local_references: None,
        name: crate::otel_genai::ORIGIN,
        modules: modules!(crate::otel_genai),
        support: Support::Parsed,
        arrival: Arrival::Import,
        route: Route::Export,
        input_identity: InputIdentity::Content,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::otel_genai::recognize,
        open: crate::otel_genai::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        import: Import {
            alone: Alone::Structure("OpenRouter room"),
            member: None,
            holds_package: always,
        },
        compressed_form_of: None,
        local_references: None,
        name: crate::openrouter::ORIGIN,
        modules: modules!(crate::openrouter),
        support: Support::Parsed,
        arrival: Arrival::Import,
        route: Route::Export,
        input_identity: InputIdentity::Content,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::openrouter::recognize,
        open: crate::openrouter::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
    // Last of the imports: its shapes are the most generic, so a specific
    // origin that recognizes the same file is tried first.
    OriginSpec {
        import: Import {
            alone: Alone::Named,
            member: None,
            holds_package: always,
        },
        compressed_form_of: None,
        local_references: None,
        name: crate::openai_chat::ORIGIN,
        modules: modules!(crate::openai_chat),
        support: Support::Parsed,
        arrival: Arrival::Import,
        route: Route::Export,
        input_identity: InputIdentity::Content,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::openai_chat::recognize,
        open: crate::openai_chat::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    },
];

pub fn by_name(name: &str) -> Option<&'static OriginSpec> {
    ORIGINS.iter().find(|o| o.name == name)
}

/// Every origin's export asset rules.
pub fn export_assets() -> impl Iterator<Item = &'static ExportAssets> {
    ORIGINS.iter().filter_map(|o| o.assets.as_ref())
}

/// The origin whose web chat the browser extension calls `name`.
pub fn by_extension_name(name: &str) -> Option<&'static OriginSpec> {
    ORIGINS.iter().find(|o| {
        o.web
            .as_ref()
            .is_some_and(|w| w.extension_names.contains(&name))
    })
}

/// The extension's first name for the web chat it calls `name` (`claude`
/// is `claude-ai`).
pub fn web_chat_name(name: &str) -> Option<&'static str> {
    by_extension_name(name)?
        .web
        .as_ref()?
        .extension_names
        .first()
        .copied()
}

/// The browser extension's provider table (`extension/src/background/providers.js`),
/// written from this registry by `cargo xtask extension-providers`. Each
/// provider the extension names gets its current host, the page it opens and
/// the path segment before a conversation's ID; the extension's permissions
/// and content scripts cover the current hosts only.
pub fn extension_providers_module() -> String {
    let mut out = String::from(
        "// Written by `cargo xtask extension-providers` from Polylogue's origin\n\
         // registry (crates/sources/src/origin.rs); edit the registry, not this file.\n\
         //\n\
         // Each provider the extension captures, by the name the extension gives it:\n\
         // the hosts that serve its conversation pages, the page opened to work in\n\
         // when none is open, and the path segment before a conversation's ID.\n\
         export const PROVIDERS = {\n",
    );
    for origin in ORIGINS {
        let Some(web) = &origin.web else { continue };
        let Some(name) = web.extension_names.first() else {
            continue;
        };
        let json = |value: &str| serde_json::to_string(value).expect("a string serializes");
        out.push_str(&format!(
            "  {}: {{ hosts: [{}], home: {}, path: {} }},\n",
            json(name),
            json(web.hosts[0]),
            json(web.home),
            json(web.path),
        ));
    }
    out.push_str("};\n");
    out
}

/// The session a provider conversation URL names, whether or not the
/// archive holds it.
pub fn conversation_session(url: &str) -> Option<polylogue_model::SessionId> {
    let parsed = conversation_catalog().read_url(url.trim()).ok()?;
    let references::Target::Object(object) = parsed.target() else {
        return None;
    };
    (object.product() == references::Product::Polylogue && object.kind() == "session")
        .then(|| polylogue_model::SessionId::parse(object.id()).ok())
        .flatten()
}

/// Conversation URL declarations shared by query and read preparation.
/// Captured path segments are provider IDs, keeping their original spelling.
pub fn conversation_catalog() -> std::sync::Arc<references::Catalog> {
    use references::{Catalog, Kind, Product, ProductCatalog, UrlPattern};
    use std::sync::{Arc, LazyLock};
    static CATALOG: LazyLock<Arc<Catalog>> = LazyLock::new(|| {
        let mut patterns = Vec::new();
        for origin in ORIGINS {
            let Some(web) = &origin.web else { continue };
            let mut paths = vec![format!("/{}/{{id}}", web.path)];
            paths.extend(
                web.prefixes
                    .iter()
                    .map(|prefix| format!("/{}/{{account}}/{}/{{id}}", prefix.path, web.path)),
            );
            for host in web.hosts {
                for host in [host.to_string(), format!("www.{host}")] {
                    for path in &paths {
                        for path in [path.to_string(), format!("{path}/")] {
                            patterns.push(
                                UrlPattern::new(
                                    &host,
                                    &path,
                                    &[],
                                    &format!("session:{}:{{id}}", origin.name),
                                )
                                .and_then(|pattern| pattern.with_literal_capture("id"))
                                .expect("origin conversation URL declarations are valid"),
                            );
                        }
                    }
                }
            }
        }
        Arc::new(
            Catalog::new(
                Product::Polylogue,
                vec![
                    ProductCatalog::new(Product::Polylogue, vec![Kind::new("session", &[])])
                        .with_patterns(patterns),
                ],
            )
            .expect("conversation catalog declares valid product kinds"),
        )
    });
    Arc::clone(&CATALOG)
}

#[cfg(test)]
mod tests;
