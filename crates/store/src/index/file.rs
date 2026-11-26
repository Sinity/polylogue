//! Which index files exist under an archive root, and what they mean
//! (docs/architecture.md, Durability: the index file). [`IndexFiles::at`] is
//! the one derivation from disk: the archive's open and any process that
//! opens the archive without holding it use it, and every change to the
//! files goes through [`IndexFiles::step`].
//!
//! The files:
//! - `index.db`: the installed index, with its write-ahead log and the
//!   full-text files its meta names. It stays in place until a rebuild's
//!   file is renamed over it.
//! - `index-rebuild/reason.json`: why a rebuild runs. It is the first file
//!   written into the directory, so every later state of the rebuild knows
//!   its reason, and it is removed only by the install.
//! - `index-rebuild/build.db`: the rebuild being written, without syncs;
//!   `index-rebuild/sorted/` holds its sorted files.
//! - `index-rebuild/ready.db`: the finished rebuild, synced and in WAL,
//!   until it is renamed over `index.db`.
//!
//! An index belongs to one archive: its meta holds the archive's identity
//! (`archive_id`, from `archive.db`'s meta) and the record position it has
//! taken in (`archive_position`). An index of another archive, or one ahead
//! of the archive beside it, is never read: it is [`Installed::Unusable`]
//! and a rebuild replaces it.
//!
//! What a crash leaves at each point of [`IndexFiles::step`], and what the
//! next open does with it:
//!
//! | interrupted | on disk | derived | the open |
//! |---|---|---|---|
//! | `Start`, before `reason.json` is renamed into place | the directory, perhaps `reason.json.partial` | rebuilding `No` | discards the directory and starts the rebuild again for the reason the installed index gives |
//! | after `Start`, any time before `Ready` | `reason.json`, perhaps `build.db` and `sorted/` | `Writing { reason }` | starts again from the stored bytes for the same reason |
//! | `Ready`, before or after its rename | `reason.json` with `build.db` or `ready.db` | `Writing` or `Ready` | rebuilds, or installs |
//! | `Install`, after `reason.json` is removed | `ready.db` | `Ready` | installs |
//! | `Install`, after the rename | `index.db` is the rebuild; perhaps an emptied directory | rebuilding `No` | removes the directory |

use crate::archive::writer::Position;
use crate::{Result, StoreError, schema};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::path::{Path, PathBuf};

/// The directory beside `index.db` that holds a rebuild.
pub const DIR: &str = "index-rebuild";

/// The index's meta key holding the identity of the archive it was built from.
pub(crate) const ARCHIVE_ID: &str = "archive_id";

/// The index's meta key holding the record position it has taken in.
pub(crate) const ARCHIVE_POSITION: &str = "archive_position";

/// The index's meta key holding why it replaced the index before it.
const REBUILD: &str = "index_rebuild";

/// The files of an archive's index, under its root.
pub(crate) struct Paths {
    pub index: PathBuf,
    pub dir: PathBuf,
    pub reason: PathBuf,
    pub build: PathBuf,
    pub ready: PathBuf,
    /// The sorted files a rebuild prepares its readings into.
    pub sorted: PathBuf,
}

impl Paths {
    pub(crate) fn of(root: &Path) -> Self {
        let dir = root.join(DIR);
        Self {
            index: root.join("index.db"),
            reason: dir.join("reason.json"),
            build: dir.join("build.db"),
            ready: dir.join("ready.db"),
            sorted: dir.join("sorted"),
            dir,
        }
    }
}

/// An archive's identity: 32 hex digits, random, written once into
/// `archive.db`'s meta when the file is created.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(transparent)]
pub struct ArchiveId(pub String);

/// Why an index is rebuilt, or why the installed one cannot be read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RebuildReason {
    /// The archive had no index: a new archive, or one whose index was moved
    /// away. Never the reason an installed index is unusable.
    NoIndex,
    /// Its tables, indexes or triggers differ from this build's.
    SchemaDiffers,
    /// It is at another index format version.
    FormatVersion { found: u32, wanted: u32 },
    /// It carries no Polylogue format identity: written before the baseline.
    OlderFormat,
    /// It was built from another archive, or names none.
    OtherArchive,
    /// It has taken in record positions the archive beside it does not have:
    /// `archive.db` is older than the index (put back from a copy).
    AheadOfArchive { index: u64, archive: u64 },
}

impl std::fmt::Display for RebuildReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoIndex => write!(f, "the archive had no index"),
            Self::SchemaDiffers => write!(f, "its tables changed"),
            Self::FormatVersion { found, wanted } => {
                write!(f, "its format changed ({found} to {wanted})")
            }
            Self::OlderFormat => write!(f, "it was written by an older Polylogue"),
            Self::OtherArchive => write!(f, "it was built from another archive"),
            Self::AheadOfArchive { index, archive } => write!(
                f,
                "it is ahead of its archive (index at record position {index}, archive at {archive})"
            ),
        }
    }
}

impl RebuildReason {
    /// The reason an opened index must be replaced, for the refusals that
    /// mean "another schema". Any other error is not a reason to replace it.
    pub(crate) fn of(error: &StoreError) -> Option<Self> {
        match error {
            StoreError::OtherFormat {
                database: "index", ..
            } => Some(Self::SchemaDiffers),
            StoreError::IndexVersion { found, wanted } => Some(Self::FormatVersion {
                found: *found,
                wanted: *wanted,
            }),
            StoreError::OlderFormat { database: "index" } => Some(Self::OlderFormat),
            _ => None,
        }
    }
}

/// Why an index replaced the one before it: written as `reason.json` when the
/// rebuild starts and into the new index's meta, where `status` reads it
/// (`store.index_rebuild`) once it is installed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IndexRebuild {
    pub reason: RebuildReason,
    /// The schema identity of the index it replaces; none when there was
    /// none, or it could not be read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_schema: Option<String>,
    /// The schema identity of this build's index.
    pub schema: String,
}

impl IndexRebuild {
    /// A rebuild of this build's schema, replacing an index of
    /// `previous_schema`, for `reason`.
    pub(crate) fn new(reason: RebuildReason, previous_schema: Option<String>) -> Result<Self> {
        Ok(Self {
            reason,
            previous_schema,
            schema: schema::build_index_identity()?,
        })
    }

    /// The record an index holds of the rebuild that made it.
    pub(crate) fn recorded(index: &Connection) -> Result<Option<Self>> {
        if !crate::ConnExt::exists(
            index,
            "SELECT EXISTS(SELECT 1 FROM main.sqlite_schema WHERE type='table' AND name='meta')",
            [],
        )? {
            return Ok(None);
        }
        crate::meta_get(index, REBUILD)?
            .map(|value| serde_json::from_str(&value).map_err(StoreError::from))
            .transpose()
    }
}

/// The index at `index.db`, as this build reads it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Installed {
    /// No index file.
    None,
    /// An index of this archive that this build reads, at the record
    /// position it has taken in. Its read indexes may differ from this
    /// build's, which a rebuild then corrects while reads go on from it.
    Current {
        archive: ArchiveId,
        position: Position,
    },
    /// An index this build does not read: another format, another archive,
    /// or ahead of its archive. Reads answer `index_building` until a
    /// rebuild replaces it.
    Unusable { why: RebuildReason },
}

/// The rebuild in `index-rebuild/`, if any.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Rebuilding {
    No,
    /// `build.db` is being written, for `reason`.
    Writing {
        reason: RebuildReason,
    },
    /// `ready.db` is finished and waits to be renamed over `index.db`.
    Ready,
}

/// Which index files exist, and what they mean: the pair `status` shows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct IndexFiles {
    pub installed: Installed,
    pub rebuilding: Rebuilding,
}

/// A change to the index files ([`IndexFiles::step`]).
#[derive(Debug, Clone)]
pub enum Transition {
    /// Writes `reason.json`, then clears any earlier attempt's `build.db`
    /// and sorted files: a rebuild starts, or starts again.
    Start(IndexRebuild),
    /// Renames the finished `build.db` to `ready.db`.
    Ready,
    /// Renames `ready.db` over `index.db` and removes the rebuild's directory.
    Install,
    /// Removes the rebuild's directory and everything in it.
    Discard,
}

impl IndexFiles {
    /// The index files under `root`, beside the archive `archive` at record
    /// `position`.
    pub fn at(root: &Path, archive: &ArchiveId, position: Position) -> Result<Self> {
        let paths = Paths::of(root);
        Ok(Self {
            rebuilding: rebuilding(&paths)?,
            installed: installed(&paths.index, Some((archive, position)))?,
        })
    }

    /// [`Self::at`] for a process that does not hold the archive: the
    /// archive's identity and position are read from `archive.db` first.
    pub fn read(root: &Path) -> Result<Self> {
        let archive = root.join("archive.db");
        if !archive.exists() {
            let paths = Paths::of(root);
            return Ok(Self {
                rebuilding: rebuilding(&paths)?,
                installed: installed(&paths.index, None)?,
            });
        }
        let (id, position) = archive_identity(&crate::archive::open_read(&archive)?)?;
        Self::at(root, &id, position)
    }

    /// Whether reads are served from `index.db`.
    pub fn serves(&self) -> bool {
        matches!(self.installed, Installed::Current { .. })
    }

    /// Why a rebuild runs, or else why no index can be read.
    pub fn reason(&self) -> Option<RebuildReason> {
        match (&self.rebuilding, &self.installed) {
            (Rebuilding::Writing { reason }, _) => Some(reason.clone()),
            (_, Installed::None) => Some(RebuildReason::NoIndex),
            (_, Installed::Unusable { why }) => Some(why.clone()),
            (_, Installed::Current { .. }) => None,
        }
    }

    /// Moves the files under `root` to the next state. Each transition syncs
    /// what it renamed or wrote before it returns.
    pub fn step(&mut self, root: &Path, to: Transition) -> Result<()> {
        let paths = Paths::of(root);
        match (to, &self.rebuilding) {
            (Transition::Start(record), Rebuilding::No | Rebuilding::Writing { .. }) => {
                if !paths.dir.exists() {
                    fs::create_dir(&paths.dir)?;
                    crate::archive::no_cow(&paths.dir);
                    sync_directory(root)?;
                }
                let partial = paths.dir.join("reason.json.partial");
                {
                    let mut file = File::create(&partial)?;
                    serde_json::to_writer(&mut file, &record)?;
                    file.sync_all()?;
                }
                fs::rename(&partial, &paths.reason)?;
                remove_database(&paths.build)?;
                remove_tree(&paths.sorted)?;
                sync_directory(&paths.dir)?;
                self.rebuilding = Rebuilding::Writing {
                    reason: record.reason,
                };
            }
            (Transition::Ready, Rebuilding::Writing { .. }) => {
                fs::rename(&paths.build, &paths.ready)?;
                sync_directory(&paths.dir)?;
                self.rebuilding = Rebuilding::Ready;
            }
            (Transition::Install, Rebuilding::Ready) => {
                // The reason goes first: a crash after it leaves `ready.db`,
                // which is installed, never a reason without a rebuild over
                // an index that is already the rebuild.
                remove(&paths.reason)?;
                sync_directory(&paths.dir)?;
                crate::fault::check("index.install.after_reason", &root.to_string_lossy())?;
                // A log beside `index.db` belongs to the index being replaced
                // and must not be read with the new one. Nothing writes the
                // installed index while a rebuild runs, so its file holds
                // every checkpointed page whole.
                let replaced = crate::search::files::named_on_disk(&paths.index);
                for suffix in ["-wal", "-shm"] {
                    remove(&with_suffix(&paths.index, suffix))?;
                }
                fs::rename(&paths.ready, &paths.index)?;
                sync_directory(root)?;
                self.rebuilding = Rebuilding::No;
                self.installed = current(&crate::archive::open_read(&paths.index)?)?;
                crate::fault::check("index.install.after_rename", &root.to_string_lossy())?;
                // A reader of the replaced index keeps its open files.
                if let Some(replaced) = replaced {
                    crate::search::files::remove(&replaced)?;
                }
                remove_rebuild_dir(root, &paths)?;
            }
            (Transition::Discard, _) => {
                remove_rebuild_dir(root, &paths)?;
                self.rebuilding = Rebuilding::No;
            }
            (to, from) => {
                return Err(StoreError::internal(format!(
                    "the index files cannot go from {from:?} by {to:?}"
                )));
            }
        }
        Ok(())
    }
}

/// The full record of the rebuild in progress, from `reason.json`.
pub(crate) fn rebuild_record(root: &Path) -> Result<Option<IndexRebuild>> {
    match fs::read(Paths::of(root).reason) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Records in a new index why it replaced the one before it, which archive
/// it is built from, and that it has taken in no record position yet.
pub(crate) fn record_start(
    index: &Connection,
    record: &IndexRebuild,
    archive: &ArchiveId,
) -> Result<()> {
    crate::meta_set(index, REBUILD, serde_json::to_string(record)?)?;
    crate::meta_set(index, ARCHIVE_ID, &archive.0)?;
    crate::meta_set(index, ARCHIVE_POSITION, "0")
}

/// The identity of the archive `conn` has open, and its record position.
pub(crate) fn archive_identity(conn: &Connection) -> Result<(ArchiveId, Position)> {
    let id = crate::meta_get(conn, ARCHIVE_ID)?
        .ok_or_else(|| StoreError::stored_data("archive.db names no archive identity"))?;
    let position = u64::try_from(crate::write_counter(conn)?)
        .map_err(|_| StoreError::stored_data("archive.db's write counter is negative"))?;
    Ok((ArchiveId(id), Position(position)))
}

fn rebuilding(paths: &Paths) -> Result<Rebuilding> {
    if paths.ready.exists() {
        return Ok(Rebuilding::Ready);
    }
    Ok(
        match rebuild_record(paths.dir.parent().unwrap_or(Path::new(".")))? {
            Some(record) => Rebuilding::Writing {
                reason: record.reason,
            },
            None => Rebuilding::No,
        },
    )
}

/// What `index` is beside the archive `archive` at its position; `None`
/// when there is no archive, so that any index there is of another one.
fn installed(index: &Path, archive: Option<(&ArchiveId, Position)>) -> Result<Installed> {
    if !index.exists() {
        return Ok(Installed::None);
    }
    let conn = crate::archive::open_read(index)?;
    if let Err(error) = schema::check_standalone_index(&conn) {
        let Some(why) = RebuildReason::of(&error) else {
            return Err(error);
        };
        // An index that differs only in its indexes is this build's tables:
        // it is read as it is until a rebuild replaces it.
        let readable = why == RebuildReason::SchemaDiffers
            && schema::index_identity(&conn)? == schema::build_index_identity()?;
        if !readable {
            return Ok(Installed::Unusable { why });
        }
    } else if schema::version(&conn)? == 0 {
        // An empty file: an index whose creation did not commit.
        return Ok(Installed::None);
    }
    let Some((archive, position)) = archive else {
        return Ok(Installed::Unusable {
            why: RebuildReason::OtherArchive,
        });
    };
    match current(&conn)? {
        Installed::Current {
            archive: built_from,
            position: taken_in,
        } if built_from == *archive => {
            if taken_in > position {
                Ok(Installed::Unusable {
                    why: RebuildReason::AheadOfArchive {
                        index: taken_in.0,
                        archive: position.0,
                    },
                })
            } else {
                Ok(Installed::Current {
                    archive: built_from,
                    position: taken_in,
                })
            }
        }
        _ => Ok(Installed::Unusable {
            why: RebuildReason::OtherArchive,
        }),
    }
}

/// An index of this build's schema as its meta names it: the archive it was
/// built from and the position it has taken in.
fn current(index: &Connection) -> Result<Installed> {
    let Some(archive) = crate::meta_get(index, ARCHIVE_ID)? else {
        return Ok(Installed::Unusable {
            why: RebuildReason::OtherArchive,
        });
    };
    let position = crate::meta_get(index, ARCHIVE_POSITION)?
        .map(|kept| {
            kept.parse().map_err(|_| {
                StoreError::stored_data("the index's archive position is not a number")
            })
        })
        .transpose()?
        .unwrap_or(0);
    Ok(Installed::Current {
        archive: ArchiveId(archive),
        position: Position(position),
    })
}

/// Removes the rebuild's directory: its database files with the full-text
/// files they name, its sorted files and its reason.
fn remove_rebuild_dir(root: &Path, paths: &Paths) -> Result<()> {
    if !paths.dir.exists() {
        return Ok(());
    }
    remove_database(&paths.build)?;
    remove_database(&paths.ready)?;
    remove_tree(&paths.dir)?;
    sync_directory(root)
}

/// Removes a SQLite database file with its journal, log and shared-memory
/// siblings, then the full-text files it names.
pub(crate) fn remove_database(path: &Path) -> Result<()> {
    let search = crate::search::files::named_on_disk(path);
    for suffix in ["-journal", "-wal", "-shm", ""] {
        remove(&with_suffix(path, suffix))?;
    }
    if let Some(search) = search {
        crate::search::files::remove(&search)?;
    }
    Ok(())
}

/// Removes a directory and everything in it, if present.
pub(crate) fn remove_tree(dir: &Path) -> Result<()> {
    match fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

pub(crate) fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

pub(crate) fn remove(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

pub(crate) fn sync_directory(dir: &Path) -> Result<()> {
    File::open(dir)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests;
