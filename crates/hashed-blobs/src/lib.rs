//! `hashed-blobs`: bytes by BLAKE3 hash, plain or as a delta, sealed or not.
//!
//! The shared content-store crate of Polylogue and Sinex (joint decisions #11, #12, #14, #19, #23,
//! #24). It holds no policy: inline-versus-file size, when to use a delta, snapshots, page maps,
//! joining, excision rewrites, privacy classes and key management all belong to the caller.
//!
//! Files live at `<root>/blobs/<first byte hex>/<name hex>.zst`, written to `<root>/blobs/tmp/`,
//! synced, renamed into place, with every directory created on the way synced. Three layouts:
//!
//! - **plain**: independent zstd frames (each with its content checksum) followed by a seek table
//!   in the zstd seekable format, so stock `zstd -d` decompresses the file and a range read
//!   decompresses only the frames it needs. Named by the BLAKE3 of the content.
//! - **delta**: a skippable header frame naming the base, then one zstd frame made as `zstd --patch-from`
//!   makes one (the base as prefix, a window over base and content, long-distance matching). The base is always plain, so a read decompresses at most two
//!   blobs. Not seekable.
//! - **sealed**: each compressed frame encrypted with ChaCha20-Poly1305 under a fresh per-encoding frame key
//!   (#23), then a footer with the key reference, the frame table and a BLAKE3 of the ciphertext
//!   so `verify` can check integrity without the key (#19b). Named by a BLAKE3 keyed hash of the
//!   content under a naming key derived from the data key (#24). A sealed delta is one sealed
//!   frame whose base is a sealed blob under the same key.
//!
//! Integrity never relies on the filesystem (Polylogue's archive is No_COW on btrfs, so btrfs
//! checksums are off): full reads check the name, range reads check frame checksums or AEAD tags.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use chacha20poly1305::aead::{rand_core::RngCore, Aead, OsRng, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, KeyInit, Nonce};

pub const DEFAULT_FRAME: usize = 4 << 20;

const SEEK_SKIP_MAGIC: u32 = 0x184D_2A5E; // zstd seekable format: the seek table's skippable frame
const SEEKABLE_MAGIC: u32 = 0x8F92_EAB1;
const DELTA_SKIP_MAGIC: u32 = 0x184D_2A50; // a skippable frame stock zstd ignores
const DELTA_TAG: &[u8; 8] = b"HBDELTA1";
/// A delta file's header: skippable-frame magic (4), frame length 48 (4), tag (8), base hash (32), plain length (8).
const DELTA_HEAD_LEN: u64 = 56;
const SEAL_TAG: &[u8; 8] = b"HBSEAL2\0";
const SEAL_TRAILER: &[u8; 4] = b"HBS2";
const NAME_INFO: &[u8] = b"hashed-blobs name v1";
const FRAME_INFO: &[u8] = b"hashed-blobs frame v2";

/// A blob's name: 32 bytes, written `blake3:<hex>`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Hash(pub [u8; 32]);

impl Hash {
    pub fn of(bytes: &[u8]) -> Hash {
        Hash(*blake3::hash(bytes).as_bytes())
    }
    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "blake3:{}", self.hex())
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "blake3:{}", self.hex())
    }
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    /// No root holds this blob.
    Missing(Hash),
    /// The bytes on disk do not match: a name mismatch, a frame checksum or an AEAD tag.
    Corrupt { blob: Hash, what: String },
    /// A sealed blob was read without an unwrap callback.
    Locked(Hash),
    /// A range read of a delta blob.
    NotSeekable(Hash),
    /// A request the crate refuses (a delta across a key boundary, a delta on a delta, a range
    /// past the end).
    Refused(String),
    /// The caller's unwrap callback declined (wrong or destroyed key).
    Unwrap(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Missing(h) => write!(f, "blob {h} is in no root"),
            Error::Corrupt { blob, what } => write!(f, "blob {blob} is corrupt: {what}"),
            Error::Locked(h) => write!(f, "blob {h} is sealed and no key was given"),
            Error::NotSeekable(h) => write!(f, "blob {h} is a delta and cannot be read by range"),
            Error::Refused(s) => write!(f, "refused: {s}"),
            Error::Unwrap(s) => write!(f, "key unavailable: {s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// A material's data key, held by the caller only while it is unwrapped.
#[derive(Clone)]
pub struct DataKey {
    pub key_ref: String,
    pub key: [u8; 32],
}

/// The caller's way to obtain a data key for a key reference (an unlock session in Sinex).
pub type Unwrap<'a> = &'a dyn Fn(&str) -> std::result::Result<[u8; 32], String>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    Plain,
    Delta { base: Hash },
    Sealed { key_ref: String, base: Option<Hash> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verified {
    /// Content read back and matched its name.
    Intact,
    /// Sealed, checked without the key: the ciphertext matches its recorded BLAKE3.
    CiphertextIntact,
    Corrupt(String),
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    pub blob: Hash,
    pub offset: u64,
    pub len: u64,
}

pub struct Store {
    write_root: PathBuf,
    roots: Vec<PathBuf>,
    mode: u32,
    frame: usize,
}

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn hkdf32(ikm: &[u8; 32], info: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    hkdf::Hkdf::<sha2::Sha256>::new(None, ikm)
        .expand(info, &mut out)
        .expect("32 bytes is a valid HKDF-SHA256 length");
    out
}

fn frame_key(data_key: &[u8; 32], name: &Hash, salt: &[u8; 32]) -> [u8; 32] {
    let mut info = FRAME_INFO.to_vec();
    info.extend_from_slice(&name.0);
    let mut out = [0u8; 32];
    hkdf::Hkdf::<sha2::Sha256>::new(Some(salt), data_key)
        .expand(&info, &mut out)
        .expect("32 bytes is a valid HKDF-SHA256 length");
    out
}

fn nonce(k: u32) -> [u8; 12] {
    let mut n = [0u8; 12];
    n[8..].copy_from_slice(&k.to_be_bytes());
    n
}

/// Fixed encoding context and the current entry authenticate bounded independent range reads.
fn aad(f: &SealedFooter, name: &Hash, k: u32, plain_start: u64) -> Vec<u8> {
    let mut a = f.context();
    a.extend_from_slice(&name.0);
    a.extend_from_slice(&k.to_le_bytes());
    let count = f.frames.len() as u32;
    a.extend_from_slice(&count.to_le_bytes());
    a.push((k + 1 == count) as u8);
    let (c, p) = f.frames[k as usize];
    a.extend_from_slice(&c.to_le_bytes());
    a.extend_from_slice(&p.to_le_bytes());
    a.extend_from_slice(&plain_start.to_le_bytes());
    a
}

fn compressor(level: i32) -> io::Result<zstd::bulk::Compressor<'static>> {
    let mut c = zstd::bulk::Compressor::new(level)?;
    c.set_parameter(zstd::zstd_safe::CParameter::ChecksumFlag(true))?;
    Ok(c)
}

fn zerr(code: usize) -> io::Error {
    io::Error::other(zstd::zstd_safe::get_error_name(code))
}

/// Window covering the base and the new content, as `zstd --patch-from` sets it; decoding needs
/// about that much memory plus the output (measured: 0.5 GB for a 256 MiB base).
fn delta_window_log(total: usize) -> u32 {
    (usize::BITS - total.max(1024).saturating_sub(1).leading_zeros()).clamp(10, 31)
}

/// A delta in the way `zstd --patch-from` makes one: the base as a prefix (not a dictionary),
/// a window over base and content, long-distance matching. A raw-content dictionary stops
/// finding matches once the base exceeds a few MiB (measured: no saving at 16 MiB and above).
fn delta_compress(level: i32, base: &[u8], data: &[u8]) -> io::Result<Vec<u8>> {
    use zstd::zstd_safe::{CCtx, CParameter};
    let mut c = CCtx::create();
    c.set_parameter(CParameter::CompressionLevel(level)).map_err(zerr)?;
    c.set_parameter(CParameter::ChecksumFlag(true)).map_err(zerr)?;
    c.set_parameter(CParameter::WindowLog(delta_window_log(base.len() + data.len()))).map_err(zerr)?;
    c.set_parameter(CParameter::EnableLongDistanceMatching(true)).map_err(zerr)?;
    c.ref_prefix(base).map_err(zerr)?;
    let mut out = Vec::with_capacity(zstd::zstd_safe::compress_bound(data.len()));
    c.compress2(&mut out, data).map_err(zerr)?;
    Ok(out)
}

fn delta_decompress(base: &[u8], src: &[u8], plain_len: usize) -> io::Result<Vec<u8>> {
    use zstd::zstd_safe::{DCtx, DParameter, InBuffer, OutBuffer};
    let mut d = DCtx::create();
    d.set_parameter(DParameter::WindowLogMax(31)).map_err(zerr)?;
    d.ref_prefix(base).map_err(zerr)?;
    let mut out: Vec<u8> = Vec::with_capacity(plain_len);
    {
        let mut ob = OutBuffer::around(&mut out);
        let mut ib = InBuffer::around(src);
        loop {
            let before = (ib.pos(), ob.pos());
            let r = d.decompress_stream(&mut ob, &mut ib).map_err(zerr)?;
            if r == 0 {
                break;
            }
            if (ib.pos(), ob.pos()) == before {
                return Err(io::Error::other("delta frame truncated or larger than recorded"));
            }
        }
    }
    if out.len() != plain_len {
        return Err(io::Error::other("delta length differs from its header"));
    }
    Ok(out)
}

fn read_full(r: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..])? {
            0 => break,
            k => n += k,
        }
    }
    Ok(n)
}

fn fsync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// The sealed layout's footer, in file order after the frames.
#[derive(Debug, Clone)]
pub(crate) struct SealedFooter {
    salt: [u8; 32],
    key_ref: String,
    base: Option<Hash>,
    plain_len: u64,
    /// Per frame: ciphertext length, plaintext (uncompressed) length.
    frames: Vec<(u32, u32)>,
    cipher_hash: Hash,
}

impl SealedFooter {
    fn new(key: &DataKey, base: Option<Hash>, plain_len: u64, count: u32) -> Result<Self> {
        let mut salt = [0u8; 32];
        // Every attempted encoding gets new entropy before encrypting, even if install later dedups.
        OsRng.try_fill_bytes(&mut salt).map_err(|e| io::Error::other(format!("encoding entropy: {e}")))?;
        Ok(Self { salt, key_ref: key.key_ref.clone(), base, plain_len,
            frames: vec![(0, 0); count as usize], cipher_hash: Hash([0; 32]) })
    }

    fn context(&self) -> Vec<u8> {
        let mut f = SEAL_TAG.to_vec();
        f.extend_from_slice(&self.salt);
        f.extend_from_slice(&(self.key_ref.len() as u16).to_le_bytes());
        f.extend_from_slice(self.key_ref.as_bytes());
        match self.base {
            Some(b) => {
                f.push(1);
                f.extend_from_slice(&b.0);
            }
            None => f.push(0),
        }
        f.extend_from_slice(&self.plain_len.to_le_bytes());
        f
    }

    /// Everything but the ciphertext hash; the hash covers the frames plus these bytes.
    fn body(&self) -> Vec<u8> {
        let mut f = self.context();
        f.extend_from_slice(&(self.frames.len() as u32).to_le_bytes());
        for (c, p) in &self.frames {
            f.extend_from_slice(&c.to_le_bytes());
            f.extend_from_slice(&p.to_le_bytes());
        }
        f
    }

    fn encode(&self) -> Vec<u8> {
        let mut f = self.body();
        f.extend_from_slice(&self.cipher_hash.0);
        let len = f.len() as u32;
        f.extend_from_slice(&len.to_le_bytes());
        f.extend_from_slice(SEAL_TRAILER);
        f
    }

    fn decode(file: &[u8], blob: Hash) -> Result<(SealedFooter, usize)> {
        let bad = |what: &str| Error::Corrupt { blob, what: what.to_string() };
        if file.len() < 8 || &file[file.len() - 4..] != SEAL_TRAILER {
            return Err(bad("no sealed trailer"));
        }
        let flen = u32_at(file, file.len() - 8) as usize;
        let start = file.len().checked_sub(8 + flen).ok_or_else(|| bad("footer length"))?;
        let f = &file[start..file.len() - 8];
        if f.len() < 8 + 32 + 2 || &f[..8] != SEAL_TAG {
            return Err(bad("footer tag"));
        }
        let salt = f[8..40].try_into().unwrap();
        let mut at = 40;
        let kl = u16::from_le_bytes(f[at..at + 2].try_into().unwrap()) as usize;
        at += 2;
        let key_ref = String::from_utf8(f.get(at..at + kl).ok_or_else(|| bad("key ref"))?.to_vec())
            .map_err(|_| bad("key ref"))?;
        at += kl;
        let base = match f.get(at) {
            Some(1) => {
                let b = Hash(f.get(at + 1..at + 33).ok_or_else(|| bad("base"))?.try_into().unwrap());
                at += 33;
                Some(b)
            }
            Some(0) => {
                at += 1;
                None
            }
            _ => return Err(bad("base flag")),
        };
        let need = at + 12;
        if f.len() < need {
            return Err(bad("footer short"));
        }
        let plain_len = u64_at(f, at);
        let n = u32_at(f, at + 8) as usize;
        at += 12;
        if n == 0 || f.len() != at + n * 8 + 32 {
            return Err(bad("footer frame table"));
        }
        let frames = (0..n).map(|i| (u32_at(f, at + i * 8), u32_at(f, at + i * 8 + 4))).collect();
        at += n * 8;
        let cipher_hash = Hash(f[at..at + 32].try_into().unwrap());
        Ok((SealedFooter { salt, key_ref, base, plain_len, frames, cipher_hash }, start))
    }
}

/// Positional reads of one blob file: readers fetch the seek table, header or footer and the frames
/// they need, never the whole file (a range read of a 15 GiB export reads a few MiB).
struct View {
    file: File,
    len: u64,
}

impl View {
    fn at(&self, off: u64, n: u64) -> io::Result<Vec<u8>> {
        use std::os::unix::fs::FileExt;
        if off.checked_add(n).is_none_or(|end| end > self.len) {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read past the end of the blob file"));
        }
        let mut b = vec![0u8; n as usize];
        self.file.read_exact_at(&mut b, off)?;
        Ok(b)
    }

    fn tail(&self, n: u64) -> io::Result<Vec<u8>> {
        let n = n.min(self.len);
        self.at(self.len - n, n)
    }
}

/// An append-only writer for a blob file in `blobs/tmp/`.
struct TmpFile {
    file: File,
    path: PathBuf,
    hasher: blake3::Hasher, // BLAKE3 of the bytes written (the ciphertext hash for sealed blobs)
}

impl TmpFile {
    fn write(&mut self, b: &[u8]) -> io::Result<()> {
        self.hasher.update(b);
        self.file.write_all(b)
    }
}

impl Store {
    /// One write root; `extra_roots` are searched in order on read. Nothing tiers or moves.
    pub fn open(write_root: impl Into<PathBuf>, extra_roots: Vec<PathBuf>, mode: u32) -> Result<Store> {
        let write_root = write_root.into();
        fs::create_dir_all(write_root.join("blobs").join("tmp"))?;
        let mut roots = vec![write_root.clone()];
        roots.extend(extra_roots);
        Ok(Store { write_root, roots, mode, frame: DEFAULT_FRAME })
    }

    /// Frame size for new plain and sealed blobs (4 MiB by default; smaller in tests).
    pub fn with_frame_size(mut self, frame: usize) -> Store {
        self.frame = frame;
        self
    }

    fn path_in(root: &Path, h: &Hash) -> PathBuf {
        let hex = h.hex();
        root.join("blobs").join(&hex[..2]).join(format!("{hex}.zst"))
    }

    fn locate(&self, h: &Hash) -> Option<PathBuf> {
        self.roots.iter().map(|r| Self::path_in(r, h)).find(|p| p.exists())
    }

    pub fn contains(&self, h: &Hash) -> bool {
        self.locate(h).is_some()
    }

    /// The bytes a blob's file takes on disk (what a caller records as its stored size).
    pub fn stored_size(&self, h: &Hash) -> Result<u64> {
        Ok(self.view(h)?.len)
    }

    fn tmp(&self) -> Result<TmpFile> {
        let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = self.write_root.join("blobs").join("tmp").join(format!("{}-{n}.part", std::process::id()));
        let file = OpenOptions::new().write(true).create_new(true).mode(self.mode).open(&path)?;
        Ok(TmpFile { file, path, hasher: blake3::Hasher::new() })
    }

    /// Sync the temporary file and rename it into place, syncing every directory touched. If the
    /// name already exists in any root, the new copy is dropped (same name, same content).
    fn install(&self, tmp: TmpFile, h: Hash) -> Result<Hash> {
        tmp.file.sync_all()?;
        drop(tmp.file);
        if self.locate(&h).is_some() {
            fs::remove_file(&tmp.path)?;
            return Ok(h);
        }
        let dest = Self::path_in(&self.write_root, &h);
        let dir = dest.parent().unwrap();
        if !dir.exists() {
            fs::create_dir_all(dir)?;
            fsync_dir(dir.parent().unwrap())?;
        }
        fs::rename(&tmp.path, &dest)?;
        fsync_dir(dir)?;
        fsync_dir(&self.write_root.join("blobs").join("tmp"))?;
        Ok(h)
    }

    fn view(&self, h: &Hash) -> Result<View> {
        let p = self.locate(h).ok_or(Error::Missing(*h))?;
        let file = File::open(p)?;
        let len = file.metadata()?.len();
        Ok(View { file, len })
    }

    fn sealed_footer(v: &View, h: Hash) -> Result<(SealedFooter, u64)> {
        let t8 = v.tail(8)?;
        if t8.len() < 8 {
            return Err(Error::Corrupt { blob: h, what: "no sealed trailer".into() });
        }
        let suffix = v.tail(u32_at(&t8, 0) as u64 + 8)?;
        let (f, rel) = SealedFooter::decode(&suffix, h)?;
        Ok((f, v.len - suffix.len() as u64 + rel as u64))
    }

    fn kind_of_view(v: &View, h: Hash) -> Result<Kind> {
        let t8 = v.tail(8)?;
        if t8.len() >= 4 && &t8[t8.len() - 4..] == b"HBS1" {
            return Err(Error::Corrupt { blob: h, what: "unsupported sealed version 1".into() });
        }
        if t8.len() >= 4 && &t8[t8.len() - 4..] == SEAL_TRAILER {
            let (f, _) = Self::sealed_footer(v, h)?;
            return Ok(Kind::Sealed { key_ref: f.key_ref, base: f.base });
        }
        let head = v.at(0, v.len.min(DELTA_HEAD_LEN))?;
        if head.len() >= 8 && u32_at(&head, 0) == DELTA_SKIP_MAGIC {
            let (base, _, _) = Self::delta_header(&head, h)?;
            return Ok(Kind::Delta { base });
        }
        Ok(Kind::Plain)
    }

    pub fn kind(&self, h: &Hash) -> Result<Kind> {
        Self::kind_of_view(&self.view(h)?, *h)
    }

    /// The base a delta was made against; `None` for plain and non-delta sealed blobs.
    pub fn base_of(&self, h: &Hash) -> Result<Option<Hash>> {
        Ok(match self.kind(h)? {
            Kind::Delta { base } => Some(base),
            Kind::Sealed { base, .. } => base,
            Kind::Plain => None,
        })
    }

    // ---- plain ----

    /// Store bytes as a plain seekable blob named by their BLAKE3; durable when it returns.
    pub fn put(&self, bytes: impl Read, level: i32) -> Result<Hash> {
        let (tmp, h) = self.write_plain(bytes, level)?;
        self.install(tmp, h)
    }

    /// A group of puts made durable together (see `Batch`).
    pub fn batch(&self, durability: Durability) -> Batch<'_> {
        Batch { store: self, durability, pending: Vec::new() }
    }

    fn write_plain(&self, mut bytes: impl Read, level: i32) -> Result<(TmpFile, Hash)> {
        let mut tmp = self.tmp()?;
        let mut name = blake3::Hasher::new();
        let mut table: Vec<(u32, u32)> = Vec::new();
        let mut buf = vec![0u8; self.frame];
        let mut comp = compressor(level)?;
        loop {
            let n = read_full(&mut bytes, &mut buf)?;
            if n == 0 && !table.is_empty() {
                break;
            }
            name.update(&buf[..n]);
            let c = comp.compress(&buf[..n])?;
            tmp.write(&c)?;
            table.push((c.len() as u32, n as u32));
            if n < self.frame {
                break;
            }
        }
        let mut st = Vec::with_capacity(17 + table.len() * 8);
        st.extend_from_slice(&SEEK_SKIP_MAGIC.to_le_bytes());
        st.extend_from_slice(&((table.len() * 8 + 9) as u32).to_le_bytes());
        for (c, d) in &table {
            st.extend_from_slice(&c.to_le_bytes());
            st.extend_from_slice(&d.to_le_bytes());
        }
        st.extend_from_slice(&(table.len() as u32).to_le_bytes());
        st.push(0); // descriptor: no per-frame checksum entries; each frame carries its own
        st.extend_from_slice(&SEEKABLE_MAGIC.to_le_bytes());
        tmp.write(&st)?;
        Ok((tmp, Hash(*name.finalize().as_bytes())))
    }

    /// The seek table: per frame (offset in file, compressed length, decompressed length). Reads only
    /// the table at the end of the file.
    fn seek_table(v: &View, h: Hash) -> Result<Vec<(u64, u32, u32)>> {
        let bad = |what: &str| Error::Corrupt { blob: h, what: what.to_string() };
        let t9 = v.tail(9)?;
        if t9.len() < 9 || u32_at(&t9, 5) != SEEKABLE_MAGIC {
            return Err(bad("no seek table"));
        }
        let n = u32_at(&t9, 0) as u64;
        let tlen = 9 + n * 8 + 8;
        if tlen > v.len {
            return Err(bad("seek table length"));
        }
        let table = v.tail(tlen)?;
        if u32_at(&table, 0) != SEEK_SKIP_MAGIC {
            return Err(bad("seek table magic"));
        }
        let mut off = 0u64;
        let mut out = Vec::with_capacity(n as usize);
        for i in 0..n as usize {
            let c = u32_at(&table, 8 + i * 8);
            let d = u32_at(&table, 12 + i * 8);
            out.push((off, c, d));
            off += c as u64;
        }
        if off != v.len - tlen {
            return Err(bad("frames do not fill the file"));
        }
        Ok(out)
    }

    fn plain_frame(v: &View, h: Hash, (off, c, d): (u64, u32, u32)) -> Result<Vec<u8>> {
        let src = v.at(off, c as u64)?;
        zstd::bulk::decompress(&src, d as usize).map_err(|e| Error::Corrupt { blob: h, what: format!("frame at {off}: {e}") })
    }

    // ---- delta ----

    fn delta_header(file: &[u8], h: Hash) -> Result<(Hash, u64, usize)> {
        let bad = |what: &str| Error::Corrupt { blob: h, what: what.to_string() };
        let size = u32_at(file, 4) as usize;
        if size != 48 || file.len() < 8 + size || &file[8..16] != DELTA_TAG {
            return Err(bad("delta header"));
        }
        let base = Hash(file[16..48].try_into().unwrap());
        let plain_len = u64_at(file, 48);
        Ok((base, plain_len, 8 + size))
    }

    /// Store bytes as a zstd delta with a plain base's content as dictionary. Refuses a base that
    /// is itself a delta (reads stay at two blobs) or sealed (no delta across a key boundary, #19a).
    pub fn put_delta(&self, base: &Hash, mut bytes: impl Read, level: i32) -> Result<Hash> {
        match self.kind(base)? {
            Kind::Plain => {}
            Kind::Delta { .. } => return Err(Error::Refused(format!("base {base} is a delta; a delta's base is always plain"))),
            Kind::Sealed { .. } => return Err(Error::Refused(format!("base {base} is sealed; a plain delta never uses a sealed base (#19a)"))),
        }
        let base_bytes = self.read_all(base, None)?;
        let mut data = Vec::new();
        bytes.read_to_end(&mut data)?;
        let c = delta_compress(level, &base_bytes, &data)?;
        self.write_delta(base, &data, &c)
    }

    /// Store `bytes` as a delta against `base` only if the delta file would be at most `max_len` bytes, compressing
    /// once: the caller's delta-or-plain choice in a single compression (measuring first and then writing would
    /// compress twice, twice the time and memory on a large base). Returns `None`, writing nothing, when the delta is larger.
    pub fn put_delta_within(&self, base: &Hash, bytes: &[u8], level: i32, max_len: u64) -> Result<Option<Hash>> {
        if self.kind(base)? != Kind::Plain {
            return Err(Error::Refused(format!("base {base} is not plain; a delta's base is always plain")));
        }
        let base_bytes = self.read_all(base, None)?;
        let c = delta_compress(level, &base_bytes, bytes)?;
        if c.len() as u64 + DELTA_HEAD_LEN > max_len {
            return Ok(None);
        }
        self.write_delta(base, bytes, &c).map(Some)
    }

    fn write_delta(&self, base: &Hash, data: &[u8], c: &[u8]) -> Result<Hash> {
        let h = Hash::of(data);
        let mut tmp = self.tmp()?;
        let mut head = DELTA_SKIP_MAGIC.to_le_bytes().to_vec();
        head.extend_from_slice(&48u32.to_le_bytes());
        head.extend_from_slice(DELTA_TAG);
        head.extend_from_slice(&base.0);
        head.extend_from_slice(&(data.len() as u64).to_le_bytes());
        debug_assert_eq!(head.len() as u64, DELTA_HEAD_LEN);
        tmp.write(&head)?;
        tmp.write(c)?;
        self.install(tmp, h)
    }

    // ---- sealed ----

    fn name_key(key: &DataKey) -> [u8; 32] {
        hkdf32(&key.key, NAME_INFO)
    }

    /// The name a sealed blob of these bytes gets under this key (#24).
    pub fn sealed_name(key: &DataKey, bytes: &[u8]) -> Hash {
        Hash(*blake3::keyed_hash(&Self::name_key(key), bytes).as_bytes())
    }

    fn seal_frames(
        &self,
        tmp: &mut TmpFile,
        chunks: &mut dyn FnMut(&mut Vec<u8>) -> Result<usize>,
        footer: &mut SealedFooter,
        name: &Hash,
        key: &DataKey,
        level: i32,
        dict: Option<&[u8]>,
    ) -> Result<()> {
        let aead = ChaCha20Poly1305::new(Key::from_slice(&frame_key(&key.key, name, &footer.salt)));
        let count = footer.frames.len() as u32;
        let mut comp = compressor(level)?;
        let mut buf = Vec::new();
        let mut plain_start = 0u64;
        for k in 0..count {
            buf.clear();
            let n = chunks(&mut buf)?;
            let c = match dict {
                Some(d) => delta_compress(level, d, &buf[..n])?,
                None => comp.compress(&buf[..n])?,
            };
            footer.frames[k as usize] = ((c.len() + 16) as u32, n as u32);
            let ct = aead
                .encrypt(Nonce::from_slice(&nonce(k)), Payload { msg: &c, aad: &aad(footer, name, k, plain_start) })
                .map_err(|_| Error::Refused("encryption failed".into()))?;
            tmp.write(&ct)?;
            plain_start += n as u64;
        }
        Ok(())
    }

    fn finish_sealed(mut tmp: TmpFile, mut footer: SealedFooter) -> Result<TmpFile> {
        let body = footer.body();
        tmp.hasher.update(&body);
        footer.cipher_hash = Hash(*tmp.hasher.finalize().as_bytes());
        let enc = footer.encode();
        tmp.file.write_all(&enc)?;
        Ok(tmp)
    }

    /// Seal bytes under a data key. Two passes (the name keys the frames), so the input must seek;
    /// no plaintext is written to disk.
    pub fn put_sealed(&self, bytes: impl Read + Seek, level: i32, key: &DataKey) -> Result<Hash> {
        let (tmp, name) = self.write_sealed(bytes, level, key)?;
        self.install(tmp, name)
    }

    fn write_sealed(&self, mut bytes: impl Read + Seek, level: i32, key: &DataKey) -> Result<(TmpFile, Hash)> {
        let mut namer = blake3::Hasher::new_keyed(&Self::name_key(key));
        let mut buf = vec![0u8; self.frame];
        let mut total = 0u64;
        loop {
            let n = read_full(&mut bytes, &mut buf)?;
            namer.update(&buf[..n]);
            total += n as u64;
            if n < self.frame {
                break;
            }
        }
        let name = Hash(*namer.finalize().as_bytes());
        bytes.seek(SeekFrom::Start(0))?;
        let count = (total.div_ceil(self.frame as u64)).max(1) as u32;
        let frame = self.frame;
        let mut tmp = self.tmp()?;
        let mut next = |out: &mut Vec<u8>| -> Result<usize> {
            out.resize(frame, 0);
            let n = read_full(&mut bytes, out)?;
            out.truncate(n);
            Ok(n)
        };
        let mut footer = SealedFooter::new(key, None, total, count)?;
        self.seal_frames(&mut tmp, &mut next, &mut footer, &name, key, level, None)?;
        Ok((Self::finish_sealed(tmp, footer)?, name))
    }

    /// A sealed delta: one sealed frame, compressed with the base's content as dictionary. The base
    /// must be a sealed, non-delta blob under the same key (#19a).
    pub fn put_delta_sealed(&self, base: &Hash, bytes: impl Read, level: i32, key: &DataKey) -> Result<Hash> {
        let (tmp, name) = self.write_delta_sealed(base, bytes, level, key)?;
        self.install(tmp, name)
    }

    fn write_delta_sealed(&self, base: &Hash, mut bytes: impl Read, level: i32, key: &DataKey) -> Result<(TmpFile, Hash)> {
        match self.kind(base)? {
            Kind::Sealed { key_ref, base: None } if key_ref == key.key_ref => {}
            Kind::Sealed { base: Some(_), .. } => return Err(Error::Refused(format!("base {base} is a delta; a delta's base is never a delta"))),
            Kind::Sealed { key_ref, .. } => return Err(Error::Refused(format!("base {base} is sealed under {key_ref}, not {}: a delta never crosses a key boundary (#19a)", key.key_ref))),
            _ => return Err(Error::Refused(format!("base {base} is plain; a sealed delta never uses a plain base (#19a)"))),
        }
        let k = key.key;
        let unwrap = move |_: &str| -> std::result::Result<[u8; 32], String> { Ok(k) };
        let base_bytes = self.read_all(base, Some(&unwrap))?;
        let mut data = Vec::new();
        bytes.read_to_end(&mut data)?;
        let name = Self::sealed_name(key, &data);
        let mut tmp = self.tmp()?;
        let mut once = Some(data.clone());
        let mut next = |out: &mut Vec<u8>| -> Result<usize> {
            let d = once.take().unwrap_or_default();
            let n = d.len();
            *out = d;
            Ok(n)
        };
        let mut footer = SealedFooter::new(key, Some(*base), data.len() as u64, 1)?;
        self.seal_frames(&mut tmp, &mut next, &mut footer, &name, key, level, Some(&base_bytes))?;
        Ok((Self::finish_sealed(tmp, footer)?, name))
    }

    fn open_sealed_frame(v: &View, h: Hash, f: &SealedFooter, dk: &[u8; 32], k: usize, off: u64, plain_start: u64, dict: Option<&[u8]>) -> Result<Vec<u8>> {
        let (c, p) = f.frames[k];
        let ct = v.at(off, c as u64).map_err(|_| Error::Corrupt { blob: h, what: "frame past the footer".into() })?;
        let ct = &ct[..];
        let aead = ChaCha20Poly1305::new(Key::from_slice(&frame_key(dk, &h, &f.salt)));
        let z = aead
            .decrypt(Nonce::from_slice(&nonce(k as u32)), Payload { msg: ct, aad: &aad(f, &h, k as u32, plain_start) })
            .map_err(|_| Error::Corrupt { blob: h, what: format!("frame {k} fails authentication (tampered, truncated, reordered, from another blob, or wrong key)") })?;
        let out = match dict {
            Some(d) => delta_decompress(d, &z, p as usize),
            None => zstd::bulk::decompress(&z, p as usize),
        };
        out.map_err(|e| Error::Corrupt { blob: h, what: format!("frame {k}: {e}") })
    }

    fn data_key(h: Hash, key_ref: &str, unwrap: Option<Unwrap>) -> Result<[u8; 32]> {
        let u = unwrap.ok_or(Error::Locked(h))?;
        u(key_ref).map_err(Error::Unwrap)
    }

    // ---- reading ----

    /// Every frame of a plain blob in order, checked by its zstd checksum; one frame in memory at a time.
    fn plain_frames(v: &View, h: Hash, mut each: impl FnMut(&[u8])) -> Result<()> {
        for fr in Self::seek_table(v, h)? {
            each(&Self::plain_frame(v, h, fr)?);
        }
        Ok(())
    }

    fn read_all(&self, h: &Hash, unwrap: Option<Unwrap>) -> Result<Vec<u8>> {
        let h = *h;
        let v = self.view(&h)?;
        let corrupt = |what: String| Error::Corrupt { blob: h, what };
        match Self::kind_of_view(&v, h)? {
            Kind::Plain => {
                let mut out = Vec::new();
                Self::plain_frames(&v, h, |b| out.extend_from_slice(b))?;
                if Hash::of(&out) != h {
                    return Err(corrupt("content does not match its name".into()));
                }
                Ok(out)
            }
            Kind::Delta { base } => {
                let file = v.at(0, v.len)?; // a delta is one frame, bounded by the caller's base-size limit
                let (_, plain_len, start) = Self::delta_header(&file, h)?;
                let base_bytes = self.read_all(&base, None).map_err(|e| match e {
                    Error::Missing(b) => corrupt(format!("its base {b} is missing")),
                    e => e,
                })?;
                let out = delta_decompress(&base_bytes, &file[start..], plain_len as usize)
                    .map_err(|e| corrupt(format!("delta frame: {e}")))?;
                if Hash::of(&out) != h {
                    return Err(corrupt("content does not match its name".into()));
                }
                Ok(out)
            }
            Kind::Sealed { key_ref, base } => {
                let dk = Self::data_key(h, &key_ref, unwrap)?;
                let (f, _) = Self::sealed_footer(&v, h)?;
                let dict = match base {
                    Some(b) => Some(self.read_all(&b, unwrap)?),
                    None => None,
                };
                let mut out = Vec::with_capacity(f.plain_len as usize);
                let mut off = 0u64;
                for k in 0..f.frames.len() {
                    out.extend(Self::open_sealed_frame(&v, h, &f, &dk, k, off, out.len() as u64, dict.as_deref())?);
                    off += f.frames[k].0 as u64;
                }
                let nk = hkdf32(&dk, NAME_INFO);
                if Hash(*blake3::keyed_hash(&nk, &out).as_bytes()) != h {
                    return Err(corrupt("content does not match its keyed name".into()));
                }
                Ok(out)
            }
        }
    }

    /// The whole content, checked against its name. Sealed blobs need `read_with`.
    pub fn read(&self, h: &Hash) -> Result<impl Read> {
        Ok(Cursor::new(self.read_all(h, None)?))
    }

    pub fn read_with(&self, h: &Hash, unwrap: Unwrap) -> Result<impl Read> {
        Ok(Cursor::new(self.read_all(h, Some(unwrap))?))
    }

    fn range(&self, h: &Hash, offset: u64, len: u64, unwrap: Option<Unwrap>) -> Result<Vec<u8>> {
        let h = *h;
        let v = self.view(&h)?;
        let kind = Self::kind_of_view(&v, h)?;
        // per frame: (offset in file, offset in content, content length)
        let (frames, total, sealed, plain_table) = match kind {
            Kind::Delta { .. } | Kind::Sealed { base: Some(_), .. } => return Err(Error::NotSeekable(h)),
            Kind::Plain => {
                let t = Self::seek_table(&v, h)?;
                let mut pos = 0;
                let fr: Vec<(u64, u64, u64)> = t.iter().map(|&(o, _, d)| { let r = (o, pos, d as u64); pos += d as u64; r }).collect();
                (fr, pos, None, t)
            }
            Kind::Sealed { key_ref, .. } => {
                let (f, _) = Self::sealed_footer(&v, h)?;
                let (mut o, mut pos) = (0u64, 0u64);
                let fr: Vec<(u64, u64, u64)> = f.frames.iter().map(|&(c, p)| { let r = (o, pos, p as u64); o += c as u64; pos += p as u64; r }).collect();
                let dk = Self::data_key(h, &key_ref, unwrap)?;
                (fr, pos, Some((f, dk)), Vec::new())
            }
        };
        if offset.checked_add(len).is_none_or(|end| end > total) {
            return Err(Error::Refused(format!("range {offset}+{len} past the end of {h} ({total} bytes)")));
        }
        let mut out = Vec::with_capacity(len as usize);
        for (k, &(foff, start, flen)) in frames.iter().enumerate() {
            if start + flen <= offset || start >= offset + len {
                continue;
            }
            let content = match &sealed {
                Some((f, dk)) => Self::open_sealed_frame(&v, h, f, dk, k, foff, start, None)?,
                None => Self::plain_frame(&v, h, plain_table[k])?,
            };
            let a = offset.saturating_sub(start) as usize;
            let b = ((offset + len).min(start + flen) - start) as usize;
            out.extend_from_slice(&content[a..b]);
        }
        Ok(out)
    }

    /// A byte range, decompressing only the frames it covers; frames are checked by their zstd
    /// checksum (plain) or AEAD tag (sealed). Deltas are not seekable.
    pub fn read_range(&self, h: &Hash, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.range(h, offset, len, None)
    }

    pub fn read_range_with(&self, h: &Hash, offset: u64, len: u64, unwrap: Unwrap) -> Result<Vec<u8>> {
        self.range(h, offset, len, Some(unwrap))
    }

    /// A revision's bytes from extents, as one seekable stream. Extents over a delta blob read the
    /// whole delta (only the raw view of an old revision does that).
    pub fn assemble(&self, extents: &[Extent]) -> Result<Assembled<'_>> {
        let mut starts = Vec::with_capacity(extents.len());
        let mut total = 0u64;
        for e in extents {
            if !self.contains(&e.blob) {
                return Err(Error::Missing(e.blob));
            }
            starts.push(total);
            total += e.len;
        }
        Ok(Assembled { store: self, extents: extents.to_vec(), starts, total, pos: 0 })
    }

    fn range_any(&self, h: &Hash, offset: u64, len: u64) -> Result<Vec<u8>> {
        match self.range(h, offset, len, None) {
            Err(Error::NotSeekable(_)) => {
                let all = self.read_all(h, None)?;
                all.get(offset as usize..(offset + len) as usize)
                    .map(|s| s.to_vec())
                    .ok_or_else(|| Error::Refused(format!("extent past the end of {h}")))
            }
            r => r,
        }
    }

    // ---- verify and delete ----

    /// Reports; never repairs. Streams: a plain blob is hashed frame by frame, a sealed one by its
    /// ciphertext in 4 MiB reads without the key, so a 15 GiB blob never sits in memory.
    pub fn verify(&self, h: &Hash) -> Result<Verified> {
        let v = match self.view(h) {
            Err(Error::Missing(_)) => return Ok(Verified::Missing),
            r => r?,
        };
        let report = |e: Error| -> Result<Verified> {
            match e {
                Error::Corrupt { what, blob } => Ok(Verified::Corrupt(format!("blob {blob}: {what}"))),
                e => Err(e),
            }
        };
        let kind = match Self::kind_of_view(&v, *h) {
            Ok(k) => k,
            Err(e) => return report(e),
        };
        match kind {
            Kind::Sealed { .. } => {
                let (f, start) = match Self::sealed_footer(&v, *h) {
                    Ok(x) => x,
                    Err(e) => return report(e),
                };
                let mut hs = blake3::Hasher::new();
                let mut off = 0u64;
                while off < start {
                    let n = (start - off).min(4 << 20);
                    hs.update(&v.at(off, n)?);
                    off += n;
                }
                hs.update(&f.body());
                Ok(if Hash(*hs.finalize().as_bytes()) == f.cipher_hash {
                    Verified::CiphertextIntact
                } else {
                    Verified::Corrupt(format!("blob {h}: ciphertext does not match its recorded BLAKE3"))
                })
            }
            Kind::Plain => {
                let mut hs = blake3::Hasher::new();
                if let Err(e) = Self::plain_frames(&v, *h, |b| { hs.update(b); }) {
                    return report(e);
                }
                Ok(if Hash(*hs.finalize().as_bytes()) == *h {
                    Verified::Intact
                } else {
                    Verified::Corrupt(format!("blob {h}: content does not match its name"))
                })
            }
            Kind::Delta { .. } => match self.read_all(h, None) {
                Ok(_) => Ok(Verified::Intact),
                Err(e) => report(e),
            },
        }
    }

    /// Full check of a sealed blob with its key: every frame authenticates and the keyed name holds.
    pub fn verify_with(&self, h: &Hash, unwrap: Unwrap) -> Result<Verified> {
        Ok(match self.read_all(h, Some(unwrap)) {
            Ok(_) => Verified::Intact,
            Err(Error::Missing(_)) => Verified::Missing,
            Err(Error::Corrupt { what, blob }) => Verified::Corrupt(format!("blob {blob}: {what}")),
            Err(e) => return Err(e),
        })
    }

    /// Delete a blob's file. The caller checks references (a delta's base included).
    pub fn delete(&self, h: &Hash) -> Result<()> {
        let p = self.locate(h).ok_or(Error::Missing(*h))?;
        fs::remove_file(&p)?;
        fsync_dir(p.parent().unwrap())?;
        Ok(())
    }
}

/// How a batch is made durable. Measured on the operator's QLC drive (PT5): three syncs per blob
/// spent 97% of the time storing a day of media (8 MiB/s); a batch shares them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Durability {
    /// Sync every file, then every directory the batch touched once.
    PerFile,
    /// One `syncfs` before the renames and one after: the filesystem's own group commit.
    Filesystem,
}

/// Puts that become visible and durable together. Nothing is visible under its name until
/// `commit` returns; when it returns, every blob's bytes and name are durable. Renames happen only
/// after the bytes are durable, so a crash never leaves a name over incomplete bytes; it can leave
/// files in `blobs/tmp/`, which are harmless and removable.
pub struct Batch<'a> {
    store: &'a Store,
    durability: Durability,
    pending: Vec<(TmpFile, Hash)>,
}

fn syncfs(f: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    if unsafe { libc::syncfs(f.as_raw_fd()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

impl Batch<'_> {
    pub fn put(&mut self, bytes: impl Read, level: i32) -> Result<Hash> {
        let (tmp, h) = self.store.write_plain(bytes, level)?;
        self.pending.push((tmp, h));
        Ok(h)
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub fn commit(self) -> Result<Vec<Hash>> {
        let s = self.store;
        let tmpdir = s.write_root.join("blobs").join("tmp");
        let root = File::open(&s.write_root)?;
        match self.durability {
            Durability::PerFile => {
                for (t, _) in &self.pending {
                    t.file.sync_all()?;
                }
            }
            Durability::Filesystem => syncfs(&root)?,
        }
        let mut dirs = std::collections::BTreeSet::new();
        let mut out = Vec::with_capacity(self.pending.len());
        for (t, h) in self.pending {
            drop(t.file);
            if s.locate(&h).is_some() {
                fs::remove_file(&t.path)?;
                out.push(h);
                continue;
            }
            let dest = Store::path_in(&s.write_root, &h);
            let dir = dest.parent().unwrap().to_path_buf();
            if !dir.exists() {
                fs::create_dir_all(&dir)?;
                dirs.insert(dir.parent().unwrap().to_path_buf());
            }
            fs::rename(&t.path, &dest)?;
            dirs.insert(dir);
            out.push(h);
        }
        dirs.insert(tmpdir);
        match self.durability {
            Durability::PerFile => {
                for d in &dirs {
                    fsync_dir(d)?;
                }
            }
            Durability::Filesystem => syncfs(&root)?,
        }
        Ok(out)
    }
}

pub struct Assembled<'a> {
    store: &'a Store,
    extents: Vec<Extent>,
    starts: Vec<u64>,
    total: u64,
    pos: u64,
}

impl Read for Assembled<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.total || buf.is_empty() {
            return Ok(0);
        }
        let i = match self.starts.binary_search(&self.pos) {
            Ok(i) => {
                // several zero-length extents may share a start; take the last one that has bytes
                let mut j = i;
                while j + 1 < self.starts.len() && self.starts[j + 1] == self.pos {
                    j += 1;
                }
                j
            }
            Err(i) => i - 1,
        };
        let e = self.extents[i];
        let within = self.pos - self.starts[i];
        let take = (buf.len() as u64).min(e.len - within).min(1 << 20);
        let data = self.store.range_any(&e.blob, e.offset + within, take).map_err(io::Error::other)?;
        buf[..data.len()].copy_from_slice(&data);
        self.pos += data.len() as u64;
        Ok(data.len())
    }
}

impl Seek for Assembled<'_> {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        let p = match to {
            SeekFrom::Start(p) => p as i128,
            SeekFrom::End(d) => self.total as i128 + d as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
        };
        if p < 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "seek before the start"));
        }
        self.pos = p as u64;
        Ok(self.pos)
    }
}

#[cfg(test)]
mod tests;
