use super::*;
use std::os::unix::fs::PermissionsExt;

const FRAME: usize = 64 * 1024;

fn store() -> (tempfile::TempDir, Store) {
    let d = tempfile::tempdir().unwrap();
    let s = Store::open(d.path().join("root"), vec![], 0o600).unwrap().with_frame_size(FRAME);
    (d, s)
}

/// Synthetic, compressible, and different per seed.
fn data(seed: u64, len: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        out.extend_from_slice(format!("record {} value {}\n", x >> 40, (x >> 20) & 0xff).as_bytes());
    }
    out.truncate(len);
    out
}

fn key(r: &str, b: u8) -> DataKey {
    DataKey { key_ref: r.into(), key: [b; 32] }
}

fn read_all(r: impl Read) -> Vec<u8> {
    let mut r = r;
    let mut v = Vec::new();
    r.read_to_end(&mut v).unwrap();
    v
}

fn file_of(s: &Store, h: &Hash) -> PathBuf {
    s.locate(h).unwrap()
}

// ---- plain ----

#[test]
fn put_read_named_by_blake3_and_dedup() {
    let (_d, s) = store();
    let b = data(1, 5 * FRAME + 123);
    let h = s.put(&b[..], 3).unwrap();
    assert_eq!(h, Hash::of(&b));
    assert_eq!(read_all(s.read(&h).unwrap()), b);
    assert_eq!(s.put(&b[..], 3).unwrap(), h);
    let tmp: Vec<_> = fs::read_dir(s.write_root.join("blobs/tmp")).unwrap().collect();
    assert!(tmp.is_empty(), "a deduplicated put leaves nothing in tmp");
    let empty = s.put(&b""[..], 3).unwrap();
    assert_eq!(read_all(s.read(&empty).unwrap()), b"");
    let exact = data(2, 2 * FRAME);
    let he = s.put(&exact[..], 3).unwrap();
    assert_eq!(read_all(s.read(&he).unwrap()), exact);
}

#[test]
fn layout_and_mode() {
    let (_d, s) = store();
    let h = s.put(&data(3, 1000)[..], 3).unwrap();
    let p = file_of(&s, &h);
    let hex = h.hex();
    assert!(p.ends_with(format!("blobs/{}/{}.zst", &hex[..2], hex)));
    assert_eq!(fs::metadata(&p).unwrap().permissions().mode() & 0o777, 0o600);
}

#[test]
fn stock_zstd_decompresses_plain_blobs() {
    let (_d, s) = store();
    let b = data(4, 3 * FRAME + 7);
    let h = s.put(&b[..], 3).unwrap();
    let out = zstd::stream::decode_all(File::open(file_of(&s, &h)).unwrap()).unwrap();
    assert_eq!(out, b, "frames plus a skippable seek table decode with a stock decoder");
}

#[test]
fn range_reads_across_frames() {
    let (_d, s) = store();
    let b = data(5, 4 * FRAME + 999);
    let h = s.put(&b[..], 3).unwrap();
    for (o, l) in [(0, 10), (FRAME as u64 - 5, 10), (3 * FRAME as u64 + 1, FRAME as u64 + 900), (0, b.len() as u64), (b.len() as u64, 0)] {
        assert_eq!(s.read_range(&h, o, l).unwrap(), &b[o as usize..(o + l) as usize]);
    }
    assert!(matches!(s.read_range(&h, b.len() as u64 - 1, 2), Err(Error::Refused(_))));
}

#[test]
fn corruption_found_by_the_crate_not_the_filesystem() {
    let (_d, s) = store();
    let b = data(6, 4 * FRAME);
    let h = s.put(&b[..], 3).unwrap();
    let p = file_of(&s, &h);
    let mut f = fs::read(&p).unwrap();
    let table = { let v = s.view(&h).unwrap(); Store::seek_table_checked::<Error>(&v, h, || Ok(())).unwrap() };
    let (off, c, _) = table[2];
    f[off as usize + c as usize / 2] ^= 0x40;
    fs::write(&p, &f).unwrap();
    assert!(matches!(s.read(&h), Err(Error::Corrupt { .. })), "a full read fails");
    match s.verify(&h).unwrap() {
        Verified::Corrupt(m) => assert!(m.contains(&h.hex()), "the report names the blob: {m}"),
        v => panic!("verify said {v:?}"),
    }
    assert!(matches!(s.read_range(&h, 2 * FRAME as u64, 10), Err(Error::Corrupt { .. })), "the damaged frame fails a range read");
    assert!(s.read_range(&h, 0, 10).is_err(), "stateless addressing validates the complete frame map");
    assert_eq!(fs::read(&p).unwrap(), f, "verify reports and never repairs");
}

#[test]
fn leftover_tmp_files_are_harmless() {
    let (_d, s) = store();
    fs::write(s.write_root.join("blobs/tmp/999-0.part"), b"half a blob").unwrap();
    let b = data(7, 100);
    let h = s.put(&b[..], 3).unwrap();
    assert_eq!(read_all(s.read(&h).unwrap()), b);
}

#[test]
fn extra_roots_are_read_in_order_and_never_written() {
    let d = tempfile::tempdir().unwrap();
    let old = Store::open(d.path().join("old"), vec![], 0o600).unwrap().with_frame_size(FRAME);
    let b = data(8, 2 * FRAME);
    let h = old.put(&b[..], 3).unwrap();
    let s = Store::open(d.path().join("new"), vec![d.path().join("old")], 0o600).unwrap().with_frame_size(FRAME);
    assert_eq!(read_all(s.read(&h).unwrap()), b);
    assert_eq!(s.put(&b[..], 3).unwrap(), h);
    assert!(!Store::path_in(&d.path().join("new"), &h).exists(), "a blob found in another root is not copied");
    let h2 = s.put(&data(9, 10)[..], 3).unwrap();
    assert!(Store::path_in(&d.path().join("new"), &h2).exists());
}

// ---- delta ----

#[test]
fn delta_against_a_plain_base() {
    let (_d, s) = store();
    let base = data(10, 3 * FRAME);
    let mut next = base.clone();
    next[FRAME..FRAME + 40].copy_from_slice(&[b'x'; 40]);
    next.extend_from_slice(b"appended tail\n");
    let hb = s.put(&base[..], 3).unwrap();
    let hd = s.put_delta(&hb, &next[..], 3).unwrap();
    assert_eq!(hd, Hash::of(&next));
    assert_eq!(s.base_of(&hd).unwrap(), Some(hb));
    assert_eq!(s.base_of(&hb).unwrap(), None);
    assert_eq!(read_all(s.read(&hd).unwrap()), next);
    let dsize = fs::metadata(file_of(&s, &hd)).unwrap().len();
    assert_eq!(s.stored_size(&hd).unwrap(), dsize);
    let plain = zstd::bulk::compress(&next, 3).unwrap().len() as u64;
    assert!(dsize * 10 < plain, "a delta of a small change is small: {dsize} vs {plain}");
    assert!(matches!(s.read_range(&hd, 0, 1), Err(Error::NotSeekable(_))));
    assert!(matches!(s.put_delta(&hd, &b"x"[..], 3), Err(Error::Refused(_))), "a delta's base is never a delta");
    s.delete(&hb).unwrap();
    match s.read(&hd) {
        Err(Error::Corrupt { what, .. }) => assert!(what.contains(&hb.hex()), "the error names the missing base: {what}"),
        _ => panic!("reading a delta without its base must fail"),
    }
}

#[test]
fn assemble_extents_as_one_seekable_stream() {
    let (_d, s) = store();
    let a = data(11, 2 * FRAME + 10);
    let b = data(12, 1000);
    let ha = s.put(&a[..], 3).unwrap();
    let hb = s.put(&b[..], 3).unwrap();
    let hd = s.put_delta(&ha, &b[..], 3).unwrap();
    let ext = [
        Extent { blob: ha, offset: 5, len: FRAME as u64 },
        Extent { blob: hb, offset: 0, len: 0 },
        Extent { blob: hb, offset: 100, len: 500 },
        Extent { blob: hd, offset: 10, len: 20 },
    ];
    let mut want = a[5..5 + FRAME].to_vec();
    want.extend_from_slice(&b[100..600]);
    want.extend_from_slice(&b[10..30]);
    let mut r = s.assemble(&ext).unwrap();
    assert_eq!(read_all(&mut r), want);
    r.seek(SeekFrom::Start(FRAME as u64 - 3)).unwrap();
    let mut six = [0u8; 6];
    r.read_exact(&mut six).unwrap();
    assert_eq!(&six, &want[FRAME - 3..FRAME + 3]);
    assert_eq!(r.seek(SeekFrom::End(-20)).unwrap(), want.len() as u64 - 20);
    assert_eq!(read_all(&mut r), &want[want.len() - 20..]);
}

/// The primitives the caller's excision rewrite needs (the rewrite itself stays out of the crate):
/// a new blob for the cleaned bytes, extents repointed, the old blob deleted and gone from reads.
#[test]
fn excision_primitives() {
    let (_d, s) = store();
    let mut b = data(13, FRAME + 500);
    let old = s.put(&b[..], 3).unwrap();
    b[100..120].copy_from_slice(&[b' '; 20]);
    let new = s.put(&b[..], 3).unwrap();
    s.delete(&old).unwrap();
    assert!(!s.contains(&old));
    assert!(matches!(s.read(&old), Err(Error::Missing(_))));
    assert_eq!(s.verify(&old).unwrap(), Verified::Missing);
    assert!(matches!(s.delete(&old), Err(Error::Missing(_))));
    let mut r = s.assemble(&[Extent { blob: new, offset: 0, len: b.len() as u64 }]).unwrap();
    assert_eq!(read_all(&mut r), b);
}

// ---- sealed (#19, #23, #24) ----

fn unwrapper(keys: &[DataKey]) -> impl Fn(&str) -> std::result::Result<[u8; 32], String> + '_ {
    move |r: &str| keys.iter().find(|k| k.key_ref == r).map(|k| k.key).ok_or_else(|| format!("no key for {r}"))
}

#[test]
fn sealed_round_trip_and_names() {
    let (_d, s) = store();
    let k1 = key("mat-1", 1);
    let k2 = key("mat-2", 2);
    let b = data(20, 3 * FRAME + 77);
    let h1 = s.put_sealed(Cursor::new(&b), 3, &k1).unwrap();
    let keys = [k1.clone(), k2.clone()];
    let u = unwrapper(&keys);
    assert_eq!(read_all(s.read_with(&h1, &u).unwrap()), b);
    assert_eq!(s.read_range_with(&h1, FRAME as u64 - 2, 4, &u).unwrap(), &b[FRAME - 2..FRAME + 2]);
    assert!(matches!(s.read(&h1), Err(Error::Locked(_))));
    assert_ne!(h1, Hash::of(&b), "#24: a sealed name is not the plaintext BLAKE3");
    let h2 = s.put_sealed(Cursor::new(&b), 3, &k2).unwrap();
    let hp = s.put(&b[..], 3).unwrap();
    assert!(h1 != h2 && h1 != hp && h2 != hp, "plain and each key get their own file");
    assert_eq!(s.put_sealed(Cursor::new(&b), 3, &k1).unwrap(), h1, "same key, same content: one file");
    s.delete(&h1).unwrap();
    assert_eq!(read_all(s.read_with(&h2, &u).unwrap()), b, "deleting one key's copy leaves the others");
    assert_eq!(read_all(s.read(&hp).unwrap()), b);
    let raw = fs::read(file_of(&s, &h2)).unwrap();
    assert!(!raw.windows(20).any(|w| b.windows(20).next() == Some(w)), "no plaintext in the file");
}

#[test]
fn sealed_verify_without_and_with_the_key() {
    let (_d, s) = store();
    let k = key("mat-1", 1);
    let h = s.put_sealed(Cursor::new(data(21, 2 * FRAME)), 3, &k).unwrap();
    let keys = [k.clone()];
    let u = unwrapper(&keys);
    assert_eq!(s.verify(&h).unwrap(), Verified::CiphertextIntact);
    assert_eq!(s.verify_with(&h, &u).unwrap(), Verified::Intact);
    let p = file_of(&s, &h);
    let mut f = fs::read(&p).unwrap();
    f[100] ^= 1;
    fs::write(&p, &f).unwrap();
    assert!(matches!(s.verify(&h).unwrap(), Verified::Corrupt(_)), "corruption is found without the key");
    assert!(matches!(s.verify_with(&h, &u).unwrap(), Verified::Corrupt(_)));
}

#[test]
fn wrong_or_destroyed_key() {
    let (_d, s) = store();
    let k = key("mat-1", 1);
    let h = s.put_sealed(Cursor::new(data(22, 1000)), 3, &k).unwrap();
    let wrong = [key("mat-1", 9)];
    assert!(matches!(s.read_with(&h, &unwrapper(&wrong)), Err(Error::Corrupt { .. })));
    let none: [DataKey; 0] = [];
    assert!(matches!(s.read_with(&h, &unwrapper(&none)), Err(Error::Unwrap(_))), "shredded: the caller has no key to give");
}

/// Rewrite a sealed file with a new frame list and a recomputed ciphertext hash, as an attacker
/// with write access would; the AEAD must still refuse.
fn rewrite_sealed(file: &[u8], h: Hash, frames: Vec<Vec<u8>>, table: Vec<(u32, u32)>) -> Vec<u8> {
    let (mut f, _) = SealedFooter::decode(file, h).unwrap();
    f.frames = table;
    let mut out: Vec<u8> = frames.concat();
    let mut hs = blake3::Hasher::new();
    hs.update(&out);
    hs.update(&f.body());
    f.cipher_hash = Hash(*hs.finalize().as_bytes());
    out.extend(f.encode());
    out
}

fn split_frames(file: &[u8], h: Hash) -> (Vec<Vec<u8>>, Vec<(u32, u32)>) {
    let (f, _) = SealedFooter::decode(file, h).unwrap();
    let mut off = 0usize;
    let mut out = Vec::new();
    for &(c, _) in &f.frames {
        out.push(file[off..off + c as usize].to_vec());
        off += c as usize;
    }
    (out, f.frames)
}

#[test]
fn sealed_attacks_fail_authentication() {
    let (_d, s) = store();
    let k = key("mat-1", 1);
    let keys = [k.clone()];
    let u = unwrapper(&keys);
    let a = data(23, 4 * FRAME);
    let b = data(24, 4 * FRAME);
    let ha = s.put_sealed(Cursor::new(&a), 3, &k).unwrap();
    let hb = s.put_sealed(Cursor::new(&b), 3, &k).unwrap();
    let pa = file_of(&s, &ha);
    let fa = fs::read(&pa).unwrap();
    let fb = fs::read(file_of(&s, &hb)).unwrap();
    let (frames_a, table_a) = split_frames(&fa, ha);
    let (frames_b, table_b) = split_frames(&fb, hb);

    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("truncated (last frame dropped)", rewrite_sealed(&fa, ha, frames_a[..3].to_vec(), table_a[..3].to_vec())),
        ("reordered (frames 1 and 2 swapped)", {
            let (mut fr, mut t) = (frames_a.clone(), table_a.clone());
            fr.swap(1, 2);
            t.swap(1, 2);
            rewrite_sealed(&fa, ha, fr, t)
        }),
        ("frame swapped in from another blob under the same key", {
            let (mut fr, mut t) = (frames_a.clone(), table_a.clone());
            fr[1] = frames_b[1].clone();
            t[1] = table_b[1];
            rewrite_sealed(&fa, ha, fr, t)
        }),
    ];
    for (what, bytes) in cases {
        fs::write(&pa, &bytes).unwrap();
        assert_eq!(s.verify(&ha).unwrap(), Verified::CiphertextIntact, "{what}: the forged ciphertext hash passes the keyless check");
        assert!(matches!(s.read_with(&ha, &u), Err(Error::Corrupt { .. })), "{what}: a full read must fail");
        assert!(matches!(s.verify_with(&ha, &u).unwrap(), Verified::Corrupt(_)), "{what}: verify with the key must fail");
        // A range read checks no name, so only the AEAD (index, count, last flag, name) protects it.
        let r0 = s.read_range_with(&ha, 0, 10, &u);
        let r1 = s.read_range_with(&ha, FRAME as u64 + 1, 10, &u);
        assert!(matches!(r0, Err(Error::Corrupt { .. })) || matches!(r1, Err(Error::Corrupt { .. })), "{what}: a range read must fail too");
    }
    // Truncation specifically: every surviving frame fails, because each was sealed with count 4.
    fs::write(&pa, rewrite_sealed(&fa, ha, frames_a[..3].to_vec(), table_a[..3].to_vec())).unwrap();
    assert!(matches!(s.read_range_with(&ha, 0, 10, &u), Err(Error::Corrupt { .. })), "frame 0 of a truncated blob must fail alone");
    fs::write(&pa, &fa).unwrap();
    assert_eq!(read_all(s.read_with(&ha, &u).unwrap()), a);
}

#[test]
fn no_nonce_reuse_across_blobs_under_one_key() {
    let k = key("mat-1", 1);
    let n1 = Store::sealed_name(&k, b"revision one");
    let n2 = Store::sealed_name(&k, b"revision two");
    assert_ne!(frame_key(&k.key, &n1, &[1; 32]), frame_key(&k.key, &n2, &[1; 32]), "#23: distinct frame keys, so (key, nonce) never repeats");
}

#[test]
fn sealed_deltas_stay_within_one_key() {
    let (_d, s) = store();
    let k1 = key("mat-1", 1);
    let k2 = key("mat-2", 2);
    let keys = [k1.clone(), k2.clone()];
    let u = unwrapper(&keys);
    let base = data(25, 2 * FRAME);
    let mut next = base.clone();
    next.extend_from_slice(b"one more line\n");
    let sb = s.put_sealed(Cursor::new(&base), 3, &k1).unwrap();
    let sd = s.put_delta_sealed(&sb, &next[..], 3, &k1).unwrap();
    assert_eq!(read_all(s.read_with(&sd, &u).unwrap()), next);
    assert_eq!(s.base_of(&sd).unwrap(), Some(sb));
    assert!(fs::metadata(file_of(&s, &sd)).unwrap().len() < 2000, "a sealed delta is small");
    let pb = s.put(&base[..], 3).unwrap();
    for (what, r) in [
        ("sealed delta on another key's base", s.put_delta_sealed(&sb, &next[..], 3, &k2)),
        ("sealed delta on a plain base", s.put_delta_sealed(&pb, &next[..], 3, &k1)),
        ("plain delta on a sealed base", s.put_delta(&sb, &next[..], 3)),
        ("sealed delta on a sealed delta", s.put_delta_sealed(&sd, &next[..], 3, &k1)),
    ] {
        assert!(matches!(r, Err(Error::Refused(_))), "{what} must be refused");
    }
    assert!(matches!(s.read_range_with(&sd, 0, 1, &u), Err(Error::NotSeekable(_))));
}

/// Incompressible data is stored in raw zstd blocks, where a flipped byte still decodes: only the
/// frame checksum catches it on a range read (the full read also has the name).
#[test]
fn flipped_byte_in_a_raw_block_is_caught_by_the_frame_checksum() {
    let (_d, s) = store();
    let mut b = vec![0u8; 3 * FRAME];
    blake3::Hasher::new().update(b"seed").finalize_xof().fill(&mut b);
    let h = s.put(&b[..], 3).unwrap();
    let p = file_of(&s, &h);
    let mut f = fs::read(&p).unwrap();
    let (off, c, _) = { let v = s.view(&h).unwrap(); Store::seek_table_checked::<Error>(&v, h, || Ok(())).unwrap()[1] };
    f[off as usize + c as usize / 2] ^= 0x01;
    fs::write(&p, &f).unwrap();
    assert!(matches!(s.read_range(&h, FRAME as u64, 100), Err(Error::Corrupt { .. })), "a range read of the damaged frame must fail");
    assert!(matches!(s.verify(&h).unwrap(), Verified::Corrupt(_)));
}

/// A base far past zstd's default window: the delta must still be small (a raw-content dictionary
/// gave no saving here; the patch-from parameters do).
#[test]
fn delta_against_a_large_base_stays_small() {
    let (_d, s) = store();
    let base = data(30, 24 << 20);
    let mut next = base.clone();
    next[12 << 20..(12 << 20) + 100].copy_from_slice(&[b'x'; 100]);
    next.extend_from_slice(b"tail\n");
    let hb = s.put(&base[..], 3).unwrap();
    let hd = s.put_delta(&hb, &next[..], 3).unwrap();
    let size = fs::metadata(file_of(&s, &hd)).unwrap().len();
    assert!(size < 64 * 1024, "delta of a 100-byte change to 24 MiB is {size} bytes");
    assert_eq!(read_all(s.read(&hd).unwrap()), next);
    let k = key("mat-1", 1);
    let keys = [k.clone()];
    let sb = s.put_sealed(Cursor::new(&base), 3, &k).unwrap();
    let sd = s.put_delta_sealed(&sb, &next[..], 3, &k).unwrap();
    assert!(fs::metadata(file_of(&s, &sd)).unwrap().len() < 64 * 1024, "sealed deltas too");
    assert_eq!(read_all(s.read_with(&sd, &unwrapper(&keys)).unwrap()), next);
}

#[test]
fn batch_success_establishes_durability_and_dedup() {
    for mode in [Durability::PerFile, Durability::Filesystem] {
        let (_d, s) = store();
        let items: Vec<Vec<u8>> = (0..20).map(|i| data(40 + i % 10, 3 * FRAME / 2)).collect(); // ten distinct, each twice
        let mut b = s.batch(mode);
        let hs: Vec<Hash> = items.iter().map(|x| b.put(&x[..], 1).unwrap()).collect();
        assert!(hs.iter().all(|h| !s.contains(h)), "{mode:?}: nothing is visible before commit");
        let done = b.commit().unwrap();
        assert_eq!(done, hs);
        for (x, h) in items.iter().zip(&hs) {
            assert_eq!(read_all(s.read(h).unwrap()), *x);
        }
        let tmp: Vec<_> = fs::read_dir(s.write_root.join("blobs/tmp")).unwrap().collect();
        assert!(tmp.is_empty(), "{mode:?}: duplicates inside a batch leave nothing in tmp");
    }
}

#[test]
fn put_delta_within_writes_only_under_the_bound() {
    let (_d, s) = store();
    let base = data(11, 3 * FRAME);
    let mut next = base.clone();
    next[FRAME..FRAME + 40].copy_from_slice(&[b'y'; 40]);
    let hb = s.put(&base[..], 3).unwrap();
    // the size a delta takes, measured in a second store
    let (_d2, s2) = store();
    let hb2 = s2.put(&base[..], 3).unwrap();
    let want = s2.stored_size(&s2.put_delta(&hb2, &next[..], 3).unwrap()).unwrap();
    // one byte too tight: nothing written
    assert_eq!(s.put_delta_within(&hb, &next, 3, want - 1).unwrap(), None);
    assert!(!s.contains(&Hash::of(&next)));
    // at the bound: written, with the predicted size
    let hd = s.put_delta_within(&hb, &next, 3, want).unwrap().expect("fits");
    assert_eq!(fs::metadata(file_of(&s, &hd)).unwrap().len(), want);
    assert_eq!(read_all(s.read(&hd).unwrap()), next);
}

#[test]
fn same_content_different_encoding_does_not_reuse_frame_keystream() {
    let d = tempfile::tempdir().unwrap();
    let k = key("neutral-material", 7);
    let bytes = data(41, 4096);
    let mut encoded = Vec::new();
    let mut compressed = Vec::new();
    for (index, frame, level) in [(0, 512, 3), (1, 1024, 9)] {
        let s = Store::open(d.path().join(index.to_string()), vec![], 0o600).unwrap().with_frame_size(frame);
        let name = s.put_sealed(Cursor::new(&bytes), level, &k).unwrap();
        let file = fs::read(file_of(&s, &name)).unwrap();
        encoded.push(split_frames(&file, name).0.remove(0));
        compressed.push(compressor(level).unwrap().compress(&bytes[..frame]).unwrap());
    }
    assert_ne!(compressed[0], compressed[1], "the encoded messages really differ");
    let n = compressed[0].len().min(compressed[1].len());
    let cipher_xor: Vec<_> = (0..n).map(|i| encoded[0][i] ^ encoded[1][i]).collect();
    let plain_xor: Vec<_> = (0..n).map(|i| compressed[0][i] ^ compressed[1][i]).collect();
    assert_ne!(cipher_xor, plain_xor, "same name, different encoding must not reuse the ChaCha frame keystream");
}

#[test]
fn encoding_salt_has_independent_hkdf_vectors_and_attempts_survive_dedup() {
    let k = key("neutral-material", 7);
    let name = Hash([3; 32]);
    // Independent HMAC-SHA256 extract/expand vectors, including a changed salt.
    assert_eq!(Hash(frame_key(&k.key, &name, &[0; 32])).hex(), "77859481b3c4c9ce7c4e69d4aa2101265a6212e312eb4bb466915a85852e055f");
    assert_eq!(Hash(frame_key(&k.key, &name, &[1; 32])).hex(), "26660414e15d0f44e1f91cf721d084977b16e782f9061f4f5d0eb3cd9a0b0ebc");
    let d = tempfile::tempdir().unwrap();
    let bytes = data(42, 4096);
    let s = Store::open(d.path(), vec![], 0o600).unwrap().with_frame_size(512);
    let base = s.put_sealed(Cursor::new(data(43, 4096)), 3, &k).unwrap();
    let expected = Store::sealed_name(&k, &bytes);
    let mut pairs = Vec::new();
    let mut installed = None;
    for (frame, level, delta) in [(512, 3, false), (512, 9, false), (1024, 3, false), (512, 3, true), (512, 3, false)] {
        let s = Store::open(d.path(), vec![], 0o600).unwrap().with_frame_size(frame);
        // Observe the real writer's completed encoding before install can discard it as a duplicate.
        let (tmp, h) = if delta { s.write_delta_sealed(&base, &bytes[..], level, &k) }
            else { s.write_sealed(Cursor::new(&bytes), level, &k) }.unwrap();
        assert_eq!(h, expected);
        let encoded = fs::read(&tmp.path).unwrap();
        let (footer, _) = SealedFooter::decode(&encoded, h).unwrap();
        assert_eq!(footer.base, if delta { Some(base) } else { None });
        assert_eq!(footer.frames.len(), if delta { 1 } else { bytes.len().div_ceil(frame) });
        let pair = (frame_key(&k.key, &h, &footer.salt), nonce(0));
        assert!(!pairs.contains(&pair), "every attempted encoding separates key/nonce, including discarded duplicates");
        pairs.push(pair);
        let temporary_path = tmp.path.clone();
        s.install(tmp, h, Some(&|_| Ok(k.key))).unwrap();
        assert!(!temporary_path.exists());
        let retained = fs::read(file_of(&s, &h)).unwrap();
        if let Some(first) = &installed { assert_eq!(&retained, first, "dedup keeps the installed encoding"); }
        else { installed = Some(retained); }
    }
    let keys = [k];
    let u = unwrapper(&keys);
    assert_eq!(read_all(s.read_with(&expected, &u).unwrap()), bytes);
    assert_eq!(s.read_range_with(&expected, 500, 40, &u).unwrap(), bytes[500..540]);
}

#[test]
fn sealed_encoding_metadata_is_authenticated_even_with_a_forged_cipher_hash() {
    let (_d, s) = store();
    let k = key("neutral-material", 7);
    let bytes = data(44, 2 * FRAME);
    let base = s.put_sealed(Cursor::new(data(45, FRAME)), 3, &k).unwrap();
    let h = s.put_sealed(Cursor::new(&bytes), 3, &k).unwrap();
    let path = file_of(&s, &h);
    let original = fs::read(&path).unwrap();
    let (footer, start) = SealedFooter::decode(&original, h).unwrap();
    let u = |_: &str| -> std::result::Result<[u8; 32], String> { Ok(k.key) };
    for field in ["salt", "key_ref", "plain_len", "frame_plain_len", "base"] {
        let mut changed = footer.clone();
        match field {
            "salt" => changed.salt[0] ^= 1,
            "key_ref" => changed.key_ref.push('x'),
            "plain_len" => changed.plain_len += 1,
            "frame_plain_len" => changed.frames[0].1 += 1,
            "base" => changed.base = Some(base),
            _ => unreachable!(),
        }
        let mut forged = original[..start].to_vec();
        let mut hasher = blake3::Hasher::new();
        hasher.update(&forged);
        hasher.update(&changed.body());
        changed.cipher_hash = Hash(*hasher.finalize().as_bytes());
        forged.extend(changed.encode());
        fs::write(&path, forged).unwrap();
        assert_eq!(s.verify(&h).unwrap(), Verified::CiphertextIntact);
        assert!(matches!(s.read_with(&h, &u), Err(Error::Corrupt { .. })), "{field} must authenticate");
        if field != "base" {
            assert!(matches!(s.read_range_with(&h, 0, 10, &u), Err(Error::Corrupt { .. })), "{field}: a range alone must authenticate");
        }
    }
    let mut obsolete = original;
    let n = obsolete.len();
    obsolete[n - 4..].copy_from_slice(b"HBS1");
    fs::write(&path, obsolete).unwrap();
    assert!(matches!(s.kind(&h), Err(Error::Corrupt { .. })), "v1 has no compatibility reader");
}

#[test]
fn empty_sealed_encoding_requires_an_authenticated_frame() {
    let (_d, s) = store();
    let k = key("neutral-empty", 7);
    let h = s.put_sealed(Cursor::new(b""), 3, &k).unwrap();
    let keys = [k];
    let u = unwrapper(&keys);
    assert!(read_all(s.read_with(&h, &u).unwrap()).is_empty());
    let path = file_of(&s, &h);
    let file = fs::read(&path).unwrap();
    let (mut footer, _) = SealedFooter::decode(&file, h).unwrap();
    assert_eq!(footer.frames.len(), 1);
    footer.frames.clear();
    footer.cipher_hash = Hash::of(&footer.body());
    fs::write(&path, footer.encode()).unwrap();
    assert!(matches!(s.read_with(&h, &u), Err(Error::Corrupt { .. })), "empty content cannot bypass salt/context authentication");
}

#[test]
fn a_later_range_authenticates_its_decoded_start() {
    let (_d, s) = store();
    let k = key("neutral-range", 7);
    let bytes = data(46, 2 * FRAME);
    let h = s.put_sealed(Cursor::new(&bytes), 3, &k).unwrap();
    let path = file_of(&s, &h);
    let file = fs::read(&path).unwrap();
    let (mut footer, start) = SealedFooter::decode(&file, h).unwrap();
    footer.frames[0].1 += 1; // ciphertext offsets and the selected frame's own entry are unchanged
    let mut forged = file[..start].to_vec();
    let mut hasher = blake3::Hasher::new();
    hasher.update(&forged);
    hasher.update(&footer.body());
    footer.cipher_hash = Hash(*hasher.finalize().as_bytes());
    forged.extend(footer.encode());
    fs::write(path, forged).unwrap();
    let keys = [k];
    let u = unwrapper(&keys);
    assert_eq!(s.verify(&h).unwrap(), Verified::CiphertextIntact);
    assert!(matches!(s.read_range_with(&h, FRAME as u64 + 20, 10, &u), Err(Error::Corrupt { .. })),
        "an earlier decoded-length mutation must not shift an authenticated later range");
}

#[test]
fn seek_metadata_refuses_one_bit_and_merged_unselected_frames() {
    let d = tempfile::tempdir().unwrap();
    let s = Store::open(d.path(), vec![], 0o600).unwrap().with_frame_size(8);
    let bytes = b"ABCDEFGHijklmnopQRSTUVWX";
    let h = s.put(&bytes[..], 1).unwrap();
    let path = file_of(&s, &h);
    let original = fs::read(&path).unwrap();
    let table = original.len() - 17 - 3 * 8;
    let mut bad = original.clone();
    bad[table + 12] ^= 1;
    fs::write(&path, &bad).unwrap();
    for offset in [8, 9, 16] {
        assert!(matches!(s.read_range(&h, offset, 1), Err(Error::Corrupt { .. })));
    }
    assert!(matches!(s.verify(&h), Ok(Verified::Corrupt(_))));
    assert!(s.verify_plain::<Error>(&h, 24, |_| Ok(())).is_err());
    // Merge the first two compressed extents, retain frame one's declared logical size,
    // and remove entry two. Header-only checks would accept displaced frame-three bytes.
    let mut bad = original.clone();
    let merged = u32_at(&bad, table + 8) + u32_at(&bad, table + 16);
    bad[table + 8..table + 12].copy_from_slice(&merged.to_le_bytes());
    bad.drain(table + 16..table + 24);
    let len = bad.len();
    bad[len - 9..len - 5].copy_from_slice(&2u32.to_le_bytes());
    bad[table + 4..table + 8].copy_from_slice(&25u32.to_le_bytes());
    fs::write(&path, &bad).unwrap();
    assert!(matches!(s.read_range(&h, 9, 1), Err(Error::Corrupt { .. })));
    assert!(s.verify_plain::<Error>(&h, 24, |_| Ok(())).is_err());
}

#[test]
fn corrupt_destinations_refuse_single_and_batch_without_repair() {
    let (_d, s) = store();
    let bytes = b"known correct new candidate";
    let h = s.put(&bytes[..], 1).unwrap();
    let path = file_of(&s, &h);
    fs::write(&path, b"corrupt destination").unwrap();
    assert!(s.put(&bytes[..], 1).is_err());
    let mut batch = s.batch(Durability::PerFile);
    batch.put(&bytes[..], 1).unwrap();
    assert!(batch.commit().is_err());
    assert_eq!(fs::read(&path).unwrap(), b"corrupt destination");
    assert_eq!(fs::read_dir(s.write_root.join("blobs/tmp")).unwrap().count(), 2);
}

#[test]
fn partial_batch_can_be_retried_without_rollback_or_false_success() {
    let (_d, s) = store();
    let first = b"first retained dependency";
    let mut second = b"second retained dependency".to_vec();
    while Hash::of(first).hex()[..2] == Hash::of(&second).hex()[..2] { second.push(b'x'); }
    let a = Hash::of(first);
    let b = Hash::of(&second);
    let obstruction = s.write_root.join("blobs").join(&b.hex()[..2]);
    fs::write(&obstruction, b"not a directory").unwrap();
    let mut batch = s.batch(Durability::PerFile);
    batch.put(&first[..], 1).unwrap();
    batch.put(&second[..], 1).unwrap();
    assert!(!s.contains(&a));
    assert!(batch.commit().is_err());
    assert_eq!(s.verify(&a).unwrap(), Verified::Intact);
    assert!(!s.contains(&b));
    fs::remove_file(obstruction).unwrap();
    let mut retry = s.batch(Durability::PerFile);
    retry.put(&first[..], 1).unwrap();
    retry.put(&second[..], 1).unwrap();
    assert_eq!(retry.commit().unwrap(), vec![a, b]);
    assert_eq!(s.verify(&b).unwrap(), Verified::Intact);
}

#[test]
fn sequential_verification_decodes_once_and_checks_identity_and_length() {
    let d = tempfile::tempdir().unwrap();
    let s = Store::open(d.path(), vec![], 0o600).unwrap();
    let bytes = data(37, 8 << 20);
    let h = s.put(&bytes[..], 1).unwrap();
    let mut visited = 0;
    let stats = s.verify_plain::<Error>(&h, bytes.len() as u64, |chunk| {
        assert!(chunk.len() <= 65536);
        assert_eq!(chunk, &bytes[visited..visited + chunk.len()]);
        visited += chunk.len();
        Ok(())
    }).unwrap();
    assert_eq!(stats.frames, 2);
    assert_eq!(stats.decoded_bytes, 8 << 20);
    assert_eq!(stats.compressed_bytes + 33, s.stored_size(&h).unwrap());
    assert_eq!(stats.metadata_bytes, 8 + 56 + 9 + 33 + 36);
    eprintln!("BOUNDED_VERIFY input={} decoded={} compressed={} metadata={} frames={} chunk_max=65536", bytes.len(), stats.decoded_bytes, stats.compressed_bytes, stats.metadata_bytes, stats.frames);
    assert!(s.verify_plain::<Error>(&h, bytes.len() as u64 - 1, |_| Ok(())).is_err());
    let other = s.put(&b"other valid object"[..], 1).unwrap();
    fs::copy(file_of(&s, &other), file_of(&s, &h)).unwrap();
    assert_eq!(s.read_range(&h, 0, 1).unwrap(), b"o"); // Stateless range proof boundary.
    assert!(Reader::new(d.path(), vec![]).read_range(&h, 0, 1).is_err());
    assert!(s.verify_plain::<Error>(&h, 18, |_| Ok(())).is_err());
    let empty = s.put(&b""[..], 1).unwrap();
    assert_eq!(s.verify_plain::<Error>(&empty, 0, |_| Ok(())).unwrap().frames, 1);
}

#[test]
fn verification_callback_cancels_and_enforces_expired_deadline() {
    let d = tempfile::tempdir().unwrap();
    let s = Store::open(d.path(), vec![], 0o600).unwrap(); // default 4 MiB frames
    let bytes = data(18, 8 << 20);
    let h = s.put(&bytes[..], 1).unwrap();
    let mut decoded = 0usize;
    let started = std::time::Instant::now();
    let result = s.verify_plain::<Error>(&h, bytes.len() as u64, |chunk| {
        decoded += chunk.len();
        if decoded >= 65536 { return Err(Error::Refused("cancelled".into())); }
        Ok(())
    });
    assert!(matches!(result, Err(Error::Refused(s)) if s == "cancelled"));
    assert_eq!(decoded, 65536);
    eprintln!("CANCEL decoded={decoded} elapsed_us={}", started.elapsed().as_micros());
    let deadline = std::time::Instant::now();
    assert!(matches!(s.verify_plain::<Error>(&h, bytes.len() as u64, |_| {
        if std::time::Instant::now() >= deadline { Err(Error::Refused("deadline".into())) } else { Ok(()) }
    }), Err(Error::Refused(_))));
}

#[test]
fn finite_valid_ranges_and_structural_negative_controls() {
    let temp = tempfile::tempdir().unwrap();
    let mut cases = 0;
    for frame in [1usize, 2, 7, 8, 31, 64] {
        for len in [0usize, 1, 2, 7, 8, 9, 33, 65] {
            let s = Store::open(temp.path().join(format!("{frame}-{len}")), vec![], 0o600).unwrap().with_frame_size(frame);
            let bytes: Vec<u8> = (0..len).map(|i| ((i*37+frame)%256) as u8).collect();
            let h = s.put(&bytes[..], 1).unwrap();
            for a in 0..=len { for b in a..=len {
                assert_eq!(s.read_range(&h, a as u64, (b-a) as u64).unwrap(), bytes[a..b]);
                cases += 1;
            }}
        }
    }
    assert_eq!(cases, 17712);
    let s = Store::open(temp.path().join("negative"), vec![], 0o600).unwrap().with_frame_size(8);
    let h = s.put(&b"0123456789abcdefghijklmn"[..], 1).unwrap();
    let path = file_of(&s, &h);
    let original = fs::read(&path).unwrap();
    let table = original.len() - 17 - 3*8;
    for entry in 0..3 { for value in [0u32, 7, 9, u32::MAX] {
        let mut bad = original.clone();
        bad[table+12+entry*8..table+16+entry*8].copy_from_slice(&value.to_le_bytes());
        fs::write(&path, bad).unwrap();
        assert!(s.read_range(&h, 16, 1).is_err());
        assert!(matches!(s.verify(&h), Ok(Verified::Corrupt(_))));
    }}
    for pos in [original.len()-5, table+4, table+8] {
        let mut bad = original.clone();
        bad[pos] = 0xff;
        fs::write(&path, bad).unwrap();
        assert!(s.read_range(&h, 16, 1).is_err());
    }
}

#[test]
fn larger_writer_frames_use_bounded_decoder_and_corrupt_large_extent_refuses() {
    let temp = tempfile::tempdir().unwrap();
    let s = Store::open(temp.path(), vec![], 0o600).unwrap().with_frame_size(9 << 20);
    let mut x = 91u64;
    let bytes: Vec<u8> = (0..10<<20).map(|_| {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17; x as u8
    }).collect();
    let h = s.put(&bytes[..], 1).unwrap();
    assert!(s.stored_size(&h).unwrap() > zstd::zstd_safe::compress_bound(DEFAULT_FRAME) as u64);
    assert_eq!(s.read_range(&h, (8<<20)+17, 31).unwrap(), bytes[(8<<20)+17..(8<<20)+48]);
    assert_eq!(s.verify_plain::<Error>(&h, bytes.len() as u64, |_| Ok(())).unwrap().decoded_bytes, bytes.len() as u64);
    let path = file_of(&s, &h);
    let mut bad = fs::read(&path).unwrap();
    let table = bad.len() - 17 - 2*8;
    bad[table+8..table+12].copy_from_slice(&u32::MAX.to_le_bytes());
    fs::write(&path, bad).unwrap();
    assert!(matches!(s.read_range(&h, 0, 1), Err(Error::Corrupt { .. })));
}

#[test]
fn frame_size_must_fit_the_written_seek_entry() {
    for frame in [0, usize::MAX] {
        let temp = tempfile::tempdir().unwrap();
        let s = Store::open(temp.path(), vec![], 0o600).unwrap().with_frame_size(frame);
        assert!(matches!(s.put(&b"bytes"[..], 1), Err(Error::Refused(_))));
        assert!(matches!(s.put_sealed(Cursor::new(b"bytes"), 1, &key("key", 7)), Err(Error::Refused(_))));
    }
}

#[test]
fn jointly_changed_frame_header_and_seek_length_cannot_displace_a_range() {
    let d = tempfile::tempdir().unwrap();
    let s = Store::open(d.path(), vec![], 0o600).unwrap().with_frame_size(8);
    let h = s.put(&b"ABCDEFGHijklmnopQRSTUVWX"[..], 1).unwrap();
    let path = file_of(&s, &h);
    let mut bytes = fs::read(&path).unwrap();
    assert_eq!(bytes[4], 0x24);
    assert_eq!(bytes[5], 8);
    let table = bytes.len() - 17 - 3 * 8;
    bytes[5] ^= 1;
    bytes[table+12] ^= 1;
    fs::write(path, bytes).unwrap();
    assert!(matches!(s.verify(&h), Ok(Verified::Corrupt(_))));
    assert!(matches!(s.read_range(&h, 9, 1), Err(Error::Corrupt { .. })));
    let reader = Reader::new(d.path(), vec![]);
    assert!(matches!(reader.read_range(&h, 9, 1), Err(Error::Corrupt { .. })));
    assert!(reader.slot().unwrap().is_none());
}

#[test]
fn one_verified_representation_reuses_bounded_frames_for_sequential_ranges() {
    let d = tempfile::tempdir().unwrap();
    let s = Store::open(d.path(), vec![], 0o600).unwrap();
    let bytes = data(99, 8<<20);
    let h = s.put(&bytes[..], 1).unwrap();
    let reader = Reader::new(d.path(), vec![]);
    let initial = reader.verify_plain::<Error>(&h, bytes.len() as u64, |_| Ok(())).unwrap();
    for offset in (0..bytes.len()).step_by(65536) {
        assert_eq!(reader.read_range(&h, offset as u64, 65536).unwrap(), bytes[offset..offset+65536]);
    }
    let retained = reader.slot().unwrap();
    let retained = retained.as_ref().unwrap();
    let after = retained.stats;
    assert_eq!(initial.frames, 2);
    assert_eq!(after.frames - initial.frames, 2);
    assert_eq!(after.decoded_bytes - initial.decoded_bytes, bytes.len() as u64);
    assert_eq!(after.compressed_bytes - initial.compressed_bytes, initial.compressed_bytes);
    assert_eq!(after.metadata_bytes, initial.metadata_bytes);
    let Some(CachedFrame { bytes: cached, .. }) = &retained.current else { panic!("frame not retained") };
    assert_eq!(cached.len(), DEFAULT_FRAME);
    eprintln!("SCOPED_RANGES validation_decoded={} validation_compressed={} metadata={} ranges=128 subsequent_decoded={} subsequent_compressed={} subsequent_metadata={} frame_cache_bytes={}", initial.decoded_bytes, initial.compressed_bytes, initial.metadata_bytes, after.decoded_bytes-initial.decoded_bytes, after.compressed_bytes-initial.compressed_bytes, after.metadata_bytes-initial.metadata_bytes, cached.len());
}

#[test]
fn scoped_reader_refuses_mutation_and_cancellation_without_publishing_a_cache() {
    let d = tempfile::tempdir().unwrap();
    let s = Store::open(d.path(), vec![], 0o600).unwrap();
    let bytes = data(21, 5<<20);
    let h = s.put(&bytes[..], 1).unwrap();
    let reader = Reader::new(d.path(), vec![]);
    reader.verify_plain::<Error>(&h, bytes.len() as u64, |_| Ok(())).unwrap();
    let path = file_of(&s, &h);
    // Replace the pathname with another valid object after verification. Never
    // follow the new name; the held descriptor either serves its original bytes
    // or conservatively refuses a filesystem generation change from unlink.
    let other = s.put(&b"different object"[..], 1).unwrap();
    let replacement = d.path().join("replacement");
    fs::copy(file_of(&s, &other), &replacement).unwrap();
    fs::rename(replacement, &path).unwrap();
    match reader.read_range(&h, 0, 16) {
        Ok(actual) => assert_eq!(actual, bytes[..16]),
        Err(Error::Corrupt { .. }) => {},
        other => panic!("unexpected replacement outcome: {other:?}"),
    }
    if let Ok(actual) = reader.read_range(&h, 0, 1) { assert_eq!(actual, bytes[..1]); }
    let h = s.put(&b"fresh object"[..], 1).unwrap();
    let mut delivered = 0;
    assert!(reader.verify_plain::<Error>(&h, 12, |chunk| {
        delivered += chunk.len();
        Err(Error::Refused("cancelled".into()))
    }).is_err());
    assert_eq!(delivered, 0);
    assert!(reader.slot().unwrap().is_none());
}

#[test]
fn reader_remains_send_and_sync_and_warms_one_frame_without_extra_decoding() {
    fn shared<T: Send + Sync>() {}
    shared::<Reader>();
    let d = tempfile::tempdir().unwrap();
    let s = Store::open(d.path(), vec![], 0o600).unwrap();
    let bytes = data(5, 200000);
    let h = s.put(&bytes[..], 1).unwrap();
    let reader = Reader::new(d.path(), vec![]);
    let initial = reader.verify_plain::<Error>(&h, bytes.len() as u64, |_| Ok(())).unwrap();
    assert_eq!(reader.read_range(&h, 100, 65536).unwrap(), bytes[100..65636]);
    assert_eq!(reader.slot().unwrap().as_ref().unwrap().stats.decoded_bytes, initial.decoded_bytes);
    assert!(reader.read_range_checked::<Error>(&h, Some(1), 0, 1, || Ok(())).is_err());
    assert!(reader.slot().unwrap().is_none());
    reader.verify_plain::<Error>(&h, bytes.len() as u64, |_| Ok(())).unwrap();
    fs::write(file_of(&s, &h), b"in-place corruption").unwrap();
    assert!(matches!(reader.read_range(&h, 0, 1), Err(Error::Corrupt { .. })));
    assert!(reader.slot().unwrap().is_none());
}

#[test]
fn assembled_extents_refuse_arithmetic_overflow() {
    let (_d, s) = store();
    let h = s.put(&b"bytes"[..], 1).unwrap();
    assert!(matches!(s.assemble(&[Extent { blob: h, offset: 0, len: u64::MAX }, Extent { blob: h, offset: 0, len: 1 }]), Err(Error::Refused(_))));
    let mut assembled = s.assemble(&[Extent { blob: h, offset: u64::MAX, len: 2 }]).unwrap();
    assembled.seek(SeekFrom::Start(1)).unwrap();
    assert!(assembled.read(&mut [0u8; 1]).is_err());
    let mut assembled = s.assemble(&[Extent { blob: h, offset: 0, len: u64::MAX }]).unwrap();
    assert!(assembled.seek(SeekFrom::End(1)).is_err());
}
