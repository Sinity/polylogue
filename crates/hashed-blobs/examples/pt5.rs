//! PT5: hashed-blobs with Sinex volumes. Stores the files listed in a file (one path per line) into
//! a throwaway root at a zstd level, then reports throughput, CPU per GiB, the stored/input ratio,
//! random range-read latency on the largest blob and a streaming verify of it. Prints counts only.
//!
//! cargo run --release --example pt5 -- <root> <file list> <level> [single|perfile|syncfs]
//! (single: one durable put per file; perfile/syncfs: batches of 1,000 files, `Durability`)

use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

use hashed_blobs::{Durability, Hash, Store, Verified};

fn cpu_s() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap();
    let f: Vec<&str> = s.rsplit(')').next().unwrap().split_whitespace().collect();
    (f[11].parse::<f64>().unwrap() + f[12].parse::<f64>().unwrap()) / 100.0 // utime + stime, in clock ticks of 1/100 s
}

fn dir_bytes(p: &std::path::Path) -> u64 {
    let mut n = 0;
    for e in std::fs::read_dir(p).unwrap().flatten() {
        let m = e.metadata().unwrap();
        n += if m.is_dir() { dir_bytes(&e.path()) } else { m.len() };
    }
    n
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (root, list, level) = (&a[1], &a[2], a[3].parse::<i32>().unwrap());
    let files: Vec<String> = std::fs::read_to_string(list).unwrap().lines().filter(|l| !l.is_empty()).map(String::from).collect();
    let store = Store::open(root, vec![], 0o600).unwrap();
    let (t0, c0) = (Instant::now(), cpu_s());
    let mut input = 0u64;
    let mut biggest: Option<(u64, Hash)> = None;
    let mode = a.get(4).map(String::as_str).unwrap_or("single");
    for chunk in files.chunks(if mode == "single" { 1 } else { 1000 }) {
        let mut batch = match mode {
            "perfile" => Some(store.batch(Durability::PerFile)),
            "syncfs" => Some(store.batch(Durability::Filesystem)),
            _ => None,
        };
        for p in chunk {
            let n = std::fs::metadata(p).unwrap().len();
            input += n;
            let r = BufReader::with_capacity(1 << 20, File::open(p).unwrap());
            let h = match batch.as_mut() {
                Some(b) => b.put(r, level).unwrap(),
                None => store.put(r, level).unwrap(),
            };
            if biggest.is_none_or(|(b, _)| n > b) {
                biggest = Some((n, h));
            }
        }
        if let Some(b) = batch {
            b.commit().unwrap();
        }
    }
    let (wall, cpu) = (t0.elapsed().as_secs_f64(), cpu_s() - c0);
    let stored = dir_bytes(&std::path::Path::new(root).join("blobs"));
    let gib = input as f64 / (1u64 << 30) as f64;
    println!("files {}  mode {}  input {:.2} GiB  stored {:.2} GiB ({:.1}%)  level {}", files.len(), mode, gib, stored as f64 / (1u64 << 30) as f64, 100.0 * stored as f64 / input as f64, level);
    println!("put: {:.1} s wall, {:.0} MiB/s, {:.1} CPU s per GiB", wall, input as f64 / wall / (1u64 << 20) as f64, cpu / gib);

    let (size, h) = biggest.unwrap();
    let mut lat = Vec::new();
    let mut x: u64 = 0x9E3779B97F4A7C15;
    for _ in 0..200 {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        let off = x % size.saturating_sub(65536).max(1);
        let t = Instant::now();
        let b = store.read_range(&h, off, 65536.min(size)).unwrap();
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(b.len() as u64, 65536.min(size));
    }
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("range reads of 64 KiB on the largest blob ({:.2} GiB): p50 {:.1} ms, p99 {:.1} ms, max {:.1} ms",
             size as f64 / (1u64 << 30) as f64, lat[100], lat[198], lat[199]);
    let (t1, c1) = (Instant::now(), cpu_s());
    let v = store.verify(&h).unwrap();
    println!("verify of the largest blob: {:?} in {:.1} s ({:.0} MiB/s, {:.1} CPU s)", v, t1.elapsed().as_secs_f64(),
             size as f64 / t1.elapsed().as_secs_f64() / (1u64 << 20) as f64, cpu_s() - c1);
    assert_eq!(v, Verified::Intact);
}
