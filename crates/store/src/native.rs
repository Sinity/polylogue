//! Audited native boundary: handles stay owned and valid for each synchronous call.
#![allow(unsafe_code)]
use std::{fs::File, io, os::fd::AsRawFd, path::Path};
use rusqlite::Connection;

unsafe extern "C" {
    fn sqlite3_register_cksumvfs(unused: *const std::ffi::c_char) -> i32;
    fn polylogue_reserve_bytes(db: *mut rusqlite::ffi::sqlite3) -> i32;
}

pub(crate) fn register() -> i32 {
    // Static extension takes an ignored nullable pointer and registers process-lived VFS storage.
    unsafe { sqlite3_register_cksumvfs(std::ptr::null()) }
}
pub(crate) fn reserve(conn: &Connection) -> i32 {
    // Connection cannot close or be used concurrently during this call; no handle escapes.
    unsafe { polylogue_reserve_bytes(conn.handle()) }
}

pub(crate) fn no_cow(path: &Path) -> io::Result<()> {
    let file = File::open(path)?;
    let mut fs: libc::statfs = unsafe { std::mem::zeroed() };
    // fstatfs writes exactly the initialized statfs object and borrows a live descriptor.
    if unsafe { libc::fstatfs(file.as_raw_fd(), &mut fs) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if fs.f_type != 0x9123683e { return Ok(()); } // BTRFS_SUPER_MAGIC
    let mut flags: libc::c_long = 0;
    // Linux FS_IOC_GETFLAGS/SETFLAGS expect a pointer to a long. Directory stays open.
    if unsafe { libc::ioctl(file.as_raw_fd(), 0x80086601_u64, &mut flags) } != 0 {
        return Err(io::Error::last_os_error());
    }
    flags |= 0x00800000; // FS_NOCOW_FL, before any child exists.
    if unsafe { libc::ioctl(file.as_raw_fd(), 0x40086602_u64, &flags) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
