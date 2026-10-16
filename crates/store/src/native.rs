//! Audited native boundary: handles stay owned and valid for each synchronous call.
#![allow(unsafe_code)]
use rusqlite::Connection;
use std::{fs::File, io, os::fd::AsRawFd, path::Path};

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

/// Observe SQLite's actual page-cache occupancy in resource discriminators.
#[cfg(test)]
pub(crate) fn cache_bytes(conn: &Connection) -> rusqlite::Result<i32> {
    let mut current = 0;
    let mut unused_highwater = 0;
    // Borrowed connection and output cells stay valid for this synchronous
    // read-only status call. No SQLite handle or pointer escapes.
    let code = unsafe {
        rusqlite::ffi::sqlite3_db_status(
            conn.handle(),
            rusqlite::ffi::SQLITE_DBSTATUS_CACHE_USED,
            &mut current,
            &mut unused_highwater,
            0,
        )
    };
    if code == rusqlite::ffi::SQLITE_OK {
        Ok(current)
    } else {
        Err(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(code),
            None,
        ))
    }
}

/// Linux renameat2 is the publication boundary: an existing destination is never replaced.
pub(crate) fn publish(source: &Path, destination: &Path) -> io::Result<()> {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let source = CString::new(source.as_os_str().as_bytes())?;
    let destination = CString::new(destination.as_os_str().as_bytes())?;
    // Both strings live through the synchronous syscall. The flag is atomic,
    // unlike a preceding existence check followed by an ordinary rename.
    if unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            destination.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    } == 0
    {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

pub(crate) struct Progress<'a>(&'a Connection);
impl Drop for Progress<'_> {
    fn drop(&mut self) {
        // Remove the callback before its borrowed cancellation flag can expire.
        unsafe {
            rusqlite::ffi::sqlite3_progress_handler(self.0.handle(), 0, None, std::ptr::null_mut());
        }
    }
}
pub(crate) fn cancellation<'a>(
    conn: &'a Connection,
    cancelled: &'a std::sync::atomic::AtomicBool,
) -> Progress<'a> {
    unsafe extern "C" fn check(value: *mut std::ffi::c_void) -> i32 {
        // The guard borrows the flag for the whole synchronous SQLite operation.
        let cancelled = unsafe { &*value.cast::<std::sync::atomic::AtomicBool>() };
        i32::from(cancelled.load(std::sync::atomic::Ordering::Acquire))
    }
    unsafe {
        rusqlite::ffi::sqlite3_progress_handler(
            conn.handle(),
            1000,
            Some(check),
            std::ptr::from_ref(cancelled).cast_mut().cast(),
        );
    }
    Progress(conn)
}

pub(crate) fn no_cow(path: &Path) -> io::Result<()> {
    let file = File::open(path)?;
    let mut fs: libc::statfs = unsafe { std::mem::zeroed() };
    // fstatfs writes exactly the initialized statfs object and borrows a live descriptor.
    if unsafe { libc::fstatfs(file.as_raw_fd(), &mut fs) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if fs.f_type != 0x9123683e {
        return Ok(());
    } // BTRFS_SUPER_MAGIC
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

// SQLite calls are synchronous on the owning thread. Capture only the current
// operation; unrelated errors must never inherit a previous checksum location.
#[derive(Debug)]
pub(crate) struct ChecksumFault {
    pub file: std::path::PathBuf,
    pub page: u64,
    pub offset: u64,
    pub page_size: u32,
}
thread_local! {
    static CHECKSUM_FAULT: std::cell::RefCell<Option<ChecksumFault>> = const { std::cell::RefCell::new(None) };
}
pub(crate) fn take_checksum_fault() -> Option<ChecksumFault> {
    CHECKSUM_FAULT.with(|slot| slot.borrow_mut().take())
}
#[unsafe(no_mangle)]
unsafe extern "C" fn polylogue_checksum_fault(
    filename: *const std::ffi::c_char,
    offset: i64,
    size: i32,
) {
    use std::os::unix::ffi::OsStringExt;
    if filename.is_null()
        || offset < 0
        || size < 512
        || !(size as u32).is_power_of_two()
        || offset % i64::from(size) != 0
    {
        return;
    }
    // The VFS owns a NUL-terminated filename for the complete synchronous call.
    // Copy before returning; no borrowed C pointer escapes.
    let file = unsafe { std::ffi::CStr::from_ptr(filename) }
        .to_bytes()
        .to_vec();
    let fault = ChecksumFault {
        file: std::ffi::OsString::from_vec(file).into(),
        page: offset as u64 / size as u64 + 1,
        offset: offset as u64,
        page_size: size as u32,
    };
    CHECKSUM_FAULT.with(|slot| {
        *slot.borrow_mut() = Some(fault);
    });
}
