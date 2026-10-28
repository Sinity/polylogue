//! A delegating test VFS pauses where SQLite starts a WAL read, without replacing SQL.
#![allow(unsafe_code)]
use rusqlite::ffi;
use std::collections::BTreeMap;
use std::ffi::{c_char, c_int};
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::mpsc::{Receiver, Sender};

struct Methods {
    original: *const ffi::sqlite3_io_methods,
    copied: ffi::sqlite3_io_methods,
}
static BASE: AtomicPtr<ffi::sqlite3_vfs> = AtomicPtr::new(ptr::null_mut());
static FILES: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());
struct Hook {
    database: usize,
    reached: Sender<()>,
    proceed: Receiver<()>,
}
static HOOK: Mutex<Option<Hook>> = Mutex::new(None);

pub(super) struct Registration {
    vfs: Box<ffi::sqlite3_vfs>,
}
impl Registration {
    pub(super) fn install() -> Self {
        // The copied VFS and name outlive every connection created in this test.
        unsafe {
            let base = ffi::sqlite3_vfs_find(ptr::null());
            assert!(!base.is_null());
            BASE.store(base, Ordering::SeqCst);
            let mut vfs = Box::new(ptr::read(base));
            vfs.zName = c"polylogue-read-start-test".as_ptr();
            vfs.pNext = ptr::null_mut();
            vfs.xOpen = Some(open);
            assert_eq!(ffi::sqlite3_vfs_register(&mut *vfs, 1), ffi::SQLITE_OK);
            Self { vfs }
        }
    }
    pub(super) fn pause(
        &self,
        db: &rusqlite::Connection,
        reached: Sender<()>,
        proceed: Receiver<()>,
    ) {
        // This address is observed only while the reader owns its live connection.
        let database = unsafe { db.handle() } as usize;
        *HOOK.lock().unwrap() = Some(Hook {
            database,
            reached,
            proceed,
        });
    }
}
impl Drop for Registration {
    fn drop(&mut self) {
        *HOOK.lock().unwrap() = None;
        assert!(
            FILES.lock().unwrap().is_empty(),
            "a test connection outlived its VFS"
        );
        unsafe {
            assert_eq!(ffi::sqlite3_vfs_unregister(&mut *self.vfs), ffi::SQLITE_OK);
            assert_eq!(
                ffi::sqlite3_vfs_register(BASE.load(Ordering::SeqCst), 1),
                ffi::SQLITE_OK
            );
        }
    }
}
unsafe extern "C" fn open(
    _: *mut ffi::sqlite3_vfs,
    name: *const c_char,
    file: *mut ffi::sqlite3_file,
    flags: c_int,
    out: *mut c_int,
) -> c_int {
    unsafe {
        let base = BASE.load(Ordering::SeqCst);
        let rc = ((*base).xOpen.unwrap())(base, name, file, flags, out);
        if rc != ffi::SQLITE_OK || (*file).pMethods.is_null() {
            return rc;
        }
        let original = (*file).pMethods;
        if (*original).iVersion < 2 || (*original).xShmLock.is_none() {
            return rc;
        }
        let mut methods = Box::new(Methods {
            original,
            copied: ptr::read(original),
        });
        methods.copied.xShmLock = Some(shm_lock);
        methods.copied.xClose = Some(close);
        (*file).pMethods = &methods.copied;
        FILES
            .lock()
            .unwrap()
            .insert(file as usize, Box::into_raw(methods) as usize);
        rc
    }
}
unsafe extern "C" fn close(file: *mut ffi::sqlite3_file) -> c_int {
    unsafe {
        let address = FILES.lock().unwrap().remove(&(file as usize)).unwrap();
        let methods = Box::from_raw(address as *mut Methods);
        (*file).pMethods = methods.original;
        ((*methods.original).xClose.unwrap())(file)
    }
}
unsafe extern "C" fn shm_lock(
    file: *mut ffi::sqlite3_file,
    offset: c_int,
    count: c_int,
    flags: c_int,
) -> c_int {
    unsafe {
        let original = {
            let files = FILES.lock().unwrap();
            let methods = &*(files[&(file as usize)] as *const Methods);
            (*methods.original).xShmLock.unwrap()
        };
        let hook = {
            let mut hook = HOOK.lock().unwrap();
            let at_gap = hook.as_ref().is_some_and(|hook| {
                let db = hook.database as *mut ffi::sqlite3;
                ffi::sqlite3_txn_state(db, c"main".as_ptr()) == ffi::SQLITE_TXN_READ
                    && ffi::sqlite3_txn_state(db, c"a".as_ptr()) == ffi::SQLITE_TXN_NONE
            });
            if at_gap { hook.take() } else { None }
        };
        if let Some(hook) = hook
            && (hook.reached.send(()).is_err() || hook.proceed.recv().is_err())
        {
            return ffi::SQLITE_IOERR;
        }
        original(file, offset, count, flags)
    }
}
