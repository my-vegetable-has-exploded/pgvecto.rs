use crate::prelude::*;
use crate::utils::cells::PgRefCell;
use service::prelude::*;
use std::ops::DerefMut;

static DIRTY: PgRefCell<Vec<Handle>> = unsafe { PgRefCell::new(Vec::new()) };

pub fn callback_dirty(handle: Handle) {
    DIRTY.borrow_mut().push(handle);
}

pub fn commit() {
    let pending_deletes = pending_deletes(true);
    let pending_dirty = std::mem::take(DIRTY.borrow_mut().deref_mut());
    if pending_deletes.is_empty() && pending_dirty.is_empty() {
        return;
    }
    let mut rpc = crate::ipc::client::borrow_mut();
    rpc.commit(pending_deletes, pending_dirty);
}

pub fn abort() {
    let pending_deletes = pending_deletes(false);
    if pending_deletes.is_empty() {
        return;
    }
    let mut rpc = crate::ipc::client::borrow_mut();
    rpc.abort(pending_deletes);
}

#[cfg(any(feature = "pg12", feature = "pg13", feature = "pg14", feature = "pg15"))]
fn pending_deletes(for_commit: bool) -> Vec<Handle> {
    let mut ptr: *mut pgrx::pg_sys::RelFileNode = std::ptr::null_mut();
    let n = unsafe { pgrx::pg_sys::smgrGetPendingDeletes(for_commit, &mut ptr as *mut _) };
    if n > 0 {
        let nodes = unsafe { std::slice::from_raw_parts(ptr, n as usize) };
        nodes
            .iter()
            .map(|node| Handle::from_sys(node.relNode))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    }
}

#[cfg(feature = "pg16")]
fn pending_deletes(for_commit: bool) -> Vec<Handle> {
    let mut ptr: *mut pgrx::pg_sys::RelFileLocator = std::ptr::null_mut();
    let n = unsafe { pgrx::pg_sys::smgrGetPendingDeletes(for_commit, &mut ptr as *mut _) };
    if n > 0 {
        let nodes = unsafe { std::slice::from_raw_parts(ptr, n as usize) };
        nodes
            .iter()
            .map(|node| Handle::from_sys(node.relNumber))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    }
}
