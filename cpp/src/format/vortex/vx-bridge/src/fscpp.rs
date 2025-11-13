
use std::ffi::c_void;
use std::sync::Arc;
use std::io::Write;
use std::marker::PhantomData;

use async_compat::Compat;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};

use vortex::buffer::ByteBufferMut;
use vortex::error::{VortexError, VortexResult};
use vortex::io::runtime::Handle;
use vortex::io::file::{CoalesceWindow, IntoReadSource, ReadSource, ReadSourceRef, IoRequest};

unsafe extern "C" {
    unsafe fn fscpp_create(config: *mut std::ffi::c_void) -> *mut std::ffi::c_void;

    // C-ABI: write data from pointer + size, return number of bytes written or negative error code
    unsafe fn fscpp_open_writer(fs :*mut std::ffi::c_void, path: *const u8, path_len: usize) 
        -> *mut std::ffi::c_void;
    unsafe fn fscpp_write(
        writer: *mut std::ffi::c_void,
        data: *const u8,
        size: usize,
    ) -> isize;
    unsafe fn fscpp_flush(writer: *mut std::ffi::c_void);
    unsafe fn fscpp_close(writer: *mut std::ffi::c_void);

    // C-ABI: pass path as pointer + len, avoid Rust String across FFI
    unsafe fn fscpp_head_object(
        ptr: *mut std::ffi::c_void,
        path_ptr: *const u8,
        path_len: usize,
    ) -> u64;

    // C-ABI: pass path pointer/len and explicit range bounds to avoid non-FFI-safe Rust types
    unsafe fn fscpp_get_range(
        ptr: *mut std::ffi::c_void,
        path_ptr: *const u8,
        path_len: usize,
        start: u64,
        out_buf: *mut u8,
        len: usize,
    );
    unsafe fn fscpp_release(ptr: *mut std::ffi::c_void);
}


struct ThreadSafePtr<T> {
    ptr: *mut c_void,
    _marker: PhantomData<T>,
}

unsafe impl<T> Send for ThreadSafePtr<T> {}
unsafe impl<T> Sync for ThreadSafePtr<T> {}

impl<T> ThreadSafePtr<T> {
    fn new(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }
    
    // Add methods to safely access the pointer
    fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl<T> Clone for ThreadSafePtr<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

pub struct ObjectStoreWriterCpp {
    inner: ThreadSafePtr<std::ffi::c_void>,
}

impl ObjectStoreWriterCpp {
    pub fn new(config: *mut std::ffi::c_void) -> Result<Self, VortexError> {
        let inner = ThreadSafePtr::new(unsafe {
            fscpp_create(config) 
        });
        // TBD: error handling
        Ok(Self { inner })
    }
}

impl Drop for ObjectStoreWriterCpp {
    fn drop(&mut self) {
        unsafe { 
            fscpp_close(self.inner.as_ptr());
            fscpp_release(self.inner.as_ptr());
        };
    }
}

impl Write for ObjectStoreWriterCpp {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = unsafe { fscpp_write(self.inner.as_ptr(), buf.as_ptr(), buf.len()) };
        if written < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("failed to write data, error code: {}", written),
            ));
        }
        Ok(written as usize)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        unsafe { fscpp_flush(self.inner.as_ptr()) };
        Ok(())
    }
}

const COALESCING_WINDOW: CoalesceWindow = CoalesceWindow {
    distance: 1024 * 1024,      // 1 MB
    max_size: 16 * 1024 * 1024, // 16 MB
};
const CONCURRENCY: usize = 192;

pub struct ObjectStoreReadSourceCpp {
    inner: ThreadSafePtr<std::ffi::c_void>,
    path: String,
    uri: Arc<str>,
    coalesce_window: Option<CoalesceWindow>,
}

impl ObjectStoreReadSourceCpp {
    pub fn new(config: *mut std::ffi::c_void, path: &str) -> Result<Self, VortexError> {
        let uri = Arc::from(path.to_string());
        let inner = ThreadSafePtr::new(unsafe { fscpp_create(config) });
        // TBD: error handling
        Ok(Self {
            inner: inner,
            path: path.to_string(),
            uri: uri,
            coalesce_window: Some(COALESCING_WINDOW),
        })
    }
}

struct ObjectStoreIoSourceCpp {
    io: ObjectStoreReadSourceCpp,
    handle: Handle,
}

impl ReadSource for ObjectStoreIoSourceCpp {
    fn uri(&self) -> &Arc<str> {
        &self.io.uri
    }

    fn coalesce_window(&self) -> Option<CoalesceWindow> {
        self.io.coalesce_window
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        // move owned values into the async block so the future is 'static
        let inner = self.io.inner.clone();
        let path = self.io.path.clone();
        let handle = self.handle.clone();
        Compat::new(async move {
            // Pass path as bytes to FFI (no allocation across FFI boundaries)
            let path_bytes = path.into_bytes();
            let task = handle.spawn_blocking(move || unsafe {
                fscpp_head_object(inner.as_ptr(), path_bytes.as_ptr(), path_bytes.len())
            });
            let size = Compat::new(task).await;
            Ok(size)
        })
        .boxed()
    }

    fn drive_send(
        self: Arc<Self>,
        requests: BoxStream<'static, IoRequest>,
    ) -> BoxFuture<'static, ()> {
        let self2 = self.clone();
        requests
        .map(move |req| {
            let store = self.io.inner.clone();
            let path = self.io.path.clone();

            let len = req.len();
            let range = req.range();
            let alignment = req.alignment();

            // Offload sync FFI to blocking pool, copy into Rust-owned Vec<u8>
            let blocking = self.handle.spawn_blocking(move || {
                let path_bytes = path.into_bytes();
                let mut owned = Vec::with_capacity(len);

                unsafe {
                    fscpp_get_range(
                        store.as_ptr(),
                        path_bytes.as_ptr(),
                        path_bytes.len(),
                        range.start,
                        owned.as_mut_ptr(),
                        len,
                    );
                };
                owned
            });

            let fut = async move {
                let bytes: Vec<u8> = Compat::new(blocking).await;
                let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
                buffer.extend_from_slice(&bytes);
                Ok(buffer.freeze())
            };

            async move { req.resolve(Compat::new(fut).await) }
        })
        .map(move |f| self2.handle.spawn(f))
        .buffer_unordered(CONCURRENCY)
        .collect::<()>()
        .boxed()
    }
}


impl IntoReadSource for ObjectStoreReadSourceCpp {
    fn into_read_source(self, handle: Handle) -> VortexResult<ReadSourceRef> {
        Ok(Arc::new(ObjectStoreIoSourceCpp { io: self, handle }))
    }
}
