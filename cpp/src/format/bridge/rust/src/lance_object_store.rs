// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright Zilliz

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::future::Future;
use std::io;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::path::{Component, Path as FsPath, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, Weak};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use cxx::SharedPtr;
use futures::StreamExt;
#[cfg(feature = "s3-crt-async")]
use futures::channel::oneshot;
use futures::stream::{self, BoxStream};
use lru::LruCache;
use object_store::path::Path as ObjectPath;
use object_store::{
    Attributes, CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload,
    ObjectMeta, PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions,
};
use url::Url;

use lance::io::ObjectStoreParams;
use lance::session::Session;
use lance::{Error as LanceError, Result};
use lance_io::object_store::throttle::{AimdThrottleConfig, AimdThrottledStore};
use lance_io::object_store::{
    DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_DOWNLOAD_RETRY_COUNT, ObjectStore, ObjectStoreProvider,
    ObjectStoreRegistry, StorageOptionsAccessor,
};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

use crate::TOKIO_RT;
use crate::filesystem_c::{
    LoonFFIError, LoonFileInfoList, ReaderHandle, check_loon_ffi_result, loon_errcode_arrow,
    loon_errcode_aws_access_denied, loon_errcode_aws_not_found,
    loon_errcode_aws_precondition_failed, loon_errcode_file_not_found, loon_errcode_not_support,
    loon_errcode_transient_throttling, loon_filesystem_free_file_info_list,
    loon_filesystem_get_object_info, loon_filesystem_list_dir, loon_filesystem_open_reader,
    loon_filesystem_reader_close, loon_filesystem_reader_destroy, loon_filesystem_reader_readat,
    reader_supports_async,
};
#[cfg(feature = "s3-crt-async")]
use crate::filesystem_c::{LoonFFIResult, loon_filesystem_reader_readat_async};
use crate::lance_ffi::FileSystemWrapper;

#[cfg(feature = "s3-crt-async")]
struct ObjectStoreAsyncReadCallbackState {
    sender: Option<oneshot::Sender<object_store::Result<Bytes>>>,
    reader: Arc<ReaderHandle>,
    buffer: BytesMut,
    location: ObjectPath,
    expected_len: usize,
}

#[cfg(feature = "s3-crt-async")]
unsafe extern "C" fn object_store_async_read_callback(
    user_data: *mut c_void,
    mut result: LoonFFIResult,
    bytes_read: u64,
) {
    if user_data.is_null() {
        let _ = check_loon_ffi_result(&mut result, "async object read failed");
        return;
    }

    let ObjectStoreAsyncReadCallbackState {
        mut sender,
        reader: _reader,
        mut buffer,
        location,
        expected_len,
    } = *unsafe { Box::from_raw(user_data.cast::<ObjectStoreAsyncReadCallbackState>()) };

    let read_result = match check_loon_ffi_result(&mut result, "async object read failed") {
        Err(error) => Err(into_object_store_error(error, &location)),
        Ok(()) if bytes_read != expected_len as u64 => Err(object_store::Error::Generic {
            store: "filesystem_c",
            source: Box::new(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("async object read returned {bytes_read} bytes, expected {expected_len}"),
            )),
        }),
        Ok(()) => {
            unsafe { buffer.set_len(expected_len) };
            Ok(buffer.freeze())
        }
    };

    if let Some(sender) = sender.take() {
        let _ = sender.send(read_result);
    }
}

#[cfg(feature = "s3-crt-async")]
async fn read_object_store_async_via_ffi(
    reader: Arc<ReaderHandle>,
    location: ObjectPath,
    range: Range<u64>,
) -> object_store::Result<Bytes> {
    let length = range.end - range.start;
    let expected_len = usize::try_from(length).map_err(|source| object_store::Error::Generic {
        store: "filesystem_c",
        source: Box::new(source),
    })?;
    let mut buffer = BytesMut::with_capacity(expected_len);
    let out_data = buffer.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    let (sender, receiver) = oneshot::channel();
    let state = Box::new(ObjectStoreAsyncReadCallbackState {
        sender: Some(sender),
        reader: reader.clone(),
        buffer,
        location: location.clone(),
        expected_len,
    });
    let state_ptr = Box::into_raw(state);

    {
        let mut result = unsafe {
            loon_filesystem_reader_readat_async(
                reader.as_ptr(),
                range.start,
                length,
                out_data,
                object_store_async_read_callback,
                state_ptr.cast::<c_void>(),
            )
        };
        if result.err_code != 0 {
            unsafe { drop(Box::from_raw(state_ptr)) };
            check_loon_ffi_result(&mut result, "submit async object read")
                .map_err(|error| into_object_store_error(error, &location))?;
        }
    }
    drop(reader);

    receiver.await.map_err(|_| object_store::Error::Generic {
        store: "filesystem_c",
        source: Box::new(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "async object read completion channel closed",
        )),
    })?
}

#[derive(Clone, Debug)]
enum FFIPathMapper {
    Remote { scheme: String, authority: String },
    Local { normalized_root: PathBuf },
}

impl FFIPathMapper {
    fn remote(scheme: impl Into<String>, authority: impl Into<String>) -> Self {
        Self::Remote {
            scheme: scheme.into(),
            authority: authority.into(),
        }
    }

    fn local(root: impl AsRef<FsPath>) -> Self {
        Self::Local {
            normalized_root: root.as_ref().to_path_buf(),
        }
    }

    fn from_url(&self, url: &url::Url) -> object_store::Result<ObjectPath> {
        match self {
            Self::Remote { scheme, authority } => {
                if url.scheme() != scheme || url.host_str() != Some(authority.as_str()) {
                    return Err(object_store::Error::Generic {
                        store: "filesystem_c",
                        source: Box::new(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "URI storage identity does not match the bound filesystem",
                        )),
                    });
                }
                ObjectPath::from_url_path(url.path()).map_err(Into::into)
            }
            Self::Local { .. } => {
                let path = url
                    .to_file_path()
                    .map_err(|_| object_store::Error::Generic {
                        store: "filesystem_c",
                        source: Box::new(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "local dataset URI is not a valid file path",
                        )),
                    })?;
                self.from_local_path(path)
            }
        }
    }

    fn from_local_path(&self, path: impl AsRef<FsPath>) -> object_store::Result<ObjectPath> {
        let Self::Local { normalized_root } = self else {
            return Err(object_store::Error::Generic {
                store: "filesystem_c",
                source: Box::new(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "local path used with a remote filesystem",
                )),
            });
        };
        let path = path.as_ref();
        if !path.is_absolute()
            || path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
        {
            return Err(object_store::Error::Generic {
                store: "filesystem_c",
                source: Box::new(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "local dataset path must be absolute and must not contain '..'",
                )),
            });
        }

        let relative =
            path.strip_prefix(normalized_root)
                .map_err(|_| object_store::Error::Generic {
                    store: "filesystem_c",
                    source: Box::new(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "local dataset path is outside the bound filesystem root",
                    )),
                })?;
        let relative = relative
            .to_str()
            .ok_or_else(|| object_store::Error::Generic {
                store: "filesystem_c",
                source: Box::new(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "local dataset path is not valid UTF-8",
                )),
            })?
            .replace(std::path::MAIN_SEPARATOR, "/");
        ObjectPath::parse(relative).map_err(Into::into)
    }

    fn to_ffi_path(&self, location: &ObjectPath) -> String {
        location.to_string()
    }

    fn from_ffi_path(&self, path: &str) -> object_store::Result<ObjectPath> {
        ObjectPath::parse(path).map_err(Into::into)
    }
}

const FFI_OBJECT_META_CACHE_CAPACITY: usize = 256;
type MetaCell = Arc<tokio::sync::OnceCell<ObjectMeta>>;

#[derive(Clone)]
struct MetadataCache {
    entries: Arc<Mutex<LruCache<ObjectPath, MetaCell>>>,
}

impl MetadataCache {
    fn new() -> Self {
        // LruCache::new(capacity) preallocates the full hash table. Keep the
        // 256-entry limit while avoiding that fixed cost for every opened dataset.
        let mut entries = LruCache::new(NonZeroUsize::new(1).unwrap());
        entries.resize(NonZeroUsize::new(FFI_OBJECT_META_CACHE_CAPACITY).unwrap());
        Self {
            entries: Arc::new(Mutex::new(entries)),
        }
    }

    async fn get_or_load<F, Fut>(
        &self,
        location: ObjectPath,
        loader: F,
    ) -> object_store::Result<ObjectMeta>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = object_store::Result<ObjectMeta>>,
    {
        let cell = {
            let mut entries = self.entries.lock().unwrap();
            if let Some(cell) = entries.get(&location) {
                cell.clone()
            } else {
                let cell = Arc::new(tokio::sync::OnceCell::new());
                entries.put(location, cell.clone());
                cell
            }
        };

        cell.get_or_try_init(loader).await.cloned()
    }

    fn prime(&self, meta: ObjectMeta) {
        let cell = {
            let mut entries = self.entries.lock().unwrap();
            if let Some(cell) = entries.get(&meta.location) {
                cell.clone()
            } else {
                let cell = Arc::new(tokio::sync::OnceCell::new());
                entries.put(meta.location.clone(), cell.clone());
                cell
            }
        };
        let _ = cell.set(meta);
    }
}

struct FileInfoListGuard(LoonFileInfoList);

impl Drop for FileInfoListGuard {
    fn drop(&mut self) {
        unsafe { loon_filesystem_free_file_info_list(&mut self.0) }
    }
}

fn is_not_found_error_code(code: i32) -> bool {
    code == unsafe { loon_errcode_file_not_found } || code == unsafe { loon_errcode_aws_not_found }
}

#[derive(Debug)]
struct AimdThrottleError(LoonFFIError);

impl fmt::Display for AimdThrottleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Lance 7 identifies throttling from this RetryError fragment because
        // object_store does not expose a typed throttling error for custom stores.
        write!(f, "{}; AIMD compatibility: retries, max_retries", self.0)
    }
}

impl std::error::Error for AimdThrottleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

fn into_object_store_error(error: LoonFFIError, path: &ObjectPath) -> object_store::Error {
    let code = error.err_code;
    if is_not_found_error_code(code) {
        return object_store::Error::NotFound {
            path: path.to_string(),
            source: Box::new(error),
        };
    }
    if code == unsafe { loon_errcode_aws_access_denied } {
        return object_store::Error::PermissionDenied {
            path: path.to_string(),
            source: Box::new(error),
        };
    }
    if code == unsafe { loon_errcode_aws_precondition_failed } {
        return object_store::Error::Precondition {
            path: path.to_string(),
            source: Box::new(error),
        };
    }
    if code == unsafe { loon_errcode_not_support } {
        return object_store::Error::NotSupported {
            source: Box::new(error),
        };
    }
    if code == unsafe { loon_errcode_transient_throttling } {
        return object_store::Error::Generic {
            store: "filesystem_c",
            source: Box::new(AimdThrottleError(error)),
        };
    }
    object_store::Error::Generic {
        store: "filesystem_c",
        source: Box::new(error),
    }
}

fn validate_get_options(options: &GetOptions) -> object_store::Result<()> {
    let unsupported = if options.version.is_some() {
        Some("version")
    } else if options.if_match.is_some() {
        Some("if_match")
    } else if options.if_none_match.is_some() {
        Some("if_none_match")
    } else if options.if_modified_since.is_some() {
        Some("if_modified_since")
    } else if options.if_unmodified_since.is_some() {
        Some("if_unmodified_since")
    } else {
        None
    };

    match unsupported {
        Some(option) => Err(object_store::Error::NotSupported {
            source: Box::new(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("filesystem_c does not support get option '{option}'"),
            )),
        }),
        None => Ok(()),
    }
}

#[derive(Clone)]
struct FFIObjectStore {
    filesystem: SharedPtr<FileSystemWrapper>,
    mapper: FFIPathMapper,
    metadata: MetadataCache,
}

impl FFIObjectStore {
    fn new(filesystem: SharedPtr<FileSystemWrapper>, mapper: FFIPathMapper) -> Self {
        Self {
            filesystem,
            mapper,
            metadata: MetadataCache::new(),
        }
    }

    fn object_meta(location: ObjectPath, size: u64, mtime_ns: i64) -> ObjectMeta {
        let last_modified = if mtime_ns <= 0 {
            chrono::DateTime::<chrono::Utc>::UNIX_EPOCH
        } else {
            let seconds = mtime_ns.div_euclid(1_000_000_000);
            let nanoseconds = mtime_ns.rem_euclid(1_000_000_000) as u32;
            chrono::DateTime::<chrono::Utc>::from_timestamp(seconds, nanoseconds)
                .unwrap_or(chrono::DateTime::<chrono::Utc>::UNIX_EPOCH)
        };
        ObjectMeta {
            location,
            last_modified,
            size,
            e_tag: None,
            version: None,
        }
    }

    async fn load_metadata(&self, location: &ObjectPath) -> object_store::Result<ObjectMeta> {
        let filesystem = self.filesystem.clone();
        let ffi_path = self.mapper.to_ffi_path(location);
        let object_path = location.clone();
        self.metadata
            .get_or_load(location.clone(), move || async move {
                TOKIO_RT
                    .spawn_blocking(move || {
                        let path_len = u32::try_from(ffi_path.len()).map_err(|error| {
                            object_store::Error::Generic {
                                store: "filesystem_c",
                                source: Box::new(error),
                            }
                        })?;
                        let mut size = 0;
                        let mut mtime_ns = 0;
                        let mut is_dir = false;
                        let mut result = unsafe {
                            loon_filesystem_get_object_info(
                                filesystem.as_mut_ptr().cast::<c_void>(),
                                ffi_path.as_ptr(),
                                path_len,
                                &mut size,
                                &mut mtime_ns,
                                &mut is_dir,
                            )
                        };
                        check_loon_ffi_result(&mut result, "get object metadata")
                            .map_err(|error| into_object_store_error(error, &object_path))?;
                        if is_dir {
                            return Err(object_store::Error::Generic {
                                store: "filesystem_c",
                                source: Box::new(io::Error::new(
                                    io::ErrorKind::IsADirectory,
                                    format!("{} is a directory", object_path),
                                )),
                            });
                        }
                        Ok(Self::object_meta(object_path, size, mtime_ns))
                    })
                    .await
                    .map_err(|source| object_store::Error::Generic {
                        store: "filesystem_c",
                        source: Box::new(source),
                    })?
            })
            .await
    }

    async fn list_impl(
        &self,
        prefix: ObjectPath,
        recursive: bool,
    ) -> object_store::Result<ListResult> {
        let filesystem = self.filesystem.clone();
        let ffi_path = self.mapper.to_ffi_path(&prefix);
        let error_path = prefix.clone();
        let entries = TOKIO_RT
            .spawn_blocking(move || {
                let path_len = u32::try_from(ffi_path.len()).map_err(|error| {
                    object_store::Error::Generic {
                        store: "filesystem_c",
                        source: Box::new(error),
                    }
                })?;
                let mut list = FileInfoListGuard(LoonFileInfoList {
                    entries: std::ptr::null_mut(),
                    count: 0,
                });
                let mut result = unsafe {
                    loon_filesystem_list_dir(
                        filesystem.as_mut_ptr().cast::<c_void>(),
                        ffi_path.as_ptr(),
                        path_len,
                        recursive,
                        &mut list.0,
                    )
                };
                if let Err(list_error) =
                    check_loon_ffi_result(&mut result, "list filesystem directory")
                {
                    if is_not_found_error_code(list_error.err_code) {
                        return Ok(Vec::new());
                    }

                    // Arrow directory listing rejects an exact file path with
                    // ENOTDIR. ObjectStore listing defines exact-file and
                    // nonexistent prefixes as successful empty results. Only
                    // probe metadata after this generic Arrow failure so normal
                    // directory lists do not pay for an extra remote request.
                    if list_error.err_code == unsafe { loon_errcode_arrow } {
                        let mut size = 0;
                        let mut mtime_ns = 0;
                        let mut is_dir = false;
                        let mut info_result = unsafe {
                            loon_filesystem_get_object_info(
                                filesystem.as_mut_ptr().cast::<c_void>(),
                                ffi_path.as_ptr(),
                                path_len,
                                &mut size,
                                &mut mtime_ns,
                                &mut is_dir,
                            )
                        };
                        match check_loon_ffi_result(
                            &mut info_result,
                            "inspect failed filesystem list prefix",
                        ) {
                            Ok(()) if !is_dir => return Ok(Vec::new()),
                            Err(info_error) if is_not_found_error_code(info_error.err_code) => {
                                return Ok(Vec::new());
                            }
                            Ok(()) | Err(_) => {}
                        }
                    }

                    return Err(into_object_store_error(list_error, &error_path));
                }

                if list.0.count != 0 && list.0.entries.is_null() {
                    return Err(object_store::Error::Generic {
                        store: "filesystem_c",
                        source: Box::new(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "filesystem list returned a null entries pointer",
                        )),
                    });
                }

                if list.0.count == 0 {
                    return Ok(Vec::new());
                }

                let infos =
                    unsafe { std::slice::from_raw_parts(list.0.entries, list.0.count as usize) };
                let mut owned = Vec::with_capacity(infos.len());
                for info in infos {
                    if info.path.is_null() {
                        return Err(object_store::Error::Generic {
                            store: "filesystem_c",
                            source: Box::new(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "filesystem list returned a null path",
                            )),
                        });
                    }
                    let bytes = unsafe {
                        std::slice::from_raw_parts(info.path.cast::<u8>(), info.path_len as usize)
                    };
                    let path = std::str::from_utf8(bytes)
                        .map_err(|source| object_store::Error::Generic {
                            store: "filesystem_c",
                            source: Box::new(source),
                        })?
                        .to_string();
                    owned.push((path, info.is_dir, info.size, info.mtime_ns));
                }
                Ok::<_, object_store::Error>(owned)
            })
            .await
            .map_err(|source| object_store::Error::Generic {
                store: "filesystem_c",
                source: Box::new(source),
            })??;

        let mut objects = Vec::new();
        let mut common_prefixes = Vec::new();
        for (path, is_dir, size, mtime_ns) in entries {
            let location = self.mapper.from_ffi_path(&path)?;
            if is_dir {
                if !recursive {
                    common_prefixes.push(location);
                }
            } else {
                let meta = Self::object_meta(location, size, mtime_ns);
                self.metadata.prime(meta.clone());
                objects.push(meta);
            }
        }
        objects.sort_by(|left, right| left.location.cmp(&right.location));
        common_prefixes.sort();
        Ok(ListResult {
            common_prefixes,
            objects,
        })
    }

    async fn read_range(
        &self,
        location: &ObjectPath,
        size: u64,
        range: Range<u64>,
    ) -> object_store::Result<Bytes> {
        let length =
            range
                .end
                .checked_sub(range.start)
                .ok_or_else(|| object_store::Error::Generic {
                    store: "filesystem_c",
                    source: Box::new(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "read range end precedes its start",
                    )),
                })?;
        if length == 0 {
            return Ok(Bytes::new());
        }
        let length_usize =
            usize::try_from(length).map_err(|source| object_store::Error::Generic {
                store: "filesystem_c",
                source: Box::new(source),
            })?;
        let filesystem = self.filesystem.clone();
        let ffi_path = self.mapper.to_ffi_path(location);
        let object_path = location.clone();
        let reader = TOKIO_RT
            .spawn_blocking(move || {
                let path_len = u32::try_from(ffi_path.len()).map_err(|source| {
                    object_store::Error::Generic {
                        store: "filesystem_c",
                        source: Box::new(source),
                    }
                })?;
                let mut reader_raw = std::ptr::null_mut();
                let mut result = unsafe {
                    loon_filesystem_open_reader(
                        filesystem.as_mut_ptr().cast::<c_void>(),
                        ffi_path.as_ptr(),
                        path_len,
                        size,
                        &mut reader_raw,
                    )
                };
                check_loon_ffi_result(&mut result, "open object reader")
                    .map_err(|error| into_object_store_error(error, &object_path))?;

                let supports_async = match reader_supports_async(reader_raw) {
                    Ok(supported) => supported,
                    Err(error) => unsafe {
                        let mut close_result = loon_filesystem_reader_close(reader_raw);
                        if let Err(close_error) =
                            check_loon_ffi_result(&mut close_result, "close object reader")
                        {
                            eprintln!("Warning: ReaderHandle close failed: {close_error}");
                        }
                        loon_filesystem_reader_destroy(reader_raw);
                        return Err(into_object_store_error(error, &object_path));
                    },
                };
                Ok::<_, object_store::Error>(Arc::new(ReaderHandle {
                    ptr: reader_raw,
                    supports_async,
                }))
            })
            .await
            .map_err(|source| object_store::Error::Generic {
                store: "filesystem_c",
                source: Box::new(source),
            })??;

        #[cfg(feature = "s3-crt-async")]
        if reader.supports_async {
            return read_object_store_async_via_ffi(reader, location.clone(), range).await;
        }

        let object_path = location.clone();
        TOKIO_RT
            .spawn_blocking(move || {
                let mut buffer = BytesMut::with_capacity(length_usize);
                let out_data = buffer.spare_capacity_mut().as_mut_ptr().cast::<u8>();
                let mut result = unsafe {
                    loon_filesystem_reader_readat(reader.as_ptr(), range.start, length, out_data)
                };
                check_loon_ffi_result(&mut result, "read object range")
                    .map_err(|error| into_object_store_error(error, &object_path))?;
                unsafe { buffer.set_len(length_usize) };
                Ok(buffer.freeze())
            })
            .await
            .map_err(|source| object_store::Error::Generic {
                store: "filesystem_c",
                source: Box::new(source),
            })?
    }
}

impl fmt::Debug for FFIObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FFIObjectStore(filesystem_c)")
    }
}

impl fmt::Display for FFIObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("filesystem_c")
    }
}

#[async_trait]
impl object_store::ObjectStore for FFIObjectStore {
    async fn put_opts(
        &self,
        _location: &ObjectPath,
        _payload: PutPayload,
        _options: PutOptions,
    ) -> object_store::Result<PutResult> {
        Err(object_store::Error::NotImplemented {
            operation: "put_opts".into(),
            implementer: "FFIObjectStore".into(),
        })
    }

    async fn put_multipart_opts(
        &self,
        _location: &ObjectPath,
        _options: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        Err(object_store::Error::NotImplemented {
            operation: "put_multipart_opts".into(),
            implementer: "FFIObjectStore".into(),
        })
    }

    async fn get_opts(
        &self,
        location: &ObjectPath,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        validate_get_options(&options)?;
        let meta = self.load_metadata(location).await?;
        if options.head {
            return Ok(GetResult {
                payload: GetResultPayload::Stream(stream::once(async { Ok(Bytes::new()) }).boxed()),
                meta,
                range: 0..0,
                attributes: Attributes::default(),
            });
        }

        let range = match options.range {
            Some(range) => {
                range
                    .as_range(meta.size)
                    .map_err(|source| object_store::Error::Generic {
                        store: "filesystem_c",
                        source: Box::new(source),
                    })?
            }
            None => 0..meta.size,
        };
        // FFI range reads return one complete Bytes value. Resolve it before
        // returning GetResult so the outer AIMD wrapper can classify and retry
        // throttling errors instead of losing them in the deferred payload.
        let bytes = self.read_range(location, meta.size, range.clone()).await?;
        Ok(GetResult {
            payload: GetResultPayload::Stream(stream::once(async move { Ok(bytes) }).boxed()),
            meta,
            range,
            attributes: Attributes::default(),
        })
    }

    fn delete_stream(
        &self,
        _locations: BoxStream<'static, object_store::Result<ObjectPath>>,
    ) -> BoxStream<'static, object_store::Result<ObjectPath>> {
        stream::once(async {
            Err(object_store::Error::NotImplemented {
                operation: "delete_stream".into(),
                implementer: "FFIObjectStore".into(),
            })
        })
        .boxed()
    }

    fn list(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        let store = self.clone();
        let prefix = prefix.cloned().unwrap_or(ObjectPath::ROOT);
        async_stream::try_stream! {
            let result = store.list_impl(prefix, true).await?;
            for meta in result.objects {
                yield meta;
            }
        }
        .boxed()
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> object_store::Result<ListResult> {
        self.list_impl(prefix.cloned().unwrap_or(ObjectPath::ROOT), false)
            .await
    }

    async fn copy_opts(
        &self,
        _from: &ObjectPath,
        _to: &ObjectPath,
        _options: CopyOptions,
    ) -> object_store::Result<()> {
        Err(object_store::Error::NotImplemented {
            operation: "copy_opts".into(),
            implementer: "FFIObjectStore".into(),
        })
    }

    async fn rename_opts(
        &self,
        _from: &ObjectPath,
        _to: &ObjectPath,
        _options: RenameOptions,
    ) -> object_store::Result<()> {
        Err(object_store::Error::NotImplemented {
            operation: "rename_opts".into(),
            implementer: "FFIObjectStore".into(),
        })
    }
}

const FS_CACHE_KEY: &str = "milvus_fs_cache_key";
const FS_ROOT_PATH_KEY: &str = "milvus_fs_root_path";
const FS_IS_LOCAL_KEY: &str = "milvus_fs_is_local";
const LANCE_IO_PARALLELISM_KEY: &str = "milvus_lance_io_parallelism";
const AIMD_INITIAL_RATE_KEY: &str = "lance_aimd_initial_rate";
const AIMD_MAX_RATE_KEY: &str = "lance_aimd_max_rate";
const MAX_LANCE_IO_PARALLELISM: usize = 256;

pub(crate) fn native_mutation_store_params(
    mut storage_options: HashMap<String, String>,
) -> Result<ObjectStoreParams> {
    storage_options.remove(LANCE_IO_PARALLELISM_KEY);
    const DYNAMIC_CREDENTIAL_KEYS: &[&str] = &[
        "cloud_provider",
        "milvus_fs_cache_key",
        "aws_role_arn",
        "aws_session_name",
        "aws_external_id",
        "aws_credential_refresh_secs",
        "azure_broker_endpoint",
        "azure_broker_client_id",
        "azure_broker_tenant_id",
        "azure_broker_account_name",
        "azure_broker_region",
        "azure_broker_bucket",
        "azure_broker_duration_seconds",
        "azure_broker_request_timeout_ms",
        "gcp_target_service_account",
        "gcp_credential_refresh_secs",
        "oss_role_arn",
        "oss_role_session_name",
        "oss_external_id",
        "oss_credential_refresh_secs",
    ];
    if let Some(key) = DYNAMIC_CREDENTIAL_KEYS
        .iter()
        .find(|key| storage_options.contains_key(**key))
    {
        return Err(LanceError::invalid_input(format!(
            "unsupported Lance native mutation option: {key}"
        )));
    }

    Ok(ObjectStoreParams {
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            storage_options,
        ))),
        ..Default::default()
    })
}

#[derive(Clone, Debug)]
pub(crate) struct FFIReadOptions {
    fs_cache_key: String,
    pub(crate) is_local: bool,
    root_path: Option<PathBuf>,
    pub(crate) io_parallelism: usize,
    pub(crate) aimd_options: HashMap<String, String>,
}

impl FFIReadOptions {
    pub(crate) fn parse(mut options: HashMap<String, String>) -> Result<Self> {
        let fs_cache_key = options
            .remove(FS_CACHE_KEY)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                LanceError::invalid_input(format!("missing required option: {FS_CACHE_KEY}"))
            })?;
        fs_cache_key
            .strip_prefix("fs:")
            .filter(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
            .ok_or_else(|| {
                LanceError::invalid_input(format!(
                    "{FS_CACHE_KEY} must have the form fs:<unsigned integer>"
                ))
            })?
            .parse::<u64>()
            .map_err(|_| {
                LanceError::invalid_input(format!(
                    "{FS_CACHE_KEY} must have the form fs:<unsigned integer>"
                ))
            })?;

        let is_local_value = options.remove(FS_IS_LOCAL_KEY).ok_or_else(|| {
            LanceError::invalid_input(format!("missing required option: {FS_IS_LOCAL_KEY}"))
        })?;
        let is_local = is_local_value.parse::<bool>().map_err(|_| {
            LanceError::invalid_input(format!(
                "{FS_IS_LOCAL_KEY} must be 'true' or 'false', got '{is_local_value}'"
            ))
        })?;

        let root_path = options.remove(FS_ROOT_PATH_KEY);
        let root_path = if is_local {
            let root = root_path.filter(|value| !value.is_empty()).ok_or_else(|| {
                LanceError::invalid_input(format!(
                    "{FS_ROOT_PATH_KEY} is required for a local filesystem"
                ))
            })?;
            let path = PathBuf::from(root);
            if !path.is_absolute()
                || path
                    .components()
                    .any(|component| matches!(component, Component::ParentDir))
            {
                return Err(LanceError::invalid_input(format!(
                    "{FS_ROOT_PATH_KEY} must be an absolute normalized path"
                )));
            }
            Some(path)
        } else {
            if root_path.is_some() {
                return Err(LanceError::invalid_input(format!(
                    "{FS_ROOT_PATH_KEY} is valid only for a local filesystem"
                )));
            }
            None
        };

        let parallelism_value = options.remove(LANCE_IO_PARALLELISM_KEY).ok_or_else(|| {
            LanceError::invalid_input(format!(
                "missing required option: {LANCE_IO_PARALLELISM_KEY}"
            ))
        })?;
        let io_parallelism = parallelism_value.parse::<usize>().map_err(|_| {
            LanceError::invalid_input(format!(
                "{LANCE_IO_PARALLELISM_KEY} must be an integer in [0, {MAX_LANCE_IO_PARALLELISM}], got '{parallelism_value}'"
            ))
        })?;
        if io_parallelism > MAX_LANCE_IO_PARALLELISM {
            return Err(LanceError::invalid_input(format!(
                "{LANCE_IO_PARALLELISM_KEY} must be in [0, {MAX_LANCE_IO_PARALLELISM}], got {io_parallelism}"
            )));
        }
        let io_parallelism = if io_parallelism == 0 {
            DEFAULT_CLOUD_IO_PARALLELISM
        } else {
            io_parallelism
        };

        let initial_rate_value = options.remove(AIMD_INITIAL_RATE_KEY).ok_or_else(|| {
            LanceError::invalid_input(format!("missing required option: {AIMD_INITIAL_RATE_KEY}"))
        })?;
        let max_rate_value = options.remove(AIMD_MAX_RATE_KEY).ok_or_else(|| {
            LanceError::invalid_input(format!("missing required option: {AIMD_MAX_RATE_KEY}"))
        })?;
        let initial_rate = initial_rate_value.parse::<f64>().map_err(|_| {
            LanceError::invalid_input(format!(
                "{AIMD_INITIAL_RATE_KEY} must be a positive number, got '{initial_rate_value}'"
            ))
        })?;
        let max_rate = max_rate_value.parse::<f64>().map_err(|_| {
            LanceError::invalid_input(format!(
                "{AIMD_MAX_RATE_KEY} must be a non-negative number, got '{max_rate_value}'"
            ))
        })?;
        if !initial_rate.is_finite() || initial_rate <= 0.0 {
            return Err(LanceError::invalid_input(format!(
                "{AIMD_INITIAL_RATE_KEY} must be a positive finite number"
            )));
        }
        if !max_rate.is_finite() || max_rate < 0.0 {
            return Err(LanceError::invalid_input(format!(
                "{AIMD_MAX_RATE_KEY} must be a non-negative finite number"
            )));
        }
        if max_rate > 0.0 && initial_rate > max_rate {
            return Err(LanceError::invalid_input(format!(
                "{AIMD_INITIAL_RATE_KEY} must not exceed {AIMD_MAX_RATE_KEY}"
            )));
        }

        if let Some((key, _)) = options.into_iter().next() {
            return Err(LanceError::invalid_input(format!(
                "unsupported Lance filesystem read option: {key}"
            )));
        }

        Ok(Self {
            fs_cache_key,
            is_local,
            root_path,
            io_parallelism,
            aimd_options: HashMap::from([
                (AIMD_INITIAL_RATE_KEY.into(), initial_rate_value),
                (AIMD_MAX_RATE_KEY.into(), max_rate_value),
            ]),
        })
    }

    fn path_mapper(&self, dataset_url: &Url) -> Result<FFIPathMapper> {
        if self.is_local {
            let mapper = FFIPathMapper::local(
                self.root_path
                    .as_ref()
                    .expect("validated local read options contain a root path"),
            );
            mapper
                .from_url(dataset_url)
                .map_err(|error| LanceError::invalid_input(error.to_string()))?;
            Ok(mapper)
        } else {
            let authority = dataset_url.host_str().ok_or_else(|| {
                LanceError::invalid_input("remote Lance dataset URI must contain an authority")
            })?;
            let mapper = FFIPathMapper::remote(dataset_url.scheme(), authority);
            mapper
                .from_url(dataset_url)
                .map_err(|error| LanceError::invalid_input(error.to_string()))?;
            Ok(mapper)
        }
    }

    pub(crate) fn object_store_prefix(&self) -> &str {
        &self.fs_cache_key
    }
}

pub(crate) struct FFIObjectStoreProvider {
    filesystem: SharedPtr<FileSystemWrapper>,
    mapper: FFIPathMapper,
    fs_cache_key: String,
    block_size: usize,
    io_parallelism: usize,
    aimd_options: HashMap<String, String>,
}

impl FFIObjectStoreProvider {
    pub(crate) fn new(
        filesystem: SharedPtr<FileSystemWrapper>,
        dataset_url: &Url,
        options: &FFIReadOptions,
    ) -> Result<Self> {
        Ok(Self {
            filesystem,
            mapper: options.path_mapper(dataset_url)?,
            fs_cache_key: options.fs_cache_key.clone(),
            block_size: if options.is_local {
                4 * 1024
            } else {
                64 * 1024
            },
            io_parallelism: options.io_parallelism,
            aimd_options: options.aimd_options.clone(),
        })
    }
}

fn build_unwrapped_ffi_store(
    inner: Arc<dyn object_store::ObjectStore>,
    store_prefix: &str,
    block_size: usize,
    io_parallelism: usize,
    download_retry_count: usize,
) -> ObjectStore {
    // `ObjectStoreRegistry::get_store` owns tracing, custom wrappers, and I/O
    // tracking. Use Lance's public constructor only to initialize its private
    // fields, then restore the raw inner store before returning to the registry.
    // This registered scheme has a pure prefix calculation and, unlike `file`,
    // forces Lance to use the supplied ObjectStore. It therefore initializes
    // private wrapper fields without consulting any native cloud provider or
    // reading provider environment variables.
    let constructor_location =
        Url::parse("file-object-store:///").expect("constant file-object-store URL is valid");
    let mut store = ObjectStore::new(
        inner.clone(),
        constructor_location,
        Some(block_size),
        None,
        false,
        true,
        io_parallelism,
        download_retry_count,
        None,
    );
    store.inner = inner;
    store.store_prefix = store_prefix.to_string();
    store
}

impl std::fmt::Debug for FFIObjectStoreProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FFIObjectStoreProvider(filesystem_c)")
    }
}

#[async_trait::async_trait]
impl ObjectStoreProvider for FFIObjectStoreProvider {
    async fn new_store(&self, base_path: Url, _params: &ObjectStoreParams) -> Result<ObjectStore> {
        self.mapper
            .from_url(&base_path)
            .map_err(|error| LanceError::invalid_input(error.to_string()))?;

        let inner = Arc::new(FFIObjectStore::new(
            self.filesystem.clone(),
            self.mapper.clone(),
        )) as Arc<dyn object_store::ObjectStore>;
        let throttle_config = AimdThrottleConfig::from_storage_options(Some(&self.aimd_options))?;
        let inner = if throttle_config.is_disabled() {
            inner
        } else {
            Arc::new(AimdThrottledStore::new(inner, throttle_config)?)
                as Arc<dyn object_store::ObjectStore>
        };

        Ok(build_unwrapped_ffi_store(
            inner,
            &self.fs_cache_key,
            self.block_size,
            self.io_parallelism,
            DEFAULT_DOWNLOAD_RETRY_COUNT,
        ))
    }

    fn extract_path(&self, url: &Url) -> Result<object_store::path::Path> {
        self.mapper
            .from_url(url)
            .map_err(|error| LanceError::invalid_input(error.to_string()))
    }

    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        _storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        self.extract_path(url)?;
        Ok(self.fs_cache_key.clone())
    }
}

pub(crate) fn build_filesystem_session(
    scheme: &str,
    provider: Arc<dyn ObjectStoreProvider>,
) -> Arc<Session> {
    let registry = ObjectStoreRegistry::empty();
    registry.insert(scheme, provider);
    Arc::new(Session::new(0, 0, Arc::new(registry)))
}

// Weak values let the registry reuse active schedulers without extending their
// lifetime after all datasets in the domain have been dropped.
static LANCE_IO_SCHEDULERS: LazyLock<Mutex<HashMap<String, Weak<ScanScheduler>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Returns the single active ScanScheduler for a filesystem.
///
/// The registry lock covers lookup, construction, and insertion so concurrent opens
/// cannot create competing schedulers for the same filesystem. Only weak references are
/// retained, allowing a later open to create a fresh scheduler after the domain is idle.
pub(crate) fn shared_scan_scheduler(
    fs_cache_key: String,
    object_store: &Arc<ObjectStore>,
) -> Result<Arc<ScanScheduler>> {
    let mut schedulers = LANCE_IO_SCHEDULERS
        .lock()
        .map_err(|_| LanceError::Internal {
            message: "Lance I/O scheduler registry mutex poisoned".into(),
            location: snafu::location!(),
        })?;
    if let Some(scheduler) = schedulers.get(&fs_cache_key).and_then(Weak::upgrade) {
        return Ok(scheduler);
    }

    // Sweep expired weak entries only on a registry miss; active hits stay O(1).
    schedulers.retain(|_, scheduler| scheduler.strong_count() != 0);
    let scheduler = TOKIO_RT.block_on(async {
        ScanScheduler::new(
            object_store.clone(),
            SchedulerConfig::max_bandwidth(object_store),
        )
    });
    schedulers.insert(fs_cache_key, Arc::downgrade(&scheduler));
    Ok(scheduler)
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn native_mutation_options_reject_dynamic_credentials() {
        for key in [
            "cloud_provider",
            "milvus_fs_cache_key",
            "aws_role_arn",
            "azure_broker_endpoint",
            "gcp_target_service_account",
            "oss_role_arn",
        ] {
            let options = HashMap::from([(key.to_string(), "value".to_string())]);
            let error = native_mutation_store_params(options).err().unwrap();
            assert!(error.to_string().contains(key));
        }

        let static_options = HashMap::from([
            ("aws_access_key_id".to_string(), "access-key".to_string()),
            (
                "aws_secret_access_key".to_string(),
                "secret-key".to_string(),
            ),
        ]);
        assert!(native_mutation_store_params(static_options).is_ok());
    }

    #[test]
    fn remote_uri_path_is_relative_to_bucket() {
        let mapper = FFIPathMapper::remote("s3", "bucket");
        assert_eq!(
            mapper
                .from_url(&url::Url::parse("s3://bucket/a/b.lance").unwrap())
                .unwrap(),
            object_store::path::Path::from("a/b.lance")
        );
    }

    #[test]
    fn remote_uri_requires_matching_storage_identity() {
        let mapper = FFIPathMapper::remote("s3", "bucket");
        assert!(
            mapper
                .from_url(&url::Url::parse("gs://bucket/a.lance").unwrap())
                .is_err()
        );
        assert!(
            mapper
                .from_url(&url::Url::parse("s3://other/a.lance").unwrap())
                .is_err()
        );
    }

    #[test]
    fn local_path_is_relative_to_normalized_root() {
        let mapper = FFIPathMapper::local("/data/root/");
        assert_eq!(
            mapper.from_local_path("/data/root/a/b.lance").unwrap(),
            object_store::path::Path::from("a/b.lance")
        );
    }

    #[test]
    fn local_path_outside_root_is_rejected() {
        let mapper = FFIPathMapper::local("/data/root");
        assert!(mapper.from_local_path("/other/file.lance").is_err());
        assert!(
            mapper
                .from_local_path("/data/root/../other/file.lance")
                .is_err()
        );
    }

    #[test]
    fn versioned_and_conditional_gets_are_rejected() {
        let now = chrono::Utc::now();
        let unsupported = [
            (
                "version",
                object_store::GetOptions::new().with_version(Some("7")),
            ),
            (
                "if_match",
                object_store::GetOptions::new().with_if_match(Some("etag")),
            ),
            (
                "if_none_match",
                object_store::GetOptions::new().with_if_none_match(Some("etag")),
            ),
            (
                "if_modified_since",
                object_store::GetOptions::new().with_if_modified_since(Some(now)),
            ),
            (
                "if_unmodified_since",
                object_store::GetOptions::new().with_if_unmodified_since(Some(now)),
            ),
        ];

        for (name, options) in unsupported {
            assert!(
                validate_get_options(&options).is_err(),
                "{name} must be rejected"
            );
        }

        assert!(
            validate_get_options(
                &object_store::GetOptions::new()
                    .with_range(Some(1_u64..2))
                    .with_head(true)
            )
            .is_ok()
        );
    }

    #[tokio::test]
    async fn metadata_cache_single_flights_concurrent_loads() {
        let cache = Arc::new(MetadataCache::new());
        let loads = Arc::new(AtomicUsize::new(0));
        let location = object_store::path::Path::from("same.lance");
        let mut tasks = Vec::new();

        for _ in 0..16 {
            let cache = cache.clone();
            let loads = loads.clone();
            let location = location.clone();
            tasks.push(tokio::spawn(async move {
                cache
                    .get_or_load(location.clone(), || async move {
                        loads.fetch_add(1, Ordering::SeqCst);
                        tokio::task::yield_now().await;
                        Ok(object_store::ObjectMeta {
                            location,
                            last_modified: chrono::DateTime::<chrono::Utc>::UNIX_EPOCH,
                            size: 42,
                            e_tag: None,
                            version: None,
                        })
                    })
                    .await
            }));
        }

        for task in tasks {
            assert_eq!(task.await.unwrap().unwrap().size, 42);
        }
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn metadata_cache_retries_failed_initialization() {
        let cache = MetadataCache::new();
        let location = object_store::path::Path::from("retry.lance");
        let first = cache
            .get_or_load(location.clone(), || async {
                Err(object_store::Error::Generic {
                    store: "test",
                    source: "first load failed".into(),
                })
            })
            .await;
        assert!(first.is_err());

        let second = cache
            .get_or_load(location.clone(), || async {
                Ok(object_store::ObjectMeta {
                    location,
                    last_modified: chrono::DateTime::<chrono::Utc>::UNIX_EPOCH,
                    size: 7,
                    e_tag: None,
                    version: None,
                })
            })
            .await
            .unwrap();
        assert_eq!(second.size, 7);
    }

    #[test]
    fn metadata_cache_capacity_is_bounded() {
        let cache = MetadataCache::new();
        for index in 0..(FFI_OBJECT_META_CACHE_CAPACITY + 10) {
            cache.prime(object_store::ObjectMeta {
                location: object_store::path::Path::from(format!("{index}.lance")),
                last_modified: chrono::DateTime::<chrono::Utc>::UNIX_EPOCH,
                size: index as u64,
                e_tag: None,
                version: None,
            });
        }

        assert_eq!(
            cache.entries.lock().unwrap().len(),
            FFI_OBJECT_META_CACHE_CAPACITY
        );
    }

    fn valid_filesystem_read_options(is_local: bool) -> HashMap<String, String> {
        let mut options = HashMap::from([
            (FS_CACHE_KEY.into(), "fs:123".into()),
            (FS_IS_LOCAL_KEY.into(), is_local.to_string()),
            (LANCE_IO_PARALLELISM_KEY.into(), "64".into()),
            (AIMD_INITIAL_RATE_KEY.into(), "2000".into()),
            (AIMD_MAX_RATE_KEY.into(), "5000".into()),
        ]);
        if is_local {
            options.insert(FS_ROOT_PATH_KEY.into(), "/data/root".into());
        }
        options
    }

    #[test]
    fn read_options_reject_credentials() {
        let mut options = valid_filesystem_read_options(false);
        options.insert("aws_access_key_id".into(), "secret".into());
        assert!(FFIReadOptions::parse(options).is_err());
    }

    #[test]
    fn read_options_validate_required_identity_and_limits() {
        let mut missing_cache_key = valid_filesystem_read_options(false);
        missing_cache_key.remove(FS_CACHE_KEY);
        assert!(FFIReadOptions::parse(missing_cache_key).is_err());

        let mut invalid_bool = valid_filesystem_read_options(false);
        invalid_bool.insert(FS_IS_LOCAL_KEY.into(), "remote".into());
        assert!(FFIReadOptions::parse(invalid_bool).is_err());

        let mut missing_local_root = valid_filesystem_read_options(true);
        missing_local_root.remove(FS_ROOT_PATH_KEY);
        assert!(FFIReadOptions::parse(missing_local_root).is_err());

        let mut excessive_parallelism = valid_filesystem_read_options(false);
        excessive_parallelism.insert(LANCE_IO_PARALLELISM_KEY.into(), "257".into());
        assert!(FFIReadOptions::parse(excessive_parallelism).is_err());
    }

    #[test]
    fn read_options_zero_parallelism_uses_default() {
        let mut options = valid_filesystem_read_options(false);
        options.insert(LANCE_IO_PARALLELISM_KEY.into(), "0".into());

        let options = FFIReadOptions::parse(options).unwrap();
        assert_eq!(options.io_parallelism, DEFAULT_CLOUD_IO_PARALLELISM);
    }

    #[test]
    fn read_options_accept_valid_remote_and_local_maps() {
        let remote = FFIReadOptions::parse(valid_filesystem_read_options(false)).unwrap();
        assert!(!remote.is_local);
        assert_eq!(remote.root_path, None);
        assert_eq!(remote.io_parallelism, 64);

        let local = FFIReadOptions::parse(valid_filesystem_read_options(true)).unwrap();
        assert!(local.is_local);
        assert_eq!(
            local.root_path.as_deref(),
            Some(std::path::Path::new("/data/root"))
        );
        assert_eq!(local.io_parallelism, 64);
    }

    #[test]
    fn filesystem_provider_uses_configured_parallelism() {
        for (is_local, dataset_uri) in [
            (true, "file:///data/root/table"),
            (false, "s3://bucket/table"),
        ] {
            let mut options = valid_filesystem_read_options(is_local);
            options.insert(LANCE_IO_PARALLELISM_KEY.into(), "17".into());
            let options = FFIReadOptions::parse(options).unwrap();
            let dataset_url = Url::parse(dataset_uri).unwrap();
            let provider =
                FFIObjectStoreProvider::new(SharedPtr::null(), &dataset_url, &options).unwrap();

            assert_eq!(provider.io_parallelism, 17);
        }
    }

    #[test]
    fn filesystem_provider_identity_and_paths_are_strict() {
        let options = FFIReadOptions::parse(valid_filesystem_read_options(false)).unwrap();
        let dataset_url = Url::parse("s3://bucket/table").unwrap();
        let mapper = options.path_mapper(&dataset_url).unwrap();

        assert!(mapper.from_url(&dataset_url).is_ok());
        assert!(
            mapper
                .from_url(&Url::parse("gs://bucket/table").unwrap())
                .is_err()
        );
        assert!(
            mapper
                .from_url(&Url::parse("s3://other/table").unwrap())
                .is_err()
        );
        assert_eq!(options.object_store_prefix(), "fs:123");

        assert!(!options.object_store_prefix().contains("0x"));
    }

    #[test]
    fn filesystem_store_leaves_wrapping_and_tracking_to_registry() {
        let inner = Arc::new(InMemory::new()) as Arc<dyn object_store::ObjectStore>;
        let store = build_unwrapped_ffi_store(
            inner.clone(),
            "fs:123",
            64 * 1024,
            64,
            DEFAULT_DOWNLOAD_RETRY_COUNT,
        );

        assert!(Arc::ptr_eq(&store.inner, &inner));
        assert_eq!(store.store_prefix, "fs:123");
        assert_eq!(store.scheme(), "file-object-store");
    }

    #[derive(Debug)]
    struct UnusedObjectStoreProvider;

    #[async_trait::async_trait]
    impl ObjectStoreProvider for UnusedObjectStoreProvider {
        async fn new_store(
            &self,
            _base_path: Url,
            _params: &ObjectStoreParams,
        ) -> Result<ObjectStore> {
            unreachable!("registry membership test never creates a store")
        }
    }

    #[test]
    fn filesystem_session_registry_has_no_native_fallback() {
        let session = build_filesystem_session("s3", Arc::new(UnusedObjectStoreProvider));
        let registry = session.store_registry();

        assert!(registry.get_provider("s3").is_some());
        assert!(registry.get_provider("gs").is_none());
        assert!(registry.get_provider("file").is_none());
    }

    #[test]
    fn scheduler_registry_reuses_active_scheduler_without_retaining_it() {
        let object_store = Arc::new(ObjectStore::new(
            Arc::new(InMemory::new()),
            Url::parse("s3://shared-bucket/dataset").unwrap(),
            Some(64 * 1024),
            None,
            false,
            true,
            64,
            3,
            None,
        ));
        let key = "fs:scheduler-reuse".to_string();

        let first = shared_scan_scheduler(key.clone(), &object_store).unwrap();
        let second = shared_scan_scheduler(key.clone(), &object_store).unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        let later_object_store = Arc::new(ObjectStore::new(
            Arc::new(InMemory::new()),
            Url::parse("s3://shared-bucket/other-dataset").unwrap(),
            Some(64 * 1024),
            None,
            false,
            true,
            1,
            3,
            None,
        ));
        let later_value = shared_scan_scheduler(key.clone(), &later_object_store).unwrap();
        assert!(Arc::ptr_eq(&first, &later_value));

        let scheduler = Arc::downgrade(&first);
        drop(first);
        drop(second);
        drop(later_value);
        assert!(scheduler.upgrade().is_none());

        let replacement = shared_scan_scheduler(key, &object_store).unwrap();
        assert!(!Weak::ptr_eq(&scheduler, &Arc::downgrade(&replacement)));
    }

    #[test]
    fn scheduler_registry_get_or_create_is_atomic() {
        let object_store = Arc::new(ObjectStore::new(
            Arc::new(InMemory::new()),
            Url::parse("s3://concurrent-bucket/dataset").unwrap(),
            Some(64 * 1024),
            None,
            false,
            true,
            64,
            3,
            None,
        ));
        let key = "fs:scheduler-atomic".to_string();
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let threads = (0..8)
            .map(|_| {
                let object_store = object_store.clone();
                let key = key.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    shared_scan_scheduler(key, &object_store).unwrap()
                })
            })
            .collect::<Vec<_>>();
        let schedulers = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        for scheduler in schedulers.iter().skip(1) {
            assert!(Arc::ptr_eq(&schedulers[0], scheduler));
        }
    }
}
