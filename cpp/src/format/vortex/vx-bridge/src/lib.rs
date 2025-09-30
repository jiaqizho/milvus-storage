// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod objstore;
pub mod objwriter;

pub use objstore::ObjectStoreWrapper;

pub use objstore::{
    create_object_store,
    free_object_store_wrapper
};

pub use objwriter::{
    create_object_store_writer,
    free_object_store_writer
};

pub mod errcode;

pub mod test;
pub use test::{
    test_bridge_object_store_async_to_sync
};

pub mod reader;
use reader::*;

#[cxx::bridge(namespace = "milvus_storage::vortex::ffi")]
mod ffi {

    extern "Rust" {
        type DType;
        // Factory functions for creating DType
        fn dtype_null() -> Box<DType>;
        fn dtype_bool(nullable: bool) -> Box<DType>;
        fn dtype_primitive(ptype: PType, nullable: bool) -> Box<DType>;
        fn dtype_decimal(precision: u8, scale: i8, nullable: bool) -> Box<DType>;
        fn dtype_utf8(nullable: bool) -> Box<DType>;
        fn dtype_binary(nullable: bool) -> Box<DType>;
        unsafe fn from_arrow(ffi_schema: *mut u8, non_nullable: bool) -> Result<Box<DType>>;
        // Methods for DType
        fn to_string(self: &DType) -> String;

        type Scalar;
        fn bool_scalar_new(value: bool) -> Box<Scalar>;
        fn i8_scalar_new(value: i8) -> Box<Scalar>;
        fn i16_scalar_new(value: i16) -> Box<Scalar>;
        fn i32_scalar_new(value: i32) -> Box<Scalar>;
        fn i64_scalar_new(value: i64) -> Box<Scalar>;
        fn u8_scalar_new(value: u8) -> Box<Scalar>;
        fn u16_scalar_new(value: u16) -> Box<Scalar>;
        fn u32_scalar_new(value: u32) -> Box<Scalar>;
        fn u64_scalar_new(value: u64) -> Box<Scalar>;
        fn f32_scalar_new(value: f32) -> Box<Scalar>;
        fn f64_scalar_new(value: f64) -> Box<Scalar>;
        fn string_scalar_new(value: &str) -> Box<Scalar>;
        fn binary_scalar_new(value: &[u8]) -> Box<Scalar>;
        fn cast_scalar(self: &Scalar, dtype: &DType) -> Result<Box<Scalar>>;

        type Expr;
        fn literal(scalar: Box<Scalar>) -> Box<Expr>;
        fn root() -> Box<Expr>;
        fn column(name: String) -> Box<Expr>;
        fn get_item(field: String, child: Box<Expr>) -> Box<Expr>;
        fn not_(child: Box<Expr>) -> Box<Expr>;
        fn is_null(child: Box<Expr>) -> Box<Expr>;
        // binary op
        fn eq(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn not_eq_(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn gt(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn gt_eq(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn lt(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn lt_eq(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn and_(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn or_(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn checked_add(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn select(fields: Vec<String>, child: Box<Expr>) -> Box<Expr>;

        type ObjectStoreWrapper2;
        pub(crate) fn open_object_store(ostype: &str, endpoint: &str, access_key_id: &str, secret_access_key: &str, region: &str, bucket_name: &str) -> Result<Box<ObjectStoreWrapper2>>;
        type VortexFile;
        fn row_count(self: &VortexFile) -> u64;
        fn scan_builder(self: &VortexFile) -> Result<Box<VortexScanBuilder>>;

        fn open_file(object_store_wrapper: &Box<ObjectStoreWrapper2>, path: &str) -> Result<Box<VortexFile>>;

        type VortexScanBuilder;
        fn with_filter(self: &mut VortexScanBuilder, filter: Box<Expr>);
        fn with_filter_ref(self: &mut VortexScanBuilder, filter: &Expr);
        fn with_projection(self: &mut VortexScanBuilder, projection: Box<Expr>);
        fn with_projection_ref(self: &mut VortexScanBuilder, projection: &Expr);
        fn with_row_range(self: &mut VortexScanBuilder, row_range_start: u64, row_range_end: u64);
        fn with_include_by_index(self: &mut VortexScanBuilder, include_by_index: &[u64]);
        fn with_limit(self: &mut VortexScanBuilder, limit: usize);
        unsafe fn with_output_schema(
            self: &mut VortexScanBuilder,
            output_schema: *mut u8,
        ) -> Result<()>;
        unsafe fn scan_builder_into_stream(
            builder: Box<VortexScanBuilder>,
            out_stream: *mut u8,
        ) -> Result<()>;
        fn scan_builder_into_threadsafe_cloneable_reader(
            builder: Box<VortexScanBuilder>,
        ) -> Result<Box<ThreadsafeCloneableReader>>;

        type ThreadsafeCloneableReader;
        unsafe fn clone_a_stream(self: &ThreadsafeCloneableReader, out_stream: *mut u8);
    }

    #[repr(u8)]
    #[derive(Debug, Clone, Copy)]
    enum PType {
        U8,
        U16,
        U32,
        U64,
        I8,
        I16,
        I32,
        I64,
        F16,
        F32,
        F64,
    }
}
