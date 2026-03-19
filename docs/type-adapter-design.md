# TypeAdapter: Format-Agnostic Type Conversion Layer

## Problem

When the read schema requires `FixedSizeBinary(N)` but the underlying format stores
`FixedSizeList<u8, N>`, reads fail with a type mismatch — except for Vortex, which
has its own conversion layer in the Rust bridge.

| Format  | Stores FSB as | Has read-side conversion? |
|---------|---------------|---------------------------|
| Parquet | FSB (native)  | N/A — types already match |
| Vortex  | FSL\<u8\>     | Yes (in Rust bridge)      |
| Lance   | FSB (native)  | No                        |

Vortex cannot represent `FixedSizeBinary` in its type system, so it transparently
converts `FSB → FSL<u8>` on write and `FSL<u8> → FSB` on read. This conversion
lives deep inside the Vortex Rust bridge (`vortex_bridgeimpl.rs`), making it
invisible to other formats. If a Parquet or Lance file happens to store `FSL<u8>`
(e.g., written by a non-standard producer), the read will fail because there is no
equivalent conversion layer for those formats.

## Goal

Introduce a **unified, format-agnostic type conversion layer** (`TypeAdapter`) that
sits between `FormatReader` and `ColumnGroupReader`. Each format reader reports its
physical output schema; `TypeAdapter` reconciles it with the target schema the caller
expects.

## Design

### Conversion Rules

Each rule is a template specialization of `TypeConversion<From, To>` with two static
methods: `Match` and `Convert`. The primary template is intentionally left undefined —
an unspecialized pair triggers a compile error.

```cpp
// type_adapter.h

using ConvertFn = arrow::Result<std::shared_ptr<arrow::Array>>(*)(
    const std::shared_ptr<arrow::Array>& source,
    const std::shared_ptr<arrow::DataType>& target_type);

template <typename From, typename To>
struct TypeConversion;  // primary — undefined
```

Rules are defined in `type_adapter.cpp`:

```cpp
template <>
struct TypeConversion<arrow::FixedSizeListType, arrow::FixedSizeBinaryType> {
    static bool Match(const arrow::DataType& from, const arrow::DataType& to);
    static arrow::Result<std::shared_ptr<arrow::Array>> Convert(
        const std::shared_ptr<arrow::Array>& source,
        const std::shared_ptr<arrow::DataType>& target_type);
};

template <>
struct TypeConversion<arrow::FixedSizeBinaryType, arrow::FixedSizeListType> {
    static bool Match(const arrow::DataType& from, const arrow::DataType& to);
    static arrow::Result<std::shared_ptr<arrow::Array>> Convert(
        const std::shared_ptr<arrow::Array>& source,
        const std::shared_ptr<arrow::DataType>& target_type);
};
```

Adding a new conversion: write a specialization and register it in `AllRules`.

### TypeAdapter Class

`TypeAdapter` is a non-template class. Only the internal `Resolve` method is
templated, and it is hidden in the `.cpp` file.

```cpp
class TypeAdapter {
  public:
    static arrow::Result<TypeAdapter> Build(
        std::shared_ptr<arrow::Schema> target,
        const std::shared_ptr<arrow::Schema>& source);

    arrow::Result<std::shared_ptr<arrow::RecordBatch>> operator()(
        const std::shared_ptr<arrow::RecordBatch>& batch) const;

    arrow::Result<std::shared_ptr<arrow::Table>> operator()(
        const std::shared_ptr<arrow::Table>& table) const;

  private:
    template <typename... Conversions>
    static ConvertFn Resolve(const arrow::DataType& from, const arrow::DataType& to);

    struct Step {
        int col;
        ConvertFn fn;
        std::shared_ptr<arrow::DataType> target_type;
    };

    std::shared_ptr<arrow::Schema> target_;
    std::vector<Step> plan_;  // only columns that need conversion
};
```

Key properties:

- **Value semantics.** `Build` returns `TypeAdapter` by value, not a pointer.
- **Passthrough when plan is empty.** `operator()` returns the input unchanged when
  no columns need conversion — zero overhead.
- **Pre-computed plan.** Schema comparison and rule matching happen once at `Build`
  time. Per-batch cost is a loop over `plan_` entries only.
- **Raw function pointers.** `ConvertFn` is a plain function pointer, not
  `std::function` — no heap allocation, no virtual call.
- **Callers do not see conversion rules.** Rules are registered inside
  `type_adapter.cpp` via an `AllRules` tuple + fold expression. The call site is
  simply `TypeAdapter::Build(target, source)`.

### Rule Resolution (inside `.cpp`)

```cpp
using AllRules = std::tuple<
    TypeConversion<arrow::FixedSizeListType, arrow::FixedSizeBinaryType>,
    TypeConversion<arrow::FixedSizeBinaryType, arrow::FixedSizeListType>
>;

template <typename... Conversions>
ConvertFn TypeAdapter::Resolve(const arrow::DataType& from, const arrow::DataType& to) {
    ConvertFn fn = nullptr;
    ((Conversions::Match(from, to) ? (fn = &Conversions::Convert, true) : false) || ...);
    return fn;
}
```

`Build` uses `std::apply` to expand `AllRules` into `Resolve<...>`.

### Call Site

```cpp
// ColumnGroupReaderImpl::open()
auto source_schema = format_readers_[0]->output_schema();
ARROW_ASSIGN_OR_RAISE(adapter_, TypeAdapter::Build(schema_, source_schema));

// ColumnGroupReaderImpl::get_chunk()
ARROW_ASSIGN_OR_RAISE(auto rb, format_readers_[...]->get_chunk(...));
ARROW_ASSIGN_OR_RAISE(rb, adapter_(rb));
```

## Implementation Plan

### Step 1 — Vortex Bridge: Add `GetSchema()` FFI

Expose the Vortex file's native schema (which uses `FixedSizeList<u8>` for vectors)
to the C++ side.

**Rust side** (`vortex_bridgeimpl.rs`):
```rust
impl VortexFile {
    pub(crate) fn schema(&self, out_schema: *mut u8) -> Result<()> {
        let dtype = self.inner.dtype();
        let arrow_schema = dtype.to_arrow_schema()?;
        let ffi_schema = FFI_ArrowSchema::try_from(&arrow_schema)?;
        unsafe { std::ptr::write(out_schema as *mut FFI_ArrowSchema, ffi_schema) };
        Ok(())
    }
}
```

The path already exists internally — `scan_builder_into_stream` calls
`builder.inner.dtype()?.to_arrow_schema()` when no output schema is set.

**Files:**

| File | Change |
|------|--------|
| `cpp/src/format/bridge/rust/src/vortex_bridgeimpl.rs` | Add `schema()` method |
| `cpp/src/format/bridge/rust/src/lib.rs` | Add FFI declaration |
| `cpp/src/format/bridge/rust/include/vortex_bridge.h` | Add `VortexFile::GetSchema()` |
| `cpp/src/format/bridge/rust/src/vortex_bridge.cpp` | C++ implementation |

### Step 2 — Lance Bridge: Add `GetFragmentSchema()` FFI

Expose the physical schema of a specific fragment to the C++ side. Using
fragment-level schema (rather than dataset-level) is more correct in the
presence of schema evolution. Lance preserves Arrow types as-is (FSB stays
FSB, FSL stays FSL).

**Rust side** (`lance_bridgeimpl.rs`):
```rust
pub fn get_fragment_schema(
    dataset: &BlockingDataset,
    fragment_id: u64,
    out_schema: *mut u8,
) -> Result<()> {
    let fragment = dataset.get_fragment(fragment_id)...;
    let file_fragment = FileFragment::new(Arc::new(dataset.inner.clone()), fragment);
    let lance_schema = file_fragment.schema();
    let arrow_schema: ArrowSchema = lance_schema.into();
    // export via C FFI
}
```

**Files:**

| File | Change |
|------|--------|
| `cpp/src/format/bridge/rust/src/lance_bridgeimpl.rs` | Add `get_fragment_schema()` |
| `cpp/src/format/bridge/rust/src/lib.rs` | Add FFI declaration |
| `cpp/src/format/bridge/rust/include/lance_bridge.h` | Add `BlockingDataset::GetFragmentSchema()` |
| `cpp/src/format/bridge/rust/src/lance_bridge.cpp` | C++ implementation |

### Step 3 — FormatReader: Add `output_schema()` Virtual Method

**Add to** `cpp/include/milvus-storage/format/format_reader.h`:
```cpp
virtual std::shared_ptr<arrow::Schema> output_schema() const = 0;
```

Each sub-class implements it:

| Sub-class | Returns | Source |
|-----------|---------|--------|
| `ParquetFormatReader` | `schema_` | Already set in `open()` via `file_reader_->GetSchema()` |
| `VortexFormatReader` | `physical_schema_` (new member) | Set in `open()` via `vxfile_->GetSchema()` (Step 1) |
| `LanceTableReader` | `physical_schema_` (new member) | Set in `open()` via `dataset_->GetFragmentSchema()` (Step 2) |

**Files:**

| File | Change |
|------|--------|
| `cpp/include/milvus-storage/format/format_reader.h` | Add pure virtual |
| `cpp/include/milvus-storage/format/parquet/parquet_format_reader.h` | Declare override |
| `cpp/src/format/parquet/parquet_format_reader.cpp` | `return schema_;` |
| `cpp/include/milvus-storage/format/vortex/vortex_format_reader.h` | Declare override + `physical_schema_` member |
| `cpp/src/format/vortex/vortex_format_reader.cpp` | Fetch in `open()`, return in `output_schema()` |
| `cpp/include/milvus-storage/format/lance/lance_table_reader.h` | Declare override + `physical_schema_` member |
| `cpp/src/format/lance/lance_table_reader.cpp` | Fetch in `open()`, return in `output_schema()` |

### Step 4 — Implement TypeAdapter

**New files:**
- `cpp/include/milvus-storage/format/type_adapter.h`
- `cpp/src/format/type_adapter.cpp`

Both are automatically picked up by `GLOB_RECURSE src/*.cpp` in `CMakeLists.txt`.

**`type_adapter.h`** — Class declaration as shown in the Design section above.

**`type_adapter.cpp`** contains:
- `TypeConversion<FixedSizeListType, FixedSizeBinaryType>` specialization
  (zero-copy: reuse underlying buffer)
- `TypeConversion<FixedSizeBinaryType, FixedSizeListType>` specialization (reverse)
- `AllRules` tuple
- `Resolve` template expansion via `std::apply`
- `Build` implementation
- `operator()(RecordBatch)` and `operator()(Table)` implementations

### Step 5 — Integrate TypeAdapter into ColumnGroupReader

**File:** `cpp/src/format/column_group_reader.cpp`

Add `TypeAdapter adapter_;` member to `ColumnGroupReaderImpl`.

**In `open()`:**
```cpp
auto source_schema = format_readers_[0]->output_schema();
ARROW_ASSIGN_OR_RAISE(adapter_, TypeAdapter::Build(schema_, source_schema));
```

**In read paths, add one line before returning:**
- `get_chunk()`: `ARROW_ASSIGN_OR_RAISE(rb, adapter_(rb));`
- `read_chunks_from_files()`: `ARROW_ASSIGN_OR_RAISE(rb, adapter_(rb));` when filling
  `chunk_rb_map`

### Step 6 — Tests

**New file:** `cpp/test/format/type_adapter_test.cpp`

| Test case | Validates |
|-----------|-----------|
| Build — no conversion needed | source == target → plan empty, operator() passthrough |
| FSL → FSB | Construct FSL array, target is FSB, verify correct output + zero-copy (same buffer address) |
| FSB → FSL | Reverse direction |
| Mixed columns | Some columns need conversion, others don't |
| Incompatible types | Build returns TypeError |
| Table with multiple chunks | ChunkedArray with multiple chunks converted correctly |
| End-to-end | Write FSB data via Vortex → read back through ColumnGroupReader → verify FSB output |

## File Change Summary

| File | Operation |
|------|-----------|
| `cpp/include/milvus-storage/format/type_adapter.h` | **New** |
| `cpp/src/format/type_adapter.cpp` | **New** |
| `cpp/test/format/type_adapter_test.cpp` | **New** |
| `cpp/include/milvus-storage/format/format_reader.h` | Add `output_schema()` pure virtual |
| `cpp/include/milvus-storage/format/parquet/parquet_format_reader.h` | Add `output_schema()` override |
| `cpp/src/format/parquet/parquet_format_reader.cpp` | Implement `output_schema()` |
| `cpp/include/milvus-storage/format/vortex/vortex_format_reader.h` | Add `output_schema()` + `physical_schema_` |
| `cpp/src/format/vortex/vortex_format_reader.cpp` | Fetch physical schema in `open()`, implement `output_schema()` |
| `cpp/include/milvus-storage/format/lance/lance_table_reader.h` | Add `output_schema()` + `physical_schema_` |
| `cpp/src/format/lance/lance_table_reader.cpp` | Fetch physical schema in `open()`, implement `output_schema()` |
| `cpp/src/format/column_group_reader.cpp` | Integrate TypeAdapter |
| `cpp/src/format/bridge/rust/src/vortex_bridgeimpl.rs` | Add `schema()` FFI method |
| `cpp/src/format/bridge/rust/src/lance_bridgeimpl.rs` | Add `get_fragment_schema()` |
| `cpp/src/format/bridge/rust/src/lib.rs` | Add FFI declarations for both |
| `cpp/src/format/bridge/rust/include/vortex_bridge.h` | Add `VortexFile::GetSchema()` |
| `cpp/src/format/bridge/rust/include/lance_bridge.h` | Add `BlockingDataset::GetFragmentSchema()` |
| `cpp/src/format/bridge/rust/src/vortex_bridge.cpp` | C++ side implementation |
| `cpp/src/format/bridge/rust/src/lance_bridge.cpp` | C++ side implementation |
