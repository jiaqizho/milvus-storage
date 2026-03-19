// Copyright 2023 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <functional>
#include <unordered_map>

#include <arrow/api.h>
#include <arrow/array/array_binary.h>
#include <arrow/array/array_nested.h>
#include <arrow/array/array_primitive.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>

#include "milvus-storage/format/schema_caster.h"
#include "milvus-storage/format/format_reader.h"
#include "milvus-storage/writer.h"
#include "test_env.h"

namespace milvus_storage::test {

static constexpr int64_t kNumRows = 5;

// A test case for SchemaCaster conversion tests.
// When adding a new FieldCaster rule, add corresponding entries to GetCastTestCases().
struct CastTestCase {
  std::shared_ptr<arrow::DataType> source_type;
  std::shared_ptr<arrow::DataType> target_type;
  std::function<std::shared_ptr<arrow::Array>(int64_t num_rows)> make_source;
  // Formats that do NOT support this test case for FormatSchemaCasterTest.
  // Listed formats will be skipped with the given reason.
  std::unordered_map<std::string, std::string> unsupported_formats;
};

// Array factories

static std::shared_ptr<arrow::Array> MakeFSBArray(int32_t byte_width, int64_t num_rows) {
  arrow::FixedSizeBinaryBuilder builder(arrow::fixed_size_binary(byte_width));
  std::vector<uint8_t> value(byte_width);
  for (int64_t i = 0; i < num_rows; ++i) {
    for (int32_t j = 0; j < byte_width; ++j) {
      value[j] = static_cast<uint8_t>((i * byte_width + j) % 256);
    }
    EXPECT_TRUE(builder.Append(value.data()).ok());
  }
  return builder.Finish().ValueOrDie();
}

static std::shared_ptr<arrow::Array> MakeFSLArray(const std::shared_ptr<arrow::DataType>& child_type,
                                                  int32_t list_size,
                                                  int64_t num_rows) {
  int32_t elem_bytes = child_type->bit_width() / 8;
  int32_t total_bytes = list_size * elem_bytes;

  arrow::UInt8Builder raw_builder;
  for (int64_t i = 0; i < num_rows * total_bytes; ++i) {
    EXPECT_TRUE(raw_builder.Append(static_cast<uint8_t>(i % 256)).ok());
  }
  auto raw_buffer = raw_builder.Finish().ValueOrDie()->data()->buffers[1];

  auto child_data = arrow::ArrayData::Make(child_type, num_rows * list_size, {nullptr, raw_buffer});
  return arrow::FixedSizeListArray::FromArrays(arrow::MakeArray(child_data), list_size).ValueOrDie();
}

// Register test cases here.
// Each entry covers one (source_type, target_type) pair with an array factory.
// When a new FieldCaster rule is added, add entries here — no test logic changes needed.
static std::vector<CastTestCase> GetCastTestCases() {
  return {
      // FixedSizeList<u8, 16> -> FixedSizeBinary(16)
      {arrow::fixed_size_list(arrow::field("item", arrow::uint8(), /*nullable=*/false), 16),
       arrow::fixed_size_binary(16),
       [](int64_t n) { return MakeFSLArray(arrow::uint8(), 16, n); },
       // Parquet has no native FixedSizeList; it degrades to variable-length List,
       // so the physical schema on read is List<u8> instead of FixedSizeList<u8, 16>.
       {{LOON_FORMAT_PARQUET, "Parquet degrades FixedSizeList to variable-length List"}}},

      // FixedSizeBinary(16) -> FixedSizeList<u8, 16>
      {arrow::fixed_size_binary(16), arrow::fixed_size_list(arrow::field("item", arrow::uint8(), /*nullable=*/false), 16),
       [](int64_t n) { return MakeFSBArray(16, n); }},

      // FixedSizeList<float32, 4> -> FixedSizeBinary(16)
      {arrow::fixed_size_list(arrow::field("item", arrow::float32(), /*nullable=*/false), 4),
       arrow::fixed_size_binary(16),
       [](int64_t n) { return MakeFSLArray(arrow::float32(), 4, n); },
       {{LOON_FORMAT_PARQUET, "Parquet degrades FixedSizeList to variable-length List"}}},

      // FixedSizeBinary(16) -> FixedSizeList<float32, 4>
      {arrow::fixed_size_binary(16), arrow::fixed_size_list(arrow::field("item", arrow::float32(), /*nullable=*/false), 4),
       [](int64_t n) { return MakeFSBArray(16, n); }},

      // FixedSizeList<u32, 4> -> FixedSizeBinary(16)
      {arrow::fixed_size_list(arrow::field("item", arrow::uint32(), /*nullable=*/false), 4),
       arrow::fixed_size_binary(16),
       [](int64_t n) { return MakeFSLArray(arrow::uint32(), 4, n); },
       {{LOON_FORMAT_PARQUET, "Parquet degrades FixedSizeList to variable-length List"}}},

      // FixedSizeBinary(16) -> FixedSizeList<u32, 4>
      {arrow::fixed_size_binary(16), arrow::fixed_size_list(arrow::field("item", arrow::uint32(), /*nullable=*/false), 4),
       [](int64_t n) { return MakeFSBArray(16, n); }},

      // FixedSizeList<u8, 16> -> FixedSizeList<float32, 4>  (1*16 == 4*4)
      {arrow::fixed_size_list(arrow::field("item", arrow::uint8(), /*nullable=*/false), 16),
       arrow::fixed_size_list(arrow::field("item", arrow::float32(), /*nullable=*/false), 4),
       [](int64_t n) { return MakeFSLArray(arrow::uint8(), 16, n); },
       {{LOON_FORMAT_PARQUET, "Parquet degrades FixedSizeList to variable-length List"}}},

      // FixedSizeList<float32, 4> -> FixedSizeList<u8, 16>  (4*4 == 1*16)
      {arrow::fixed_size_list(arrow::field("item", arrow::float32(), /*nullable=*/false), 4),
       arrow::fixed_size_list(arrow::field("item", arrow::uint8(), /*nullable=*/false), 16),
       [](int64_t n) { return MakeFSLArray(arrow::float32(), 4, n); },
       {{LOON_FORMAT_PARQUET, "Parquet degrades FixedSizeList to variable-length List"}}},

      // FixedSizeList<u8, 16> -> FixedSizeList<u32, 4>  (1*16 == 4*4)
      {arrow::fixed_size_list(arrow::field("item", arrow::uint8(), /*nullable=*/false), 16),
       arrow::fixed_size_list(arrow::field("item", arrow::uint32(), /*nullable=*/false), 4),
       [](int64_t n) { return MakeFSLArray(arrow::uint8(), 16, n); },
       {{LOON_FORMAT_PARQUET, "Parquet degrades FixedSizeList to variable-length List"}}},
  };
}

// Parameterized tests — generic, rule-agnostic.

class SchemaCasterCastTest : public ::testing::TestWithParam<CastTestCase> {};

TEST_P(SchemaCasterCastTest, ConvertRecordBatch) {
  auto& tc = GetParam();
  auto source_schema = arrow::schema({arrow::field("col", tc.source_type)});
  auto target_schema = arrow::schema({arrow::field("col", tc.target_type)});

  ASSERT_AND_ASSIGN(auto caster, SchemaCaster::Build(target_schema, source_schema));

  auto source_array = tc.make_source(kNumRows);
  ASSERT_NE(source_array, nullptr);
  auto batch = arrow::RecordBatch::Make(source_schema, kNumRows, {source_array});

  ASSERT_AND_ASSIGN(auto result, caster(batch));
  ASSERT_TRUE(result->schema()->Equals(*target_schema));
  ASSERT_TRUE(result->column(0)->type()->Equals(*tc.target_type));
  ASSERT_EQ(result->num_rows(), kNumRows);
}

TEST_P(SchemaCasterCastTest, ConvertTable) {
  auto& tc = GetParam();
  auto source_schema = arrow::schema({arrow::field("col", tc.source_type)});
  auto target_schema = arrow::schema({arrow::field("col", tc.target_type)});

  ASSERT_AND_ASSIGN(auto caster, SchemaCaster::Build(target_schema, source_schema));

  auto chunk1 = tc.make_source(kNumRows);
  auto chunk2 = tc.make_source(kNumRows);
  auto chunked = std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{chunk1, chunk2}, tc.source_type);
  auto table = arrow::Table::Make(source_schema, {chunked});

  ASSERT_AND_ASSIGN(auto result, caster(table));
  ASSERT_TRUE(result->schema()->Equals(*target_schema));
  ASSERT_EQ(result->num_rows(), kNumRows * 2);
  ASSERT_EQ(result->column(0)->num_chunks(), 2);
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(result->column(0)->chunk(i)->type()->Equals(*tc.target_type));
  }
}

TEST(SchemaCasterTest, BasicTest) {
  // Passthrough: identical schemas, no conversion needed
  {
    auto schema = arrow::schema({
        arrow::field("id", arrow::int64()),
        arrow::field("name", arrow::utf8()),
    });

    ASSERT_AND_ASSIGN(auto caster, SchemaCaster::Build(schema, schema));

    arrow::Int64Builder id_builder;
    ASSERT_TRUE(id_builder.AppendValues({1, 2, 3}).ok());
    auto id_array = id_builder.Finish().ValueOrDie();

    arrow::StringBuilder name_builder;
    ASSERT_TRUE(name_builder.AppendValues({"a", "b", "c"}).ok());
    auto name_array = name_builder.Finish().ValueOrDie();

    auto batch = arrow::RecordBatch::Make(schema, 3, {id_array, name_array});
    ASSERT_AND_ASSIGN(auto result, caster(batch));
    ASSERT_EQ(result.get(), batch.get());
  }

  // Incompatible types: Build returns TypeError
  {
    auto source_schema = arrow::schema({arrow::field("x", arrow::int64())});
    auto target_schema = arrow::schema({arrow::field("x", arrow::utf8())});

    auto result = SchemaCaster::Build(target_schema, source_schema);
    ASSERT_FALSE(result.ok());
    ASSERT_TRUE(result.status().IsTypeError());
  }

  // RegisteredCasters is non-empty and valid
  {
    const auto& casters = FieldCaster::RegisteredCasters();
    ASSERT_GT(casters.size(), 0);
    for (const auto& caster : casters) {
      ASSERT_NE(caster, nullptr);
      // source_id and target_id may be equal (e.g., FSL->FSL with different params)
      ASSERT_NE(caster, nullptr);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(AllCasters, SchemaCasterCastTest, ::testing::ValuesIn(GetCastTestCases()));

// Write with each format, read back with SchemaCaster converting types.
// Parameterized on (format, CastTestCase) — covers all formats × all registered cast rules.
class FormatSchemaCasterTest : public ::testing::TestWithParam<std::tuple<std::string, CastTestCase>> {
  protected:
  void SetUp() override {
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    base_path_ = GetTestBasePath("format-schema-caster-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));
  }

  void TearDown() override { ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_)); }

  protected:
  api::Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_path_;
};

TEST_P(FormatSchemaCasterTest, SchemaCasterTest) {
  auto [format, tc] = GetParam();

  // Skip formats that don't support this test case
  auto it = tc.unsupported_formats.find(format);
  if (it != tc.unsupported_formats.end()) {
    GTEST_SKIP() << it->second;
  }

  constexpr int64_t kTestRows = 100;

  // Write schema uses source_type, read schema uses target_type
  auto write_schema = arrow::schema({arrow::field("col", tc.source_type)});
  auto read_schema = arrow::schema({arrow::field("col", tc.target_type)});

  // Write data with source_type
  auto source_array = tc.make_source(kTestRows);
  ASSERT_NE(source_array, nullptr);
  auto write_batch = arrow::RecordBatch::Make(write_schema, kTestRows, {source_array});

  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, write_schema));
  auto writer = api::Writer::create(base_path_, write_schema, std::move(policy), properties_);
  ASSERT_STATUS_OK(writer->write(write_batch));
  ASSERT_STATUS_OK(writer->flush());

  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();
  ASSERT_GT(cgs->size(), 0);
  ASSERT_GT((*cgs)[0]->files.size(), 0);

  // Read back using format reader with write_schema (matches physical data on disk).
  // The format reader should read in physical types; SchemaCaster handles the conversion
  // from physical to target types afterward.
  ASSERT_AND_ASSIGN(auto format_reader, FormatReader::create(write_schema, format, (*cgs)[0]->files[0], properties_,
                                                             std::vector<std::string>{"col"}, nullptr));

  // Build SchemaCaster: physical output -> target read schema
  auto physical_schema = format_reader->output_schema();
  ASSERT_NE(physical_schema, nullptr);
  ASSERT_AND_ASSIGN(auto caster, SchemaCaster::Build(read_schema, physical_schema));

  // Read all chunks, convert, and verify
  ASSERT_AND_ASSIGN(auto row_group_infos, format_reader->get_row_group_infos());
  ASSERT_GT(row_group_infos.size(), 0);

  int64_t total_rows = 0;
  for (size_t i = 0; i < row_group_infos.size(); ++i) {
    ASSERT_AND_ASSIGN(auto rb, format_reader->get_chunk(i));
    ASSERT_AND_ASSIGN(rb, caster(rb));

    ASSERT_TRUE(rb->column(0)->type()->Equals(*tc.target_type))
        << "Expected " << tc.target_type->ToString() << " but got " << rb->column(0)->type()->ToString();
    total_rows += rb->num_rows();
  }
  ASSERT_EQ(total_rows, kTestRows);
}

INSTANTIATE_TEST_SUITE_P(
    AllFormatsAndCasters,
    FormatSchemaCasterTest,
    ::testing::Combine(::testing::Values(LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX, LOON_FORMAT_LANCE_TABLE),
                       ::testing::ValuesIn(GetCastTestCases())));

// Diagnostic test: write various types, read back with write_schema as FormatReader schema,
// then compare output_schema() vs write_schema to see which types get changed by each format.
class OutputSchemaTest : public ::testing::TestWithParam<std::string> {
  protected:
  void SetUp() override {
    ASSERT_STATUS_OK(InitTestProperties(properties_));
    ASSERT_AND_ASSIGN(fs_, GetFileSystem(properties_));

    base_path_ = GetTestBasePath("output-schema-test");
    ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_));
    ASSERT_STATUS_OK(CreateTestDir(fs_, base_path_));
  }

  void TearDown() override { ASSERT_STATUS_OK(DeleteTestDir(fs_, base_path_)); }

  protected:
  api::Properties properties_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::string base_path_;
};

TEST_P(OutputSchemaTest, CompareOutputSchemaWithWriteSchema) {
  auto format = GetParam();
  constexpr int64_t kRows = 10;

  // Build a schema with various types: fixed-width, variable-length, list, nested
  auto write_schema = arrow::schema({
      arrow::field("f_int64", arrow::int64()),
      arrow::field("f_float32", arrow::float32()),
      arrow::field("f_utf8", arrow::utf8()),
      arrow::field("f_binary", arrow::binary()),
      arrow::field("f_fsb16", arrow::fixed_size_binary(16)),
      arrow::field("f_fsl_u8_16", arrow::fixed_size_list(arrow::field("item", arrow::uint8(), false), 16)),
      arrow::field("f_fsl_f32_4", arrow::fixed_size_list(arrow::field("item", arrow::float32(), false), 4)),
  });

  // Build arrays for each field
  // int64
  arrow::Int64Builder b_int64;
  for (int64_t i = 0; i < kRows; ++i) ASSERT_TRUE(b_int64.Append(i).ok());
  auto a_int64 = b_int64.Finish().ValueOrDie();

  // float32
  arrow::FloatBuilder b_float;
  for (int64_t i = 0; i < kRows; ++i) ASSERT_TRUE(b_float.Append(static_cast<float>(i)).ok());
  auto a_float = b_float.Finish().ValueOrDie();

  // utf8
  arrow::StringBuilder b_utf8;
  for (int64_t i = 0; i < kRows; ++i) ASSERT_TRUE(b_utf8.Append("hello" + std::to_string(i)).ok());
  auto a_utf8 = b_utf8.Finish().ValueOrDie();

  // binary
  arrow::BinaryBuilder b_bin;
  for (int64_t i = 0; i < kRows; ++i) ASSERT_TRUE(b_bin.Append("bin" + std::to_string(i)).ok());
  auto a_bin = b_bin.Finish().ValueOrDie();

  // fixed_size_binary(16)
  auto a_fsb16 = MakeFSBArray(16, kRows);

  // fixed_size_list<u8, 16>
  auto a_fsl_u8_16 = MakeFSLArray(arrow::uint8(), 16, kRows);

  // fixed_size_list<f32, 4>
  auto a_fsl_f32_4 = MakeFSLArray(arrow::float32(), 4, kRows);

  auto write_batch = arrow::RecordBatch::Make(write_schema, kRows,
                                               {a_int64, a_float, a_utf8, a_bin, a_fsb16, a_fsl_u8_16, a_fsl_f32_4});

  // Write
  ASSERT_AND_ASSIGN(auto policy, CreateSinglePolicy(format, write_schema));
  auto writer = api::Writer::create(base_path_, write_schema, std::move(policy), properties_);
  ASSERT_STATUS_OK(writer->write(write_batch));
  ASSERT_STATUS_OK(writer->flush());
  auto cgs_result = writer->close();
  ASSERT_TRUE(cgs_result.ok()) << cgs_result.status().ToString();
  auto cgs = std::move(cgs_result).ValueOrDie();
  ASSERT_GT(cgs->size(), 0);
  ASSERT_GT((*cgs)[0]->files.size(), 0);

  // Read back using write_schema as the FormatReader schema
  std::vector<std::string> all_cols;
  for (int i = 0; i < write_schema->num_fields(); ++i) {
    all_cols.push_back(write_schema->field(i)->name());
  }
  ASSERT_AND_ASSIGN(auto format_reader,
                     FormatReader::create(write_schema, format, (*cgs)[0]->files[0], properties_, all_cols, nullptr));

  auto output_schema = format_reader->output_schema();
  ASSERT_NE(output_schema, nullptr);

  std::cout << "\n=== Format: " << format << " ===" << std::endl;
  for (int i = 0; i < write_schema->num_fields(); ++i) {
    auto write_field = write_schema->field(i);
    auto out_field = output_schema->GetFieldByName(write_field->name());
    if (!out_field) {
      std::cout << "  " << write_field->name() << ": MISSING in output_schema" << std::endl;
      continue;
    }
    bool type_match = write_field->type()->Equals(*out_field->type());
    bool nullable_match = write_field->nullable() == out_field->nullable();
    std::cout << "  " << write_field->name() << ": write=" << write_field->type()->ToString()
              << "(nullable=" << write_field->nullable() << ")"
              << " -> output=" << out_field->type()->ToString() << "(nullable=" << out_field->nullable() << ")"
              << (type_match && nullable_match ? " [MATCH]" : " [DIFFER]") << std::endl;
  }
}

INSTANTIATE_TEST_SUITE_P(AllFormats, OutputSchemaTest,
                         ::testing::Values(LOON_FORMAT_PARQUET, LOON_FORMAT_VORTEX, LOON_FORMAT_LANCE_TABLE));

}  // namespace milvus_storage::test
