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

#pragma once

#include <memory>
#include <vector>

#include <arrow/chunked_array.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/status.h>
#include <arrow/result.h>

#include "milvus-storage/format/field_caster.h"

namespace milvus_storage {

/// Pre-computed, format-agnostic type conversion plan.
///
///   auto adapter = SchemaCaster::Build(target_schema, source_schema);
///   ARROW_ASSIGN_OR_RAISE(rb, adapter(rb));  // passthrough when no conversion needed
///
class SchemaCaster {
  public:
  static arrow::Result<SchemaCaster> Build(std::shared_ptr<arrow::Schema> target,
                                           const std::shared_ptr<arrow::Schema>& source);

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> operator()(const std::shared_ptr<arrow::RecordBatch>& batch) const;

  arrow::Result<std::shared_ptr<arrow::Table>> operator()(const std::shared_ptr<arrow::Table>& table) const;

  private:
  struct Step {
    int col;
    std::shared_ptr<FieldCaster> rule;
    std::shared_ptr<arrow::DataType> target_type;
  };

  std::shared_ptr<arrow::Schema> target_;
  std::vector<Step> plan_;
};

}  // namespace milvus_storage
