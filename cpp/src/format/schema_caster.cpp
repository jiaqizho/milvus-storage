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

#include "milvus-storage/format/schema_caster.h"

#include <fmt/format.h>

namespace milvus_storage {

arrow::Result<SchemaCaster> SchemaCaster::Build(std::shared_ptr<arrow::Schema> target,
                                                const std::shared_ptr<arrow::Schema>& source) {
  SchemaCaster caster;
  caster.target_ = std::move(target);

  const auto& rules = FieldCaster::RegisteredCasters();
  for (int i = 0; i < caster.target_->num_fields(); ++i) {
    const auto& src = source->field(i)->type();
    const auto& dst = caster.target_->field(i)->type();
    if (src->Equals(dst))
      continue;

    std::shared_ptr<FieldCaster> matched;
    for (const auto& rule : rules) {
      if (rule->source_id() != src->id() || rule->target_id() != dst->id())
        continue;
      if (rule->Match(*src, *dst)) {
        matched = rule;
        break;
      }
    }

    if (!matched) {
      return arrow::Status::TypeError(fmt::format("No type conversion for column '{}': {} -> {}",
                                                  source->field(i)->name(), src->ToString(), dst->ToString()));
    }
    caster.plan_.push_back({i, matched, dst});
  }

  return caster;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> SchemaCaster::operator()(
    const std::shared_ptr<arrow::RecordBatch>& batch) const {
  if (plan_.empty())
    return batch;

  auto columns = batch->columns();
  for (const auto& [col, rule, type] : plan_) {
    ARROW_ASSIGN_OR_RAISE(columns[col], rule->Convert(columns[col], type));
  }
  return arrow::RecordBatch::Make(target_, batch->num_rows(), std::move(columns));
}

arrow::Result<std::shared_ptr<arrow::Table>> SchemaCaster::operator()(
    const std::shared_ptr<arrow::Table>& table) const {
  if (plan_.empty())
    return table;

  auto columns = table->columns();
  for (const auto& [col, rule, type] : plan_) {
    arrow::ArrayVector chunks;
    chunks.reserve(columns[col]->num_chunks());
    for (const auto& chunk : columns[col]->chunks()) {
      ARROW_ASSIGN_OR_RAISE(auto converted, rule->Convert(chunk, type));
      chunks.push_back(std::move(converted));
    }
    columns[col] = std::make_shared<arrow::ChunkedArray>(std::move(chunks), type);
  }
  return arrow::Table::Make(target_, std::move(columns));
}

}  // namespace milvus_storage
