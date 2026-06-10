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

#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/util/thread_pool.h>
#include <folly/futures/Future.h>
#include <folly/futures/Promise.h>

namespace milvus_storage::api {

using RecordBatchVector = std::vector<std::shared_ptr<arrow::RecordBatch>>;

struct AsyncReadOptions {
  size_t read_parallelism = 1;
  arrow::internal::Executor* materialize_executor = nullptr;
};

inline arrow::internal::Executor* get_materialize_executor(const AsyncReadOptions& options) {
  if (options.materialize_executor != nullptr) {
    return options.materialize_executor;
  }
  return arrow::internal::GetCpuThreadPool();
}

template <typename Result, typename Func>
folly::SemiFuture<Result> submit_to_materialize_executor(const AsyncReadOptions& options, Func&& func) {
  using TaskFunc = std::decay_t<Func>;
  auto task = std::make_shared<TaskFunc>(std::forward<Func>(func));
  auto* executor = get_materialize_executor(options);

  if (executor->OwnsThisThread()) {
    try {
      return folly::makeSemiFuture(std::invoke(*task));
    } catch (...) {
      return folly::makeSemiFuture<Result>(folly::exception_wrapper(std::current_exception()));
    }
  }

  folly::Promise<Result> promise;
  auto future = promise.getSemiFuture();
  auto shared_promise = std::make_shared<folly::Promise<Result>>(std::move(promise));
  auto status = executor->Spawn([shared_promise, task]() mutable {
    try {
      shared_promise->setValue(std::invoke(*task));
    } catch (...) {
      shared_promise->setException(folly::exception_wrapper(std::current_exception()));
    }
  });
  if (!status.ok()) {
    shared_promise->setValue(Result(std::move(status)));
  }
  return future;
}

}  // namespace milvus_storage::api
