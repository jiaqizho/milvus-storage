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

#include <arrow/status.h>
#include <folly/executors/ManualExecutor.h>

#include <memory>
#include <string>
#include <vector>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/format/async_tasks.h"
#include "milvus-storage/format/column_group_reader.h"

namespace milvus_storage::test {

using namespace milvus_storage::api;

namespace {

arrow::Result<int> maybe_int(bool ok) {
  if (!ok) {
    return arrow::Status::Invalid("bad int");
  }
  return 41;
}

folly::SemiFuture<arrow::Result<int>> return_not_ok_future(bool ok) {
  FOLLY_ARROW_RETURN_NOT_OK(ok ? arrow::Status::OK() : arrow::Status::Invalid("bad status"));
  return folly::makeSemiFuture(arrow::Result<int>(42));
}

folly::SemiFuture<arrow::Result<int>> assign_or_raise_future(bool ok) {
  FOLLY_ARROW_ASSIGN_OR_RAISE(auto value, maybe_int(ok));
  return folly::makeSemiFuture(arrow::Result<int>(value + 1));
}

folly::SemiFuture<arrow::Status> return_not_ok_status_future(bool ok) {
  FOLLY_ARROW_RETURN_NOT_OK(ok ? arrow::Status::OK() : arrow::Status::Invalid("bad status future"));
  return folly::makeSemiFuture(arrow::Status::OK());
}

folly::SemiFuture<arrow::Status> assign_or_raise_status_future(bool ok) {
  FOLLY_ARROW_ASSIGN_OR_RAISE(auto value, maybe_int(ok));
  return folly::makeSemiFuture(value == 41 ? arrow::Status::OK() : arrow::Status::UnknownError("bad value"));
}

}  // namespace

TEST(AsyncTasksTest, FollyArrowReturnNotOkPropagatesStatusToSemiFutureResult) {
  auto result = std::move(return_not_ok_future(false)).get();

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().ToString().find("bad status") != std::string::npos);
}

TEST(AsyncTasksTest, FollyArrowAssignOrRaisePropagatesResultStatusToSemiFutureResult) {
  auto result = std::move(assign_or_raise_future(false)).get();

  ASSERT_FALSE(result.ok());
  EXPECT_TRUE(result.status().ToString().find("bad int") != std::string::npos);
}

TEST(AsyncTasksTest, FollyArrowAssignOrRaiseExposesAssignedValueOnSuccess) {
  auto result = std::move(assign_or_raise_future(true)).get();

  ASSERT_TRUE(result.ok()) << result.status().ToString();
  EXPECT_EQ(result.ValueUnsafe(), 42);
}

TEST(AsyncTasksTest, FollyArrowReturnNotOkPropagatesStatusToStatusSemiFuture) {
  auto status = std::move(return_not_ok_status_future(false)).get();

  ASSERT_FALSE(status.ok());
  EXPECT_NE(status.ToString().find("bad status future"), std::string::npos);
}

TEST(AsyncTasksTest, FollyArrowAssignOrRaisePropagatesResultStatusToStatusSemiFuture) {
  auto status = std::move(assign_or_raise_status_future(false)).get();

  ASSERT_FALSE(status.ok());
  EXPECT_NE(status.ToString().find("bad int"), std::string::npos);
}

TEST(AsyncTasksTest, FollyArrowAssignOrRaiseSupportsSuccessfulStatusSemiFuture) {
  auto status = std::move(assign_or_raise_status_future(true)).get();

  EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(AsyncTasksTest, CancellationWaitsForChildrenAndPreservesReadErrors) {
  for (const int first_result : {0, 1, 2}) {
    for (const bool fail : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "first=" << first_result << " fail=" << fail);
      folly::CancellationSource cancellation;
      folly::Promise<arrow::Result<int>> first;
      folly::Promise<arrow::Result<int>> last;
      std::vector<folly::SemiFuture<arrow::Result<int>>> children;
      children.push_back(first.getSemiFuture());
      children.push_back(last.getSemiFuture());
      bool mapped = false;
      folly::ManualExecutor executor;
      auto future = CollectReadTasks(cancellation, std::move(children), [&mapped](auto) -> arrow::Result<int> {
                      mapped = true;
                      return 42;
                    }).via(&executor);
      executor.drain();
      future.cancel();
      EXPECT_TRUE(cancellation.isCancellationRequested());
      if (first_result == 0) {
        first.setValue(1);
      } else if (first_result == 1) {
        first.setValue(arrow::Status::Cancelled("child cancelled"));
      } else {
        first.setException(folly::FutureCancellation());
      }
      executor.drain();
      EXPECT_FALSE(future.isReady());
      const auto failure = arrow::Status::IOError("actual read failure");
      last.setValue(fail ? arrow::Result<int>(failure) : arrow::Result<int>(2));
      executor.drain();
      ASSERT_TRUE(future.isReady());
      EXPECT_FALSE(future.hasException());
      const auto result = std::move(future).get();
      if (fail) {
        EXPECT_EQ(result.status(), failure);
      } else {
        EXPECT_TRUE(result.status().IsCancelled()) << result.status();
      }
      EXPECT_FALSE(mapped);
    }
  }
}

TEST(AsyncTasksTest, ReadFanInCompletesAfterAbandonmentAndIgnoresLateCancel) {
  for (const bool abandon : {false, true}) {
    folly::Promise<arrow::Result<int>> pending;
    std::vector<folly::SemiFuture<arrow::Result<int>>> children;
    children.push_back(pending.getSemiFuture());
    auto owner = std::make_shared<int>(7);
    const std::weak_ptr<int> weak = owner;
    folly::ManualExecutor executor;
    auto future =
        CollectReadTasks(folly::CancellationSource{}, std::move(children), [owner](auto results) -> arrow::Result<int> {
          return *owner + results[0].value().ValueOrDie();
        }).via(&executor);
    owner.reset();
    executor.drain();
    if (abandon) {
      future = folly::Future<arrow::Result<int>>::makeEmpty();
    }
    EXPECT_FALSE(weak.expired());
    pending.setValue(5);
    executor.drain();
    if (!abandon) {
      ASSERT_TRUE(future.isReady());
      future.cancel();
      const auto result = std::move(future).get();
      ASSERT_TRUE(result.ok()) << result.status();
      EXPECT_EQ(*result, 12);
    }
    EXPECT_TRUE(weak.expired());
  }
}

TEST(AsyncTasksTest, PreStartCancellationPreservesReadyErrorsAndWaitsForStartedChildren) {
  folly::CancellationSource cancellation;
  const auto failure = arrow::Status::IOError("already failed before fan-in");
  folly::Promise<arrow::Result<int>> pending;
  std::vector<folly::SemiFuture<arrow::Result<int>>> children;
  children.push_back(StartReadTask(cancellation.getToken(), folly::makeSemiFuture(arrow::Result<int>(failure))));
  children.push_back(pending.getSemiFuture());
  folly::ManualExecutor executor;
  auto future = CollectReadTasks(std::move(cancellation), std::move(children), [](auto) -> arrow::Result<int> {
                  ADD_FAILURE() << "Cancellation and errors must skip result assembly";
                  return 0;
                }).via(&executor);
  future.cancel();
  executor.drain();
  EXPECT_FALSE(future.isReady());
  pending.setValue(1);
  executor.drain();
  ASSERT_TRUE(future.isReady());
  EXPECT_EQ(std::move(future).get().status(), failure);
}

TEST(AsyncTasksTest, ChunkTaskBuildGroupsByFileAndMergesContiguousRanges) {
  std::vector<ChunkInfo> chunk_infos = {
      {.file_index = 0, .row_offset_in_file = 0, .number_of_rows = 10},
      {.file_index = 0, .row_offset_in_file = 10, .number_of_rows = 10},
      {.file_index = 1, .row_offset_in_file = 0, .number_of_rows = 7},
      {.file_index = 0, .row_offset_in_file = 30, .number_of_rows = 5},
  };
  auto get_chunk_info = [&chunk_infos](int64_t chunk_index) -> const ChunkInfo& { return chunk_infos[chunk_index]; };

  auto tasks = ChunkTask::Build({0, 1, 2, 3}, get_chunk_info);

  ASSERT_EQ(tasks.size(), 3);
  EXPECT_EQ(tasks[0].file_index, 0);
  EXPECT_EQ(tasks[0].chunk_indices, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(tasks[0].range_start, 0);
  EXPECT_EQ(tasks[0].range_end, 20);

  EXPECT_EQ(tasks[1].file_index, 0);
  EXPECT_EQ(tasks[1].chunk_indices, (std::vector<int64_t>{3}));
  EXPECT_EQ(tasks[1].range_start, 30);
  EXPECT_EQ(tasks[1].range_end, 35);

  EXPECT_EQ(tasks[2].file_index, 1);
  EXPECT_EQ(tasks[2].chunk_indices, (std::vector<int64_t>{2}));
  EXPECT_EQ(tasks[2].range_start, 0);
  EXPECT_EQ(tasks[2].range_end, 7);
}

TEST(AsyncTasksTest, TakeTaskBuildGroupsRowsByColumnGroupAndFile) {
  auto group = std::make_shared<ColumnGroup>();
  group->files = {
      {.path = "file0", .start_index = 10, .end_index = 13},
      {.path = "file1", .start_index = 20, .end_index = 25},
  };

  auto result = TakeTask::Build({group}, {0, 2, 3, 6});
  ASSERT_TRUE(result.ok()) << result.status().ToString();

  const auto& tasks = result.ValueUnsafe();
  ASSERT_EQ(tasks.size(), 2);

  EXPECT_EQ(tasks[0].reader_index, 0);
  EXPECT_EQ(tasks[0].file_index, 0);
  EXPECT_EQ(tasks[0].row_indices, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(tasks[0].original_positions, (std::vector<size_t>{0, 1}));

  EXPECT_EQ(tasks[1].reader_index, 0);
  EXPECT_EQ(tasks[1].file_index, 1);
  EXPECT_EQ(tasks[1].row_indices, (std::vector<int64_t>{3, 6}));
  EXPECT_EQ(tasks[1].original_positions, (std::vector<size_t>{2, 3}));
}

TEST(AsyncTasksTest, TakeTaskBuildRejectsInvalidRows) {
  auto group = std::make_shared<ColumnGroup>();
  group->files = {
      {.path = "file0", .start_index = 0, .end_index = 2},
  };

  EXPECT_FALSE(TakeTask::Build({group}, {-1}).ok());
  EXPECT_FALSE(TakeTask::Build({group}, {1, 1}).ok());
  EXPECT_FALSE(TakeTask::Build({group}, {2}).ok());
}

TEST(AsyncTasksTest, TakeTaskBuildAddsReaderIndex) {
  auto group0 = std::make_shared<ColumnGroup>();
  group0->files = {
      {.path = "group0-file0", .start_index = 0, .end_index = 2},
      {.path = "group0-file1", .start_index = 0, .end_index = 2},
  };

  auto group1 = std::make_shared<ColumnGroup>();
  group1->files = {
      {.path = "group1-file0", .start_index = 0, .end_index = 4},
  };

  auto result = TakeTask::Build({group0, group1}, {0, 2, 3});
  ASSERT_TRUE(result.ok()) << result.status().ToString();

  const auto& tasks = result.ValueUnsafe();
  ASSERT_EQ(tasks.size(), 3);

  EXPECT_EQ(tasks[0].reader_index, 0);
  EXPECT_EQ(tasks[0].file_index, 0);
  EXPECT_EQ(tasks[0].row_indices, (std::vector<int64_t>{0}));
  EXPECT_EQ(tasks[0].original_positions, (std::vector<size_t>{0}));

  EXPECT_EQ(tasks[1].reader_index, 0);
  EXPECT_EQ(tasks[1].file_index, 1);
  EXPECT_EQ(tasks[1].row_indices, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(tasks[1].original_positions, (std::vector<size_t>{1, 2}));

  EXPECT_EQ(tasks[2].reader_index, 1);
  EXPECT_EQ(tasks[2].file_index, 0);
  EXPECT_EQ(tasks[2].row_indices, (std::vector<int64_t>{0, 2, 3}));
  EXPECT_EQ(tasks[2].original_positions, (std::vector<size_t>{0, 1, 2}));
}

TEST(AsyncTasksTest, ChunkTaskNestedSplitTraitsSplitsChunkRange) {
  std::vector<ChunkInfo> chunk_infos = {
      {.file_index = 0, .row_offset_in_file = 0, .number_of_rows = 10},
      {.file_index = 0, .row_offset_in_file = 10, .number_of_rows = 10},
      {.file_index = 0, .row_offset_in_file = 20, .number_of_rows = 10},
      {.file_index = 0, .row_offset_in_file = 30, .number_of_rows = 10},
  };
  auto get_chunk_info = [&chunk_infos](int64_t chunk_index) -> const ChunkInfo& { return chunk_infos[chunk_index]; };
  ChunkTask task{
      .file_index = 0,
      .chunk_indices = {0, 1, 2, 3},
      .range_start = 0,
      .range_end = 40,
  };

  ChunkTask::SplitTraits traits{get_chunk_info};
  auto right = traits.split(task);

  EXPECT_EQ(task.chunk_indices, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(task.range_start, 0);
  EXPECT_EQ(task.range_end, 20);
  EXPECT_EQ(right.chunk_indices, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(right.range_start, 20);
  EXPECT_EQ(right.range_end, 40);
}

TEST(AsyncTasksTest, TakeTaskNestedSplitTraitsSplitsRowsAndPositionsTogether) {
  TakeTask task{
      .reader_index = 3,
      .file_index = 5,
      .row_indices = {0, 1, 2, 3},
      .original_positions = {10, 11, 12, 13},
  };

  TakeTask::SplitTraits traits;
  EXPECT_EQ(traits.size(task), 4);
  EXPECT_TRUE(traits.can_split(task));
  auto right = traits.split(task);

  EXPECT_EQ(task.reader_index, 3);
  EXPECT_EQ(task.file_index, 5);
  EXPECT_EQ(task.row_indices, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(task.original_positions, (std::vector<size_t>{10, 11}));

  EXPECT_EQ(right.reader_index, 3);
  EXPECT_EQ(right.file_index, 5);
  EXPECT_EQ(right.row_indices, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(right.original_positions, (std::vector<size_t>{12, 13}));
}

TEST(AsyncTasksTest, SplitToParallelismPreservesLargestFirstOrder) {
  std::vector<TakeTask> tasks = {{
      .reader_index = 0,
      .file_index = 0,
      .row_indices = {0, 1, 2, 3, 4, 5, 6, 7},
      .original_positions = {10, 11, 12, 13, 14, 15, 16, 17},
  }};

  SplitAsyncTasks(tasks, 3, TakeTask::SplitTraits{}, AsyncTaskSplitStrategy::kSplitToParallelism);

  ASSERT_EQ(tasks.size(), 3);
  EXPECT_EQ(tasks[0].row_indices, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(tasks[1].row_indices, (std::vector<int64_t>{4, 5, 6, 7}));
  EXPECT_EQ(tasks[2].row_indices, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(tasks[0].original_positions, (std::vector<size_t>{10, 11}));
  EXPECT_EQ(tasks[1].original_positions, (std::vector<size_t>{14, 15, 16, 17}));
  EXPECT_EQ(tasks[2].original_positions, (std::vector<size_t>{12, 13}));
}

TEST(AsyncTasksTest, SplitAllCreatesOneTaskPerLogicalItem) {
  std::vector<TakeTask> tasks = {{
      .reader_index = 3,
      .file_index = 5,
      .row_indices = {0, 1, 2, 3, 4, 5, 6, 7},
      .original_positions = {10, 11, 12, 13, 14, 15, 16, 17},
  }};

  SplitAsyncTasks(tasks, 1, TakeTask::SplitTraits{}, AsyncTaskSplitStrategy::kSplitAll);

  ASSERT_EQ(tasks.size(), 8);
  std::vector<bool> seen(8, false);
  for (const auto& task : tasks) {
    ASSERT_EQ(task.row_indices.size(), 1);
    ASSERT_EQ(task.original_positions.size(), 1);
    ASSERT_GE(task.row_indices[0], 0);
    ASSERT_LT(task.row_indices[0], static_cast<int64_t>(seen.size()));
    EXPECT_EQ(task.original_positions[0], static_cast<size_t>(task.row_indices[0] + 10));
    EXPECT_FALSE(seen[task.row_indices[0]]);
    seen[task.row_indices[0]] = true;
  }
  for (bool item_seen : seen) {
    EXPECT_TRUE(item_seen);
  }
}

}  // namespace milvus_storage::test
