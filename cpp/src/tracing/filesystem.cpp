// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#include "tracing/filesystem.h"
#include <arrow/buffer.h>
#include <arrow/util/future.h>
#include <limits>
#include <mutex>
#include "milvus-storage/tracing.h"
#include "milvus-storage/filesystem/async_random_access_file.h"

namespace milvus_storage::tracing {
namespace {
int64_t Bytes(int64_t value) { return value; }
int64_t Bytes(const std::shared_ptr<arrow::Buffer>& buffer) { return buffer ? buffer->size() : 0; }

class TracedFile : public arrow::io::RandomAccessFile {
  public:
  TracedFile(std::shared_ptr<arrow::io::RandomAccessFile> file, std::string backend)
      : file_(std::move(file)), backend_(std::move(backend)) {}
  arrow::Status Close() override { return file_->Close(); }
  arrow::Future<> CloseAsync() override { return file_->CloseAsync(); }
  arrow::Status Abort() override { return file_->Abort(); }
  bool closed() const override { return file_->closed(); }
  arrow::Result<int64_t> Tell() const override { return file_->Tell(); }
  arrow::Status Seek(int64_t position) override { return file_->Seek(position); }
  bool supports_zero_copy() const override { return file_->supports_zero_copy(); }
  const arrow::io::IOContext& io_context() const override { return file_->io_context(); }
  arrow::Result<std::string_view> Peek(int64_t nbytes) override { return file_->Peek(nbytes); }
  arrow::Status WillNeed(const std::vector<arrow::io::ReadRange>& ranges) override { return file_->WillNeed(ranges); }
  arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override {
    if (!HasContext())
      return file_->ReadMetadata();
    auto scope = TraceIO("storage.fs.metadata");
    auto result = file_->ReadMetadata();
    scope.Finish(result.status());
    return result;
  }
  arrow::Future<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadataAsync(
      const arrow::io::IOContext& context) override {
    if (!HasContext())
      return file_->ReadMetadataAsync(context);
    auto scope = TraceIO("storage.fs.metadata");
    auto result = file_->ReadMetadataAsync(context);
    if (scope.span()) {
      result.AddCallback([span = scope.span(), context = scope.context()](
                             const arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>>& completed) {
        EndSpan(span, context, completed.status());
      });
    }
    scope.ReleaseSpan();
    return result;
  }
  arrow::Result<int64_t> GetSize() override {
    if (!HasContext())
      return file_->GetSize();
    auto scope = TraceIO("storage.fs.head");
    auto result = file_->GetSize();
    scope.Finish(result.status());
    return result;
  }
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    if (!HasContext())
      return file_->Read(nbytes, out);
    auto scope = TraceIO("storage.fs.read",
                         {{"storage.backend", backend_}, {"storage.offset", -1}, {"storage.requested_bytes", nbytes}});
    auto result = file_->Read(nbytes, out);
    RecordRead(scope.span(), scope.context(), nbytes, result.ok() ? Bytes(*result) : 0, result.status());
    return result;
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    if (!HasContext())
      return file_->Read(nbytes);
    auto scope = TraceIO("storage.fs.read",
                         {{"storage.backend", backend_}, {"storage.offset", -1}, {"storage.requested_bytes", nbytes}});
    auto result = file_->Read(nbytes);
    RecordRead(scope.span(), scope.context(), nbytes, result.ok() ? Bytes(*result) : 0, result.status());
    return result;
  }
  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    if (!HasContext())
      return file_->ReadAt(position, nbytes, out);
    auto scope =
        TraceIO("storage.fs.read",
                {{"storage.backend", backend_}, {"storage.offset", position}, {"storage.requested_bytes", nbytes}});
    auto result = file_->ReadAt(position, nbytes, out);
    RecordRead(scope.span(), scope.context(), nbytes, result.ok() ? Bytes(*result) : 0, result.status());
    return result;
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    if (!HasContext())
      return file_->ReadAt(position, nbytes);
    auto scope =
        TraceIO("storage.fs.read",
                {{"storage.backend", backend_}, {"storage.offset", position}, {"storage.requested_bytes", nbytes}});
    auto result = file_->ReadAt(position, nbytes);
    RecordRead(scope.span(), scope.context(), nbytes, result.ok() ? Bytes(*result) : 0, result.status());
    return result;
  }
  arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext& context,
                                                          int64_t position,
                                                          int64_t nbytes) override {
    if (!HasContext())
      return file_->ReadAsync(context, position, nbytes);
    auto scope =
        TraceIO("storage.fs.read",
                {{"storage.backend", backend_}, {"storage.offset", position}, {"storage.requested_bytes", nbytes}});
    auto result = file_->ReadAsync(context, position, nbytes);
    if (IsEnabled(scope.context())) {
      result.AddCallback([span = scope.span(), context = scope.context(),
                          nbytes](const arrow::Result<std::shared_ptr<arrow::Buffer>>& completed) {
        if (completed.ok())
          SetAttribute(span, context, "storage.short_read", static_cast<int64_t>(Bytes(*completed) < nbytes));
        RecordRead(span, context, nbytes, completed.ok() ? Bytes(*completed) : 0, completed.status());
      });
    }
    scope.ReleaseSpan();
    return result;
  }
  std::vector<arrow::Future<std::shared_ptr<arrow::Buffer>>> ReadManyAsync(
      const arrow::io::IOContext& context, const std::vector<arrow::io::ReadRange>& ranges) override {
    if (!HasContext())
      return file_->ReadManyAsync(context, ranges);
    auto scope = TraceIO("storage.fs.read",
                         {{"storage.backend", backend_}, {"storage.range_count", static_cast<int64_t>(ranges.size())}});
    const auto& trace = scope.span();
    const auto& trace_context = scope.context();
    if (!tracing::IsEnabled(trace_context))
      return file_->ReadManyAsync(context, ranges);
    int64_t requested = 0;
    for (const auto& range : ranges) {
      auto length = std::max<int64_t>(0, range.length);
      requested += std::min(length, std::numeric_limits<int64_t>::max() - requested);
    }
    tracing::SetAttribute(trace, trace_context, "storage.requested_bytes", requested);
    struct Completion {
      Completion(size_t count, SpanPtr trace, ContextPtr context)
          : remaining(count), trace(std::move(trace)), trace_context(std::move(context)) {}
      std::mutex mutex;
      size_t remaining;
      int64_t bytes = 0;
      arrow::Status status;
      SpanPtr trace;
      ContextPtr trace_context;
    };
    auto futures = file_->ReadManyAsync(context, ranges);
    if (futures.empty())
      tracing::EndSpan(trace, trace_context, arrow::Status::OK());
    auto completion = std::make_shared<Completion>(futures.size(), trace, trace_context);
    for (auto& future : futures) {
      future.AddCallback([completion, requested](const arrow::Result<std::shared_ptr<arrow::Buffer>>& result) {
        std::lock_guard<std::mutex> lock(completion->mutex);
        if (!result.ok())
          completion->status = result.status();
        else
          completion->bytes += std::min(Bytes(*result), std::numeric_limits<int64_t>::max() - completion->bytes);
        if (--completion->remaining == 0) {
          tracing::AccountRead(completion->trace_context, requested, completion->bytes);
          tracing::SetAttribute(completion->trace, completion->trace_context, "storage.returned_bytes",
                                completion->bytes);
          tracing::EndSpan(completion->trace, completion->trace_context, completion->status);
        }
      });
    }
    scope.ReleaseSpan();
    return futures;
  }

  protected:
  static void RecordRead(const SpanPtr& span,
                         const ContextPtr& context,
                         int64_t requested,
                         int64_t returned,
                         const arrow::Status& status) {
    AccountRead(context, requested, returned);
    if (status.ok()) {
      SetAttribute(span, context, "storage.returned_bytes", returned);
    }
    EndSpan(span, context, status);
  }

  std::shared_ptr<arrow::io::RandomAccessFile> file_;
  std::string backend_;
};

// Only expose this capability when the underlying file implements it. Never
// fall back to blocking I/O from a NonBlockingRandomAccessFile method.
class TracedAsyncFile final : public TracedFile, public NonBlockingRandomAccessFile {
  public:
  TracedAsyncFile(std::shared_ptr<arrow::io::RandomAccessFile> file, std::string backend)
      : TracedFile(std::move(file), std::move(backend)),
        async_(dynamic_cast<NonBlockingRandomAccessFile*>(file_.get())) {}
  arrow::Future<int64_t> ReadAtAsyncInto(int64_t position, int64_t nbytes, uint8_t* out) override {
    if (!HasContext())
      return async_->ReadAtAsyncInto(position, nbytes, out);
    auto scope =
        TraceIO("storage.fs.read",
                {{"storage.backend", backend_}, {"storage.offset", position}, {"storage.requested_bytes", nbytes}});
    auto result = async_->ReadAtAsyncInto(position, nbytes, out);
    if (IsEnabled(scope.context())) {
      result.AddCallback(
          [span = scope.span(), context = scope.context(), nbytes](const arrow::Result<int64_t>& completed) {
            if (completed.ok())
              SetAttribute(span, context, "storage.short_read", static_cast<int64_t>(Bytes(*completed) < nbytes));
            RecordRead(span, context, nbytes, completed.ok() ? Bytes(*completed) : 0, completed.status());
          });
    }
    scope.ReleaseSpan();
    return result;
  }
  arrow::Future<int64_t> GetSizeAsync() override {
    if (!HasContext())
      return async_->GetSizeAsync();
    auto scope = TraceIO("storage.fs.head");
    auto result = async_->GetSizeAsync();
    if (scope.span()) {
      result.AddCallback([span = scope.span(), context = scope.context()](const arrow::Result<int64_t>& completed) {
        EndSpan(span, context, completed.status());
      });
    }
    scope.ReleaseSpan();
    return result;
  }

  private:
  NonBlockingRandomAccessFile* async_;
};
}  // namespace
std::shared_ptr<arrow::io::RandomAccessFile> WrapFile(std::shared_ptr<arrow::io::RandomAccessFile> file,
                                                      std::string backend) {
  if (dynamic_cast<TracedFile*>(file.get()))
    return file;
  if (dynamic_cast<NonBlockingRandomAccessFile*>(file.get()))
    return std::make_shared<TracedAsyncFile>(std::move(file), std::move(backend));
  return std::make_shared<TracedFile>(std::move(file), std::move(backend));
}
}  // namespace milvus_storage::tracing
