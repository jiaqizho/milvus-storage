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

#include "milvus-storage/format/field_caster.h"

#include <arrow/array/array_binary.h>
#include <arrow/array/array_nested.h>
#include <arrow/array/data.h>
#include <arrow/type.h>

namespace milvus_storage {

// CRTP helper: handles type ID, downcast, and delegates to Derived with typed params.
template <typename Derived, typename From, typename To>
struct TypedFieldCaster : FieldCaster {
  arrow::Type::type source_id() const final { return From::type_id; }
  arrow::Type::type target_id() const final { return To::type_id; }

  bool Match(const arrow::DataType& from, const arrow::DataType& to) const final {
    return static_cast<const Derived&>(*this).Match(static_cast<const From&>(from), static_cast<const To&>(to));
  }

  arrow::Result<std::shared_ptr<arrow::Array>> Convert(
      const std::shared_ptr<arrow::Array>& source, const std::shared_ptr<arrow::DataType>& target_type) const final {
    return static_cast<const Derived&>(*this).Convert(source, std::static_pointer_cast<To>(target_type));
  }
};

// Zero-copy: FixedSizeList<T, N> -> FixedSizeBinary(N * sizeof(T))
struct FixedSizeListToFixedSizeBinary
    : TypedFieldCaster<FixedSizeListToFixedSizeBinary, arrow::FixedSizeListType, arrow::FixedSizeBinaryType> {
  bool Match(const arrow::FixedSizeListType& from, const arrow::FixedSizeBinaryType& to) const {
    if (!arrow::is_fixed_width(*from.value_type()))
      return false;
    // NOTE: We intentionally do NOT check from.value_field()->nullable() here.
    //
    // Some storage formats (e.g., Lance) lose the child field's nullable=false metadata
    // during schema serialization — Lance encodes FixedSizeList's logical type as a string
    // like "fixed_size_list:uint8:16" which doesn't carry child nullability, so it always
    // reads back as nullable=true even when the original schema was nullable=false.
    // (See lance-core/src/datatypes.rs, ArrowField::new("item", elem_type, true))
    //
    // Since the schema-level nullable flag is unreliable, we defer the safety check to
    // Convert() where we verify the actual data has no nulls (null_count == 0).
    // Per the Arrow spec, null slots have undefined values in the data buffer (no zero-padding
    // guarantee), so reinterpreting as FSB would produce garbage — the runtime check prevents this.
    return from.list_size() * (from.value_type()->bit_width() / 8) == to.byte_width();
  }

  arrow::Result<std::shared_ptr<arrow::Array>> Convert(
      const std::shared_ptr<arrow::Array>& source,
      const std::shared_ptr<arrow::FixedSizeBinaryType>& target_type) const {
    auto& fsl_array = static_cast<const arrow::FixedSizeListArray&>(*source);
    auto values = fsl_array.values();

    // Runtime null check: FSB has no per-element null concept. If the child array contains
    // actual nulls, the data buffer at null positions is undefined per the Arrow spec (no
    // zero-padding guarantee), so reinterpreting as FSB would silently produce garbage data.
    if (values->null_count() > 0) {
      return arrow::Status::Invalid(
          "Cannot convert FixedSizeList to FixedSizeBinary: child array contains null elements. "
          "FixedSizeBinary has no per-element null representation, and Arrow does not guarantee "
          "zero-padding at null positions, so the conversion would produce undefined data.");
    }

    // Arrow fixed-width ArrayData layout:
    //   buffers[0] = null bitmap (optional)
    //   buffers[1] = contiguous data values
    // FixedSizeList's child values buffer has the same memory layout as
    // FixedSizeBinary's data buffer when total byte widths match, so we
    // reuse buffers[1] from the child array as buffers[1] of the new
    // FixedSizeBinary array (zero-copy).
    if (!values || !values->data()) {
      return arrow::Status::Invalid("FixedSizeList values array has no data");
    }
    auto& values_data = *values->data();
    if (values_data.buffers.size() < 2 || !values_data.buffers[1]) {
      return arrow::Status::Invalid("FixedSizeList values array has no data buffer");
    }

    auto values_buffer = values_data.buffers[1];

    auto data = arrow::ArrayData::Make(std::static_pointer_cast<arrow::DataType>(target_type), source->length(),
                                       {source->null_bitmap(), values_buffer}, source->null_count(), source->offset());
    return arrow::MakeArray(data);
  }
};

// Zero-copy: FixedSizeBinary(N * sizeof(T)) -> FixedSizeList<T, N>
struct FixedSizeBinaryToFixedSizeList
    : TypedFieldCaster<FixedSizeBinaryToFixedSizeList, arrow::FixedSizeBinaryType, arrow::FixedSizeListType> {
  bool Match(const arrow::FixedSizeBinaryType& from, const arrow::FixedSizeListType& to) const {
    if (!arrow::is_fixed_width(*to.value_type()))
      return false;
    return to.list_size() * (to.value_type()->bit_width() / 8) == from.byte_width();
  }

  arrow::Result<std::shared_ptr<arrow::Array>> Convert(
      const std::shared_ptr<arrow::Array>& source, const std::shared_ptr<arrow::FixedSizeListType>& target_type) const {
    if (!source->data()) {
      return arrow::Status::Invalid("FixedSizeBinary array has no data");
    }

    // Arrow fixed-width ArrayData layout:
    //   buffers[0] = null bitmap (optional)
    //   buffers[1] = contiguous data values
    // FixedSizeBinary's data buffer has the same memory layout as
    // FixedSizeList's child values buffer when total byte widths match.
    // We reuse buffers[1] as the child array's data buffer (zero-copy),
    // then wrap it in a FixedSizeList.
    auto& source_data = *source->data();
    if (source_data.buffers.size() < 2 || !source_data.buffers[1]) {
      return arrow::Status::Invalid("FixedSizeBinary array has no data buffer");
    }
    auto values_buffer = source_data.buffers[1];
    auto child_type = target_type->value_type();
    int32_t list_size = target_type->list_size();

    // Build child array from the same buffer (zero-copy)
    auto child_data = arrow::ArrayData::Make(child_type, source->length() * list_size, {nullptr, values_buffer});

    // Construct FixedSizeList ArrayData directly to preserve target_type's
    // child field nullability (FromArrays would default to nullable=true).
    auto fsl_data = arrow::ArrayData::Make(
        std::static_pointer_cast<arrow::DataType>(target_type), source->length(),
        {source->null_bitmap_data() ? source->null_bitmap() : nullptr}, {child_data}, source->null_count());
    return arrow::MakeArray(fsl_data);
  }
};

// Zero-copy: FixedSizeList<M, N> -> FixedSizeList<A, B> where sizeof(M)*N == sizeof(A)*B
// Both child fields must be fixed-width with matching total byte width.
struct FixedSizeListToFixedSizeList
    : TypedFieldCaster<FixedSizeListToFixedSizeList, arrow::FixedSizeListType, arrow::FixedSizeListType> {
  bool Match(const arrow::FixedSizeListType& from, const arrow::FixedSizeListType& to) const {
    if (!arrow::is_fixed_width(*from.value_type()) || !arrow::is_fixed_width(*to.value_type()))
      return false;
    // NOTE: We intentionally do NOT check child field nullable() here.
    // See FixedSizeListToFixedSizeBinary::Match for the full rationale — Lance and other
    // formats may report nullable=true on child fields even when no nulls exist in the data.
    // The runtime null check in Convert() provides the actual safety guarantee.
    return from.list_size() * (from.value_type()->bit_width() / 8) ==
           to.list_size() * (to.value_type()->bit_width() / 8);
  }

  arrow::Result<std::shared_ptr<arrow::Array>> Convert(
      const std::shared_ptr<arrow::Array>& source, const std::shared_ptr<arrow::FixedSizeListType>& target_type) const {
    auto& fsl_array = static_cast<const arrow::FixedSizeListArray&>(*source);
    auto values = fsl_array.values();

    // Runtime null check: this conversion reinterprets the raw child buffer as a different
    // element type. If the child array contains actual nulls, the data at null positions is
    // undefined per the Arrow spec, so reinterpretation would produce garbage.
    if (values->null_count() > 0) {
      return arrow::Status::Invalid(
          "Cannot reinterpret FixedSizeList child buffer: child array contains null elements. "
          "Arrow does not guarantee zero-padding at null positions, so the byte reinterpretation "
          "would produce undefined data.");
    }

    // Arrow FixedSizeList ArrayData layout:
    //   buffers[0] = null bitmap (optional, row-level nulls)
    //   child_data[0] = child array (contiguous fixed-width elements)
    // Both source and target have identical total byte width per row,
    // so we reinterpret the child's data buffer as the target child type.
    if (!values || !values->data()) {
      return arrow::Status::Invalid("FixedSizeList values array has no data");
    }
    auto& values_data = *values->data();
    if (values_data.buffers.size() < 2 || !values_data.buffers[1]) {
      return arrow::Status::Invalid("FixedSizeList values array has no data buffer");
    }

    auto values_buffer = values_data.buffers[1];
    auto child_type = target_type->value_type();
    int32_t target_list_size = target_type->list_size();

    auto child_data =
        arrow::ArrayData::Make(child_type, source->length() * target_list_size, {nullptr, values_buffer});

    auto fsl_data = arrow::ArrayData::Make(
        std::static_pointer_cast<arrow::DataType>(target_type), source->length(),
        {source->null_bitmap_data() ? source->null_bitmap() : nullptr}, {child_data}, source->null_count());
    return arrow::MakeArray(fsl_data);
  }
};

// Registry — add new rules here.
const std::vector<std::shared_ptr<FieldCaster>>& FieldCaster::RegisteredCasters() {
  static std::vector<std::shared_ptr<FieldCaster>> rules = {
      // FixedSizeList<T, N> -> FixedSizeBinary(N * sizeof(T)), child must be fixed-width, null checked at runtime
      std::make_shared<FixedSizeListToFixedSizeBinary>(),
      // FixedSizeBinary(N * sizeof(T)) -> FixedSizeList<T, N>, target child must be fixed-width
      std::make_shared<FixedSizeBinaryToFixedSizeList>(),
      // FixedSizeList<M, N> -> FixedSizeList<A, B>, sizeof(M)*N == sizeof(A)*B, null checked at runtime
      std::make_shared<FixedSizeListToFixedSizeList>(),
  };
  return rules;
}

}  // namespace milvus_storage
