if(ARROW_S3)
  find_package(AWSSDK REQUIRED)
  if(NOT DEFINED AWSSDK_LINK_LIBRARIES)
    set(AWSSDK_LINK_LIBRARIES "${AWSSDK_LIBRARIES}")
  endif()
  set(AWSSDK_SOURCE "conan")
endif()

if(ARROW_WITH_OPENTELEMETRY)
  find_package(Protobuf CONFIG REQUIRED)
  find_package(opentelemetry-cpp CONFIG REQUIRED)
  # Parquet and Dataset object targets also include tracing_internal.h.
  link_libraries(opentelemetry-cpp::opentelemetry_common)
  get_target_property(arrow_otel_protobuf_includes protobuf::libprotobuf
                      INTERFACE_INCLUDE_DIRECTORIES)
  include_directories(SYSTEM ${arrow_otel_protobuf_includes})
  # Arrow 17 uses upstream target names; Conan exposes the library names.
  set(arrow_otel_aliases
      api:opentelemetry_common
      trace:opentelemetry_trace
      logs:opentelemetry_logs
      otlp_http_log_record_exporter:opentelemetry_exporter_otlp_http_log
      ostream_log_record_exporter:opentelemetry_exporter_ostream_logs
      ostream_span_exporter:opentelemetry_exporter_ostream_span
      otlp_http_exporter:opentelemetry_exporter_otlp_http)
  foreach(pair IN LISTS arrow_otel_aliases)
    string(REPLACE ":" ";" pair "${pair}")
    list(GET pair 0 alias)
    list(GET pair 1 target)
    if(NOT TARGET opentelemetry-cpp::${alias})
      add_library(opentelemetry-cpp::${alias} INTERFACE IMPORTED)
      target_link_libraries(opentelemetry-cpp::${alias}
                            INTERFACE opentelemetry-cpp::${target})
      target_include_directories(opentelemetry-cpp::${alias}
                                 INTERFACE ${opentelemetry-cpp_INCLUDE_DIRS})
    endif()
  endforeach()
  # Keep Arrow's own instrumentation off. Consumers reuse the exported Span
  # helpers under their own provider, context propagation and sampling policy.
  add_compile_definitions(OPENTELEMETRY_STL_VERSION=2017 ARROW_TRACING_API_ONLY)
endif()
