import os

from conan import ConanFile
from conan.tools.files import copy, get, patch, replace_in_file


class ArrowTracingConan(ConanFile):
    """Expose the pinned Arrow tracing implementation to Storage."""

    python_requires = "arrow/17.0.0@milvus/dev-2.6#c743ea7a6f2420ba5811b2be3df59892"
    python_requires_extend = "arrow.ArrowConan"
    exports_sources = "conan_cmake_project_include.cmake"

    def init(self):
        self.conan_data = self.python_requires["arrow"].conanfile.conan_data

    def export_sources(self):
        super().export_sources()
        copy(self, "tracing-api-only.patch", self.recipe_folder,
             os.path.join(self.export_sources_folder, "src"))

    def configure(self):
        super().configure()
        if self.options.with_opentelemetry:
            self.options["opentelemetry-cpp"].with_stl = True

    # Dependency list follows the pinned base recipe; OTel uses Storage's ABI.
    def requirements(self):
        if self.options.with_thrift:
            self.requires("thrift/0.17.0")
        if self.options.with_protobuf or self.options.with_opentelemetry:
            self.requires("protobuf/5.27.0@milvus/dev#42f031a96d21c230a6e05bcac4bdd633")
        if self.options.with_jemalloc:
            self.requires("jemalloc/5.3.0")
        if self.options.with_mimalloc:
            self.requires("mimalloc/1.7.6")
        if self.options.with_boost:
            self.requires("boost/1.85.0")
        if self.options.with_gflags:
            self.requires("gflags/2.2.2")
        if self.options.with_glog:
            self.requires("glog/0.6.0")
        if self.options.get_safe("with_gcs"):
            self.requires("google-cloud-cpp/1.40.1")
        if self.options.with_grpc:
            self.requires("grpc/1.50.0")
        if self._requires_rapidjson():
            self.requires("rapidjson/1.1.0")
        if self.options.with_llvm:
            self.requires("llvm-core/13.0.0")
        if self.options.with_openssl:
            self.requires("openssl/[>=1.1 <4]")
        if self.options.get_safe("with_opentelemetry"):
            self.requires("opentelemetry-cpp/1.23.0@milvus/storage-tracing#474cf5b43e5e141846f6a1d4bd2a8db5")
            self.requires("nlohmann_json/3.11.3#ffb9e9236619f1c883e36662f944345d")
        if self.options.with_s3:
            self.requires("aws-sdk-cpp/1.11.692@milvus/dev")
        if self.options.with_brotli:
            self.requires("brotli/1.1.0")
        if self.options.with_bz2:
            self.requires("bzip2/1.0.8")
        if self.options.with_lz4:
            self.requires("lz4/1.9.4")
        if self.options.with_snappy:
            self.requires("snappy/1.1.9")
        if self.options.get_safe("simd_level") != None or \
            self.options.get_safe("runtime_simd_level") != None:
            self.requires("xsimd/9.0.1")
        if self.options.with_zlib:
            self.requires("zlib/[>=1.2.11 <2]")
        if self.options.with_zstd:
            self.requires("zstd/1.5.5")
        if self.options.with_re2:
            self.requires("re2/20230301")
        if self.options.with_utf8proc:
            self.requires("utf8proc/2.8.0")
        if self.options.with_backtrace:
            self.requires("libbacktrace/cci.20210118")
        if self.options.with_orc:
            self.requires("orc/2.0.0")
        if self.options.with_azure:
            self.requires("azure-sdk-for-cpp/1.11.3@milvus/dev")

    def source(self):
        get(self,
            url="https://archive.apache.org/dist/arrow/arrow-17.0.0/apache-arrow-17.0.0.tar.gz",
            sha256=self.conan_data["sources"]["17.0.0"]["sha256"],
            strip_root=True)

    def generate(self):
        variables = dict(self.conf.get("tools.cmake.cmaketoolchain:extra_variables", default={}))
        variables["ARROW_WITH_OPENTELEMETRY"] = bool(self.options.with_opentelemetry)
        self.conf.define("tools.cmake.cmaketoolchain:extra_variables", variables)
        super().generate()

    def _patch_sources(self):
        super()._patch_sources()
        patch(self, base_path=self.source_folder,
              patch_file=os.path.join(self.source_folder, "tracing-api-only.patch"))
        header = os.path.join(self.source_folder, "cpp", "src", "arrow", "util", "tracing_internal.h")
        # Storage uses Arrow's actual Span and END_SPAN/MARK_SPAN macros across
        # the shared-library boundary, so their out-of-line helpers must export.
        replace_in_file(self, header,
                        "\nopentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>&",
                        "\nARROW_EXPORT opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>&")
        replace_in_file(self, header,
                        "\nconst opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>&",
                        "\nARROW_EXPORT const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>&")
        replace_in_file(self, header,
                        "\nopentelemetry::trace::StartSpanOptions SpanOptionsWithParent(",
                        "\nARROW_EXPORT opentelemetry::trace::StartSpanOptions SpanOptionsWithParent(")
        source = os.path.join(self.source_folder, "cpp", "src", "arrow", "util", "tracing_internal.cc")
        replace_in_file(self, source,
                        "  bool Shutdown(std::chrono::microseconds timeout =",
                        "  bool ForceFlush(std::chrono::microseconds) noexcept override {\n"
                        "    out_->flush();\n"
                        "    return true;\n"
                        "  }\n"
                        "  bool Shutdown(std::chrono::microseconds timeout =")

    def package(self):
        super().package()
        copy(self, "tracing_internal.h",
             src=os.path.join(self.source_folder, "cpp", "src", "arrow", "util"),
             dst=os.path.join(self.package_folder, "include", "arrow", "util"))

    def package_info(self):
        super().package_info()
        for component in self.cpp_info.components.values():
            component.requires = [
                "protobuf::libprotobuf" if requirement == "protobuf::protobuf" else requirement
                for requirement in component.requires
            ]
        if self.options.with_opentelemetry:
            self.cpp_info.components["libarrow"].defines.append("OPENTELEMETRY_STL_VERSION=2017")
            requirements = self.cpp_info.components["libarrow"].requires
            requirements.remove("opentelemetry-cpp::opentelemetry-cpp")
            if "protobuf::libprotobuf" not in requirements:
                requirements.append("protobuf::libprotobuf")
            requirements.extend([
                "nlohmann_json::nlohmann_json",
                "opentelemetry-cpp::opentelemetry_trace",
                "opentelemetry-cpp::opentelemetry_logs",
                "opentelemetry-cpp::opentelemetry_exporter_ostream_span",
                "opentelemetry-cpp::opentelemetry_exporter_ostream_logs",
                "opentelemetry-cpp::opentelemetry_exporter_otlp_http",
                "opentelemetry-cpp::opentelemetry_exporter_otlp_http_log",
            ])
