from conan import ConanFile
from conan.tools.build import check_min_cppstd


class StorageOpenTelemetryConan(ConanFile):
    """Match the public STL ABI already required by milvus-common."""

    python_requires = "opentelemetry-cpp/1.23.0@milvus/dev#11bc565ec6e82910ae8f7471da756720"
    python_requires_extend = "opentelemetry-cpp.OpenTelemetryCppConan"

    def init(self):
        self.conan_data = self.python_requires["opentelemetry-cpp"].conanfile.conan_data

    @property
    def _stl_value(self):
        return "CXX17" if self.options.with_stl else "OFF"

    def validate(self):
        super().validate()
        if self.options.with_stl:
            check_min_cppstd(self, "17")

    def package_info(self):
        super().package_info()
        if self.options.with_stl:
            self.cpp_info.components["opentelemetry_common"].defines.append("OPENTELEMETRY_STL_VERSION=2017")
