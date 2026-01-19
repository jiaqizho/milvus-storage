#!/bin/bash

# Configuration
BUILD_TYPE=${BUILD_TYPE:-Release}

# Detect LCOV version
LCOV_VERSION=$(lcov --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
LCOV_MAJOR=$(echo $LCOV_VERSION | cut -d. -f1)

# LCOV 2.0+ requires --ignore-errors for many common Clang/LLVM inconsistencies
if [ "$LCOV_MAJOR" -ge 2 ]; then
    LCOV_IGNORE_ERRORS="--ignore-errors mismatch,inconsistent,gcov,unused,deprecated,unsupported,format,count,category,empty,source"
else
    LCOV_IGNORE_ERRORS=""
fi

echo "Generating coverage report..."
mkdir -p coverage

# Capture coverage data (without branch coverage to avoid LLVM issues)
lcov --capture \
    --directory build/"${BUILD_TYPE}" \
    --output-file coverage/coverage.info \
    ${LCOV_IGNORE_ERRORS}

# Filter coverage data to only include src and include directories
lcov --extract coverage/coverage.info \
    '*/cpp/src/*' \
    '*/cpp/include/*' \
    --output-file coverage/coverage_filtered.info \
    ${LCOV_IGNORE_ERRORS}

# Generate HTML report (without branch coverage for LLVM compatibility)
genhtml coverage/coverage_filtered.info \
    --output-directory coverage/html \
    --title "Milvus Storage Coverage Report" \
    --legend \
    --highlight \
    ${LCOV_IGNORE_ERRORS}

echo ""
echo "====================================="
echo "Coverage report generated at: coverage/html/index.html"
echo "====================================="

# Print summary
lcov --summary coverage/coverage_filtered.info ${LCOV_IGNORE_ERRORS}