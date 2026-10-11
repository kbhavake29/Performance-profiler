cmake_minimum_required(VERSION 3.20)

project(inference_profiler
  VERSION 0.1.0
  DESCRIPTION "Native OpenTelemetry instrumentation for LLM inference workloads"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(IPP_BUILD_TESTS "Build the C++ unit tests" ON)
option(IPP_BUILD_EXAMPLES "Build the native demo" ON)
option(IPP_FETCH_OPENTELEMETRY
  "Build opentelemetry-cpp from source (pinned) when no installed package is found" OFF)
option(IPP_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
set(IPP_OPENTELEMETRY_TAG "v1.29.0" CACHE STRING "opentelemetry-cpp tag for IPP_FETCH_OPENTELEMETRY")

# ------------------------------------------------------------------ dependencies
# Preferred: an installed opentelemetry-cpp with the OTLP/HTTP exporter
# (Homebrew `opentelemetry-cpp`, vcpkg `opentelemetry-cpp[otlp-http]`, or a source install).
find_package(opentelemetry-cpp CONFIG QUIET)

if(opentelemetry-cpp_FOUND)
  message(STATUS "Using installed opentelemetry-cpp ${opentelemetry-cpp_VERSION}")
  set(IPP_OTEL_TARGETS
    opentelemetry-cpp::api
    opentelemetry-cpp::sdk
    opentelemetry-cpp::trace
    opentelemetry-cpp::metrics
    opentelemetry-cpp::resources
    opentelemetry-cpp::otlp_http_exporter
    opentelemetry-cpp::otlp_http_metric_exporter)
  set(IPP_OTEL_TEST_TARGETS
    opentelemetry-cpp::in_memory_span_exporter
    opentelemetry-cpp::in_memory_metric_exporter)
elseif(IPP_FETCH_OPENTELEMETRY)
  message(STATUS "Fetching opentelemetry-cpp ${IPP_OPENTELEMETRY_TAG} (needs protobuf and libcurl)")
  include(FetchContent)
  set(WITH_OTLP_HTTP ON CACHE BOOL "" FORCE)
  set(WITH_OTLP_GRPC OFF CACHE BOOL "" FORCE)
  set(WITH_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(WITH_BENCHMARK OFF CACHE BOOL "" FORCE)
  set(WITH_FUNC_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING_SAVED ${BUILD_TESTING})
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(OPENTELEMETRY_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(opentelemetry-cpp
    GIT_REPOSITORY https://github.com/open-telemetry/opentelemetry-cpp.git
    GIT_TAG ${IPP_OPENTELEMETRY_TAG}
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(opentelemetry-cpp)
  set(BUILD_TESTING ${BUILD_TESTING_SAVED} CACHE BOOL "" FORCE)
  set(IPP_OTEL_TARGETS
    opentelemetry_api opentelemetry_sdk opentelemetry_trace opentelemetry_metrics
    opentelemetry_resources opentelemetry_exporter_otlp_http
    opentelemetry_exporter_otlp_http_metric)
  set(IPP_OTEL_TEST_TARGETS
    opentelemetry_exporter_in_memory opentelemetry_exporter_in_memory_metric)
else()
  message(FATAL_ERROR
    "opentelemetry-cpp not found. Install it (macOS: `brew install opentelemetry-cpp`; "
    "vcpkg: `vcpkg install opentelemetry-cpp[otlp-http]`) or configure with "
    "-DIPP_FETCH_OPENTELEMETRY=ON to build it from source. See docs/integration-guide.md.")
endif()

# ----------------------------------------------------------------------- library
add_library(inference_profiler
  cpp/src/profiler.cpp
  cpp/src/span.cpp
  cpp/src/metrics.cpp)
add_library(inference_profiler::inference_profiler ALIAS inference_profiler)
target_include_directories(inference_profiler
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src)
target_link_libraries(inference_profiler PUBLIC ${IPP_OTEL_TARGETS})
target_compile_definitions(inference_profiler PRIVATE
  INFERENCE_PROFILER_VERSION="${PROJECT_VERSION}")

if(MSVC)
  set(IPP_WARNING_FLAGS /W4)
  if(IPP_WARNINGS_AS_ERRORS)
    list(APPEND IPP_WARNING_FLAGS /WX)
  endif()
else()
  set(IPP_WARNING_FLAGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
  if(IPP_WARNINGS_AS_ERRORS)
    list(APPEND IPP_WARNING_FLAGS -Werror)
  endif()
endif()
target_compile_options(inference_profiler PRIVATE ${IPP_WARNING_FLAGS})

include(GNUInstallDirs)
install(TARGETS inference_profiler ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                                   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY cpp/include/inference_profiler DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ----------------------------------------------------------------------- example
if(IPP_BUILD_EXAMPLES)
  add_executable(cpp_inference_demo cpp/examples/cpp_inference_demo.cpp)
  target_link_libraries(cpp_inference_demo PRIVATE inference_profiler)
  target_compile_options(cpp_inference_demo PRIVATE ${IPP_WARNING_FLAGS})
endif()

# ------------------------------------------------------------------------- tests
if(IPP_BUILD_TESTS)
  enable_testing()
  add_executable(inference_profiler_tests
    tests/cpp/test_main.cpp
    tests/cpp/test_config.cpp
    tests/cpp/test_spans.cpp
    tests/cpp/test_metrics.cpp
    tests/cpp/test_export.cpp)
  target_include_directories(inference_profiler_tests PRIVATE tests/cpp)
  target_link_libraries(inference_profiler_tests PRIVATE inference_profiler ${IPP_OTEL_TEST_TARGETS})
  target_compile_options(inference_profiler_tests PRIVATE ${IPP_WARNING_FLAGS})
  # One CTest entry per test case so failures are reported individually.
  set(IPP_TEST_CASES
    config_defaults config_from_env config_env_errors config_validation
    span_request_lifecycle span_nested_phases span_raii_on_scope_exit span_exception_marks_error
    span_cancel_and_fail span_move_semantics span_traceparent_propagation span_disabled_noop
    span_invalid_phase_name span_strict_mode_throws span_cache_operations span_ttft_explicit
    metrics_request_counters metrics_histograms metrics_gauges_return_to_zero
    metrics_cardinality_limit metrics_process_rss
    export_unreachable_collector export_shutdown_bounded export_shutdown_idempotent
    export_failures_counted)
  foreach(case IN LISTS IPP_TEST_CASES)
    add_test(NAME ${case} COMMAND inference_profiler_tests ${case})
    set_tests_properties(${case} PROPERTIES TIMEOUT 60)
  endforeach()
endif()
