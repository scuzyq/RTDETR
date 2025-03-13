# Generated file

if (DEFINED ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT $ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
else()
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT @CUTLASS_TEST_EXECUTION_ENVIRONMENT@)
endif()

if (NOT "@TEST_EXE_DIR@" STREQUAL "")
  set(TEST_EXE_PATH @TEST_EXE_DIR@/@TEST_EXE@)
else()
  set(TEST_EXE_PATH @TEST_EXE@)
endif()

add_test("@TEST_NAME@" ${_CUTLASS_TEST_EXECUTION_ENVIRONMENT} "${TEST_EXE_PATH}" @TEST_COMMAND_OPTIONS@)

if (NOT "@TEST_EXE_WORKING_DIRECTORY@" STREQUAL "")
  set_tests_properties("@TEST_NAME@" PROPERTIES WORKING_DIRECTORY "@TEST_EXE_WORKING_DIRECTORY@")
endif()
