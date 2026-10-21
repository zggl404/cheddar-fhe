include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

add_executable(parameter_literal_cpu_test
  ../tests/parameter_literal_test.cpp ../src/extension/BootParameter.cpp)
target_include_directories(parameter_literal_cpu_test PRIVATE ../include)

add_executable(parameter_literal_test ParameterLiteral.cpp)
target_link_libraries(parameter_literal_test PRIVATE cheddar gtest_main)

add_executable(basic_test BasicTest.cpp)
target_link_libraries(basic_test
                      PRIVATE cheddar gtest_main)

if (ENABLE_EXTENSION)
  add_executable(bootstrap_scale_probe ../tests/bootstrap_scale_probe.cpp)
  target_link_libraries(bootstrap_scale_probe PRIVATE cheddar)
  add_executable(boot_test Bootstrapping.cpp)
  target_link_libraries(boot_test
                        PRIVATE cheddar gtest_main)
endif()
