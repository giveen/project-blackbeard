# CMake generated Testfile for 
# Source directory: /mnt/storage/blackbeard/examples/eval-callback
# Build directory: /mnt/storage/blackbeard/build-fate-cuda/examples/eval-callback
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[test-eval-callback-download-model]=] "/usr/bin/cmake" "-DDEST=/mnt/storage/blackbeard/build-fate-cuda/tinyllamas/stories15M-q4_0.gguf" "-DNAME=tinyllamas/stories15M-q4_0.gguf" "-DHASH=SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739" "-P" "/mnt/storage/blackbeard/cmake/download-models.cmake")
set_tests_properties([=[test-eval-callback-download-model]=] PROPERTIES  FIXTURES_SETUP "test-eval-callback-download-model" _BACKTRACE_TRIPLES "/mnt/storage/blackbeard/examples/eval-callback/CMakeLists.txt;17;add_test;/mnt/storage/blackbeard/examples/eval-callback/CMakeLists.txt;0;")
add_test([=[test-eval-callback]=] "/mnt/storage/blackbeard/build-fate-cuda/bin/llama-eval-callback" "-m" "/mnt/storage/blackbeard/build-fate-cuda/tinyllamas/stories15M-q4_0.gguf" "--prompt" "hello" "--seed" "42" "-ngl" "0")
set_tests_properties([=[test-eval-callback]=] PROPERTIES  FIXTURES_REQUIRED "test-eval-callback-download-model" _BACKTRACE_TRIPLES "/mnt/storage/blackbeard/examples/eval-callback/CMakeLists.txt;24;add_test;/mnt/storage/blackbeard/examples/eval-callback/CMakeLists.txt;0;")
