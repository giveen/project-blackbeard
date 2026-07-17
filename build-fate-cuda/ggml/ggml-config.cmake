
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was ggml-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################


####### Expanded from @GGML_VARIABLES_EXPANED@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run        #######

set(GGML_ACCELERATE "ON")
set(GGML_ALL_WARNINGS "ON")
set(GGML_ALL_WARNINGS_3RD_PARTY "OFF")
set(GGML_AMX_BF16 "OFF")
set(GGML_AMX_INT8 "OFF")
set(GGML_AMX_TILE "OFF")
set(GGML_AVAILABLE_BACKENDS "ggml-cpu;ggml-cuda")
set(GGML_AVX "OFF")
set(GGML_AVX2 "OFF")
set(GGML_AVX512 "OFF")
set(GGML_AVX512_BF16 "OFF")
set(GGML_AVX512_VBMI "OFF")
set(GGML_AVX512_VNNI "OFF")
set(GGML_AVX_VNNI "OFF")
set(GGML_BACKEND_DIR "")
set(GGML_BACKEND_DL "OFF")
set(GGML_BLAS_DEFAULT "OFF")
set(GGML_BLAS_VENDOR_DEFAULT "Generic")
set(GGML_BMI2 "OFF")
set(GGML_BUILD_COMMIT "d23543dd8")
set(GGML_BUILD_EXAMPLES "OFF")
set(GGML_BUILD_NUMBER "553")
set(GGML_BUILD_TESTS "OFF")
set(GGML_CCACHE "ON")
set(GGML_CCACHE_FOUND "/usr/bin/ccache")
set(GGML_CPU "ON")
set(GGML_CPU_ALL_VARIANTS "OFF")
set(GGML_CPU_ARM_ARCH "")
set(GGML_CPU_HBM "OFF")
set(GGML_CPU_KLEIDIAI "OFF")
set(GGML_CPU_POWERPC_CPUTYPE "")
set(GGML_CPU_REPACK "ON")
set(GGML_CUDA "ON")
set(GGML_CUDA "ON")
set(GGML_CUDA_COMPRESSION_MODE "size")
set(GGML_CUDA_FA "OFF")
set(GGML_CUDA_FA_ALL_QUANTS "OFF")
set(GGML_CUDA_FORCE_CUBLAS "OFF")
set(GGML_CUDA_FORCE_MMQ "OFF")
set(GGML_CUDA_GRAPHS "ON")
set(GGML_CUDA_GRAPHS_DEFAULT "ON")
set(GGML_CUDA_NCCL "ON")
set(GGML_CUDA_NO_PEER_COPY "OFF")
set(GGML_CUDA_NO_VMM "OFF")
set(GGML_CUDA_PEER_MAX_BATCH_SIZE "128")
set(GGML_F16C "OFF")
set(GGML_FATAL_WARNINGS "OFF")
set(GGML_FMA "OFF")
set(GGML_GIT_DIRTY "0")
set(GGML_GPROF "OFF")
set(GGML_LASX "ON")
set(GGML_LLAMAFILE "ON")
set(GGML_LLAMAFILE_DEFAULT "ON")
set(GGML_LSX "ON")
set(GGML_LTO "OFF")
set(GGML_METAL_DEFAULT "OFF")
set(GGML_NATIVE "ON")
set(GGML_NATIVE_DEFAULT "ON")
set(GGML_OPENMP "ON")
set(GGML_OPENMP_ENABLED "ON")
set(GGML_PUBLIC_HEADERS "include/ggml.h;include/ggml-cpu.h;include/ggml-alloc.h;include/ggml-backend.h;include/ggml-blas.h;include/ggml-cann.h;include/ggml-cpp.h;include/ggml-cuda.h;include/ggml-opt.h;include/ggml-metal.h;include/ggml-rpc.h;include/ggml-virtgpu.h;include/ggml-sycl.h;include/ggml-vulkan.h;include/ggml-webgpu.h;include/ggml-zendnn.h;include/ggml-openvino.h;include/gguf.h")
set(GGML_RVV "ON")
set(GGML_RV_ZFH "ON")
set(GGML_RV_ZICBOP "ON")
set(GGML_RV_ZIHINTPAUSE "ON")
set(GGML_RV_ZVFBFWMA "OFF")
set(GGML_RV_ZVFH "ON")
set(GGML_SANITIZE_ADDRESS "OFF")
set(GGML_SANITIZE_THREAD "OFF")
set(GGML_SANITIZE_UNDEFINED "OFF")
set(GGML_SCCACHE_FOUND "GGML_SCCACHE_FOUND-NOTFOUND")
set(GGML_SCHED_MAX_COPIES "4")
set(GGML_SCHED_NO_REALLOC "OFF")
set(GGML_SHARED_LIB "ON")
set(GGML_SSE42 "OFF")
set(GGML_STANDALONE "OFF")
set(GGML_STATIC "OFF")
set(GGML_VERSION "0.16.0")
set(GGML_VERSION_BASE "0.16.0")
set(GGML_VERSION_MAJOR "0")
set(GGML_VERSION_MINOR "16")
set(GGML_VERSION_PATCH "0")
set(GGML_VXE "ON")
set(GGML_XTHEADVECTOR "OFF")


# Find all dependencies before creating any target.
include(CMakeFindDependencyMacro)
find_dependency(Threads)
if (NOT GGML_SHARED_LIB)
    set(GGML_BASE_INTERFACE_LINK_LIBRARIES "")
    set(GGML_CPU_INTERFACE_LINK_LIBRARIES "")
    set(GGML_CPU_INTERFACE_LINK_OPTIONS   "")

    if (APPLE AND GGML_ACCELERATE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if(NOT ACCELERATE_FRAMEWORK)
            set(${CMAKE_FIND_PACKAGE_NAME}_FOUND 0)
            return()
        endif()
        list(APPEND GGML_CPU_INTERFACE_LINK_LIBRARIES ${ACCELERATE_FRAMEWORK})
    endif()

    if (GGML_OPENMP_ENABLED)
        find_dependency(OpenMP)
        set(GGML_OPENMP_INTERFACE_LINK_LIBRARIES "")
        if (TARGET OpenMP::OpenMP_C)
            list(APPEND GGML_OPENMP_INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_C)
        endif()
        if (TARGET OpenMP::OpenMP_CXX)
            list(APPEND GGML_OPENMP_INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_CXX)
        endif()
        list(APPEND GGML_BASE_INTERFACE_LINK_LIBRARIES ${GGML_OPENMP_INTERFACE_LINK_LIBRARIES})
        list(APPEND GGML_CPU_INTERFACE_LINK_LIBRARIES ${GGML_OPENMP_INTERFACE_LINK_LIBRARIES})
    endif()

    if (GGML_CPU_HBM)
        find_library(memkind memkind)
        if(NOT memkind)
            set(${CMAKE_FIND_PACKAGE_NAME}_FOUND 0)
            return()
        endif()
        list(APPEND GGML_CPU_INTERFACE_LINK_LIBRARIES memkind)
    endif()


    if (GGML_CUDA)
        set(GGML_CUDA_INTERFACE_LINK_LIBRARIES "")
        find_dependency(CUDAToolkit)
        if (GGML_STATIC)
            list(APPEND GGML_CUDA_INTERFACE_LINK_LIBRARIES $<LINK_ONLY:CUDA::cudart_static>)
            if (WIN32)
                list(APPEND GGML_CUDA_INTERFACE_LINK_LIBRARIES $<LINK_ONLY:CUDA::cublas> $<LINK_ONLY:CUDA::cublasLt>)
            else()
                list(APPEND GGML_CUDA_INTERFACE_LINK_LIBRARIES $<LINK_ONLY:CUDA::cublas_static> $<LINK_ONLY:CUDA::cublasLt_static>)
            endif()
        endif()
        if (NOT GGML_CUDA_NO_VMM)
            list(APPEND GGML_CUDA_INTERFACE_LINK_LIBRARIES $<LINK_ONLY:CUDA::cuda_driver>)
        endif()
    endif()





endif()

set_and_check(GGML_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")
set_and_check(GGML_LIB_DIR "${PACKAGE_PREFIX_DIR}/lib")
#set_and_check(GGML_BIN_DIR "${PACKAGE_PREFIX_DIR}/bin")

if(NOT TARGET ggml::ggml)
    find_package(Threads REQUIRED)

    find_library(GGML_LIBRARY ggml
        REQUIRED
        HINTS ${GGML_LIB_DIR}
        NO_CMAKE_FIND_ROOT_PATH)

    add_library(ggml::ggml UNKNOWN IMPORTED)
    set_target_properties(ggml::ggml
        PROPERTIES
            IMPORTED_LOCATION "${GGML_LIBRARY}")

    find_library(GGML_BASE_LIBRARY ggml-base
        REQUIRED
        HINTS ${GGML_LIB_DIR}
        NO_CMAKE_FIND_ROOT_PATH)

    add_library(ggml::ggml-base UNKNOWN IMPORTED)
    set_target_properties(ggml::ggml-base
        PROPERTIES
            IMPORTED_LOCATION "${GGML_BASE_LIBRARY}"
            INTERFACE_LINK_LIBRARIES "${GGML_BASE_INTERFACE_LINK_LIBRARIES}")

    set(_ggml_all_targets "")
    if (NOT GGML_BACKEND_DL)
        foreach(_ggml_backend ${GGML_AVAILABLE_BACKENDS})
            string(REPLACE "-" "_" _ggml_backend_pfx "${_ggml_backend}")
            string(TOUPPER "${_ggml_backend_pfx}" _ggml_backend_pfx)

            find_library(${_ggml_backend_pfx}_LIBRARY ${_ggml_backend}
                REQUIRED
                HINTS ${GGML_LIB_DIR}
                NO_CMAKE_FIND_ROOT_PATH)

            message(STATUS "Found ${${_ggml_backend_pfx}_LIBRARY}")

            add_library(ggml::${_ggml_backend} UNKNOWN IMPORTED)
            set_target_properties(ggml::${_ggml_backend}
                PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${GGML_INCLUDE_DIR}"
                    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
                    IMPORTED_LOCATION "${${_ggml_backend_pfx}_LIBRARY}"
                    INTERFACE_COMPILE_FEATURES c_std_90
                    POSITION_INDEPENDENT_CODE ON)

            string(REGEX MATCH "^ggml-cpu" is_cpu_variant "${_ggml_backend}")
            if(is_cpu_variant)
                list(APPEND GGML_CPU_INTERFACE_LINK_LIBRARIES "ggml::ggml-base")
                set_target_properties(ggml::${_ggml_backend}
                PROPERTIES
                    INTERFACE_LINK_LIBRARIES "${GGML_CPU_INTERFACE_LINK_LIBRARIES}")

                if(GGML_CPU_INTERFACE_LINK_OPTIONS)
                    set_target_properties(ggml::${_ggml_backend}
                        PROPERTIES
                            INTERFACE_LINK_OPTIONS "${GGML_CPU_INTERFACE_LINK_OPTIONS}")
                endif()

            else()
                list(APPEND ${_ggml_backend_pfx}_INTERFACE_LINK_LIBRARIES "ggml::ggml-base")
                set_target_properties(ggml::${_ggml_backend}
                    PROPERTIES
                        INTERFACE_LINK_LIBRARIES "${${_ggml_backend_pfx}_INTERFACE_LINK_LIBRARIES}")

                if(${_ggml_backend_pfx}_INTERFACE_LINK_OPTIONS)
                    set_target_properties(ggml::${_ggml_backend}
                        PROPERTIES
                            INTERFACE_LINK_OPTIONS "${${_ggml_backend_pfx}_INTERFACE_LINK_OPTIONS}")
                endif()
            endif()

            list(APPEND _ggml_all_targets ggml::${_ggml_backend})
        endforeach()
    endif()

    list(APPEND GGML_INTERFACE_LINK_LIBRARIES ggml::ggml-base "${_ggml_all_targets}")
    set_target_properties(ggml::ggml
        PROPERTIES
            INTERFACE_LINK_LIBRARIES "${GGML_INTERFACE_LINK_LIBRARIES}")

    add_library(ggml::all INTERFACE IMPORTED)
    set_target_properties(ggml::all
        PROPERTIES
            INTERFACE_LINK_LIBRARIES "${_ggml_all_targets}")

endif()

check_required_components(ggml)
