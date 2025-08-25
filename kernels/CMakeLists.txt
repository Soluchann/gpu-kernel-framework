# Collect all kernel subdirectories
file(GLOB KERNEL_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

# List to collect all kernel libraries
set(ALL_KERNEL_LIBS "")

foreach(KERNEL_DIR ${KERNEL_DIRS})
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_DIR})

        # Grab all .cu files inside this kernel folder
        file(GLOB KERNEL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_DIR}/*.cu)

        if (KERNEL_SOURCES)
            # Library name = <subdir>_kernel
            string(REPLACE "/" "_" LIB_NAME ${KERNEL_DIR})
            set(TARGET_NAME ${LIB_NAME}_kernel)

            add_library(${TARGET_NAME} STATIC ${KERNEL_SOURCES})
            set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

            target_include_directories(${TARGET_NAME} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_DIR}
                ${CMAKE_SOURCE_DIR}/src
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
            )

            # Auto-enable FP16/BF16 flags based on directory name
            if (${KERNEL_DIR} MATCHES "fp16")
                target_compile_definitions(${TARGET_NAME} PRIVATE CUDA_HAS_FP16=1)
            elseif (${KERNEL_DIR} MATCHES "bf16")
                target_compile_definitions(${TARGET_NAME} PRIVATE CUDA_HAS_BF16=1)
            endif()

            # Add to list of kernel libraries
            list(APPEND ALL_KERNEL_LIBS ${TARGET_NAME})
        endif()
    endif()
endforeach()

# Export the list to parent scope
set(ALL_KERNEL_LIBS ${ALL_KERNEL_LIBS} PARENT_SCOPE)