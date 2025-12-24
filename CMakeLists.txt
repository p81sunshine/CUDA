cmake_minimum_required(VERSION 3.16)
project(meshDistCPU LANGUAGES CXX)

# ---------- 基础设置 ----------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 如果需要，可把 OFF 改成 ON 再自己装系统 freeglut
set(USE_SYSTEM_GLUT OFF CACHE BOOL "Prefer system-installed GLUT" FORCE)
option(BUILD_IMGUI_DEMO "Build ImGui demo code" OFF)
option(USE_GPU "Use GPU (CUDA) version for distance calculation" ON)

# ---------- 源文件 ----------
set(SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/obj-viewer.cpp
    ${CMAKE_SOURCE_DIR}/src/cmodel.cpp
    ${CMAKE_SOURCE_DIR}/src/crigid.cpp

    # ImGui 源
    ${CMAKE_SOURCE_DIR}/inc/IMGUI/imgui.cpp
    ${CMAKE_SOURCE_DIR}/inc/IMGUI/imgui_draw.cpp
    ${CMAKE_SOURCE_DIR}/inc/IMGUI/imgui_tables.cpp
    ${CMAKE_SOURCE_DIR}/inc/IMGUI/imgui_widgets.cpp
    ${CMAKE_SOURCE_DIR}/inc/IMGUI/imgui_impl_opengl3.cpp
)

# 根据选项添加 CUDA 源文件
if(USE_GPU)
    list(APPEND SRC_FILES ${CMAKE_SOURCE_DIR}/src/mesh_distance.cu)
endif()
if(BUILD_IMGUI_DEMO)
    list(APPEND SRC_FILES ${CMAKE_SOURCE_DIR}/inc/IMGUI/imgui_demo.cpp)
endif()

# ---------- 目标 ----------
if(USE_GPU)
    add_executable(meshDistGPU ${SRC_FILES})
    set(TARGET_NAME meshDistGPU)
else()
    add_executable(meshDistCPU ${SRC_FILES})
    set(TARGET_NAME meshDistCPU)
endif()

# 配置 CUDA 属性（仅在启用GPU时）
if(USE_GPU)
    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;86"
    )
endif()

# 头文件路径
target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/inc/GL
    ${CMAKE_SOURCE_DIR}/inc/IMGUI
    ${CMAKE_BINARY_DIR}      # 供 Imgui/ 大小写别名使用
)

# CPU-only 宏，避免无 CUDA 环境包含 cuda_runtime.h
target_compile_definitions(${TARGET_NAME} PRIVATE MESH_DIST_CPU_ONLY)

# 编译告警（只针对 C/C++ 文件，不针对 CUDA）
if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3 /permissive->)
else()
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>)
endif()

# ---------- OpenGL ----------
find_package(OpenGL REQUIRED)
# 在 Windows 上 OpenGL::GL 会映射到 opengl32，这里仍额外显式链接一次确保兼容
if(WIN32)
    target_link_libraries(${TARGET_NAME} PRIVATE opengl32 glu32)
else()
    # Linux/Unix：链接 OpenGL 和 GLU
    target_link_libraries(${TARGET_NAME} PRIVATE OpenGL::GL OpenGL::GLU)
endif()

# ---------- GLUT（重点：Windows 优先使用本地 lib） ----------
if(WIN32)
    # 选择本地打包库（与位宽匹配）
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(BUNDLED_GLUT_LIB ${CMAKE_SOURCE_DIR}/lib/glut64.lib)
    else()
        set(BUNDLED_GLUT_LIB ${CMAKE_SOURCE_DIR}/lib/glut32.lib)
    endif()

    if(NOT EXISTS ${BUNDLED_GLUT_LIB})
        if(USE_SYSTEM_GLUT)
            message(STATUS "Bundled GLUT not found, trying system GLUT...")
            find_package(GLUT)
            if(GLUT_FOUND)
                target_link_libraries(${TARGET_NAME} PRIVATE GLUT::GLUT)
            else()
                message(FATAL_ERROR
                    "GLUT not found. Provide ${BUNDLED_GLUT_LIB} or install system freeglut.")
            endif()
        else()
            message(FATAL_ERROR
                "Bundled GLUT not found at: ${BUNDLED_GLUT_LIB}\n"
                "Hint: x64 构建需要 lib\\glut64.lib；Win32 构建需要 lib\\glut32.lib。")
        endif()
    else()
        message(STATUS "Using bundled GLUT: ${BUNDLED_GLUT_LIB}")
        # 用绝对路径直接链接，不依赖 FindGLUT 名称解析
        target_link_libraries(${TARGET_NAME} PRIVATE ${BUNDLED_GLUT_LIB})
    endif()

elseif(APPLE)
    # macOS：用系统框架（如已安装 freeglut 也可 find_package(GLUT)）
    find_library(GLUT_FRAMEWORK GLUT)
    find_library(COCOA_FRAMEWORK Cocoa)
    find_library(IOKIT_FRAMEWORK IOKit)
    find_library(COREVIDEO_FRAMEWORK CoreVideo)
    if(GLUT_FRAMEWORK)
        target_link_libraries(${TARGET_NAME} PRIVATE ${GLUT_FRAMEWORK})
    else()
        message(FATAL_ERROR "GLUT framework not found on macOS. Consider installing freeglut.")
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE ${COCOA_FRAMEWORK} ${IOKIT_FRAMEWORK} ${COREVIDEO_FRAMEWORK})

else() # Linux/Unix
    # 系统 freeglut
    find_package(GLUT)
    if(GLUT_FOUND)
        target_link_libraries(${TARGET_NAME} PRIVATE GLUT::GLUT)
    else()
        find_library(GLUT_LIB NAMES glut freeglut)
        if(GLUT_LIB)
            target_link_libraries(${TARGET_NAME} PRIVATE ${GLUT_LIB})
        else()
            message(FATAL_ERROR "GLUT not found. Install freeglut (e.g. sudo apt install freeglut3-dev).")
        endif()
    endif()

    # 可能还需要 pthread / dl
    find_package(Threads)
    if(Threads_FOUND)
        target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads)
    endif()
    find_library(DL_LIB dl)
    if(DL_LIB)
        target_link_libraries(${TARGET_NAME} PRIVATE ${DL_LIB})
    endif()
endif()

# ---------- OpenMP（如 cmodel.cpp 用到 <omp.h>） ----------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(${TARGET_NAME} PRIVATE -DUSE_OMP)
endif()

# ---------- 运行资源复制 ----------
add_custom_target(copy_data ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/data ${CMAKE_BINARY_DIR}/data
    COMMENT "Copying runtime data/ folder"
)
add_dependencies(${TARGET_NAME} copy_data)

# 解决 Linux/macOS 大小写敏感：把 inc/IMGUI -> Imgui/
add_custom_target(copy_imgui ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/Imgui
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/inc/IMGUI ${CMAKE_BINARY_DIR}/Imgui
    COMMENT "Copying ImGui headers to case alias: Imgui/"
)
add_dependencies(${TARGET_NAME} copy_imgui)

# ---------- 可执行文件输出目录 ----------
set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ---------- CUDA 配置（根据 USE_GPU 选项决定是否启用） ----------
if(USE_GPU)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    target_include_directories(${TARGET_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} PRIVATE ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY})
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_CUDA)
    message(STATUS "GPU version enabled (USE_GPU=ON)")
else()
    message(STATUS "CPU version enabled (USE_GPU=OFF)")
endif()
