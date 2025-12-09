cmake_minimum_required(VERSION 3.15)
project(control_kalman)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =================== VCPKG ИНТЕГРАЦИЯ ===================
# Укажите путь к vcpkg.cmake (замените на ваш реальный путь)
set(CMAKE_TOOLCHAIN_FILE "D:/Kalman_Filter/control-kalman/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")

# =================== ПОИСК EIGEN ===================
find_package(Eigen3 REQUIRED)

# =================== ИСПОЛНЯЕМЫЙ ФАЙЛ ===================
add_executable(control_kalman
        main.cpp
        CKF.cpp
        SRCF.cpp
        SRCF.cpp
)

# =================== НАСТРОЙКИ КОМПИЛЯТОРА ===================
target_compile_options(control_kalman PRIVATE
        -Wall
        -Wextra
        -O2
)

# =================== ПОДКЛЮЧЕНИЕ BIBЛИОTEK ===================
target_link_libraries(control_kalman PRIVATE
        Eigen3::Eigen
)

# =================== ВЫВОД ИНФОРМАЦИИ ===================
message(STATUS "Using Eigen3 from: ${EIGEN3_INCLUDE_DIRS}")
message(STATUS "Project configured successfully with vcpkg!")