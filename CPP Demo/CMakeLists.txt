cmake_minimum_required(VERSION 3.14)
project(rag_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CURL REQUIRED)
find_package(nlohmann_json 3.2.0 QUIET)

# If nlohmann_json is not found as a package, you can vendor it by placing json.hpp in include/
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found as a package; expecting include/json.hpp in project include/")
    include_directories(${CMAKE_SOURCE_DIR}/include)
else()
    message(STATUS "Using system nlohmann_json")
endif()

add_executable(rag_demo
    src/main_demo.cpp
    src/rag_session.cpp
)

target_include_directories(rag_demo PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rag_demo PRIVATE CURL::libcurl)
if(nlohmann_json_FOUND)
    target_link_libraries(rag_demo PRIVATE nlohmann_json::nlohmann_json)
endif()
