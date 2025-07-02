set(CMAKE_MACOSX_BUNDLE FALSE)
if (TARGET llama-batched-bench)
    set_target_properties(llama-batched-bench
                          PROPERTIES MACOSX_BUNDLE FALSE)
endif()