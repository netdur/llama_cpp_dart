./gradlew :llamalib:clean :llamalib:assembleRelease \
          --no-daemon \
          --console=plain \
          --info --stacktrace \
          -Pandroid.native.buildOutput=verbose

# check /Users/adel/Workspace/llama_cpp_dart/android/llamalib/build/outputs/aar