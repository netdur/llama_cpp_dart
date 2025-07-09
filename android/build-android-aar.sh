./gradlew :llamalib:clean :llamalib:assembleRelease \
          --no-daemon \
          --console=plain \
          --info --stacktrace \
          -Pandroid.native.buildOutput=verbose \
          -Dcmake.verbose=true

# check /Users/adel/Workspace/llama_cpp_dart/android/llamalib/build/outputs/aar