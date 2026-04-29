#
# Vendored xcframework for llama_cpp_dart on iOS / macOS.
#
# Usage from a Flutter app's iOS or macOS Podfile:
#
#   target 'Runner' do
#     ...
#     pod 'llama_cpp', :path => '../../llama_cpp_dart'   # adjust path
#   end
#
# This points the Flutter target at the prebuilt xcframework so it gets
# embedded + signed automatically. App code should call
# `LlamaEngine.spawnFromProcess(...)` (the framework is static-linked into
# the app binary, no runtime dlopen path needed).
#
Pod::Spec.new do |s|
  s.name             = 'llama_cpp'
  s.version          = '0.9.0-dev.1'
  s.summary          = 'llama.cpp xcframework for iOS / macOS via llama_cpp_dart.'
  s.description      = <<-DESC
    Prebuilt llama.cpp + ggml + mtmd as an xcframework, vendored for use from
    Flutter apps that depend on the llama_cpp_dart binding. Three slices:
    ios-arm64, ios-arm64-simulator, macos-arm64.
  DESC
  s.homepage         = 'https://github.com/netdur/llama_cpp_dart'
  s.license          = { :type => 'MIT' }
  s.author           = { 'netdur' => 'noreply@netdur.dev' }
  s.source           = { :path => '.' }

  s.ios.deployment_target = '14.0'
  s.osx.deployment_target = '11.0'

  s.vendored_frameworks = 'build/apple/llama.xcframework'

  # Frameworks the xcframework links against.
  s.frameworks = 'Metal', 'MetalKit', 'Foundation', 'Accelerate'

  # The xcframework is a static archive wrapped in a .framework. Without
  # -force_load, the linker drops all of its symbols because nothing in
  # the Flutter Runner / App.framework references them at compile time.
  # Dart's DynamicLibrary.process() (used by LlamaEngine.spawnFromProcess)
  # then can't find e.g. llama_backend_init at runtime.
  s.user_target_xcconfig = {
    'OTHER_LDFLAGS' => '$(inherited) -force_load $(PODS_XCFRAMEWORKS_BUILD_DIR)/llama_cpp/llama.framework/llama',
  }
  s.pod_target_xcconfig = {
    'OTHER_LDFLAGS' => '$(inherited) -force_load $(PODS_XCFRAMEWORKS_BUILD_DIR)/llama_cpp/llama.framework/llama',
  }
end
