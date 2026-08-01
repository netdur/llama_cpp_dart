Pod::Spec.new do |s|
  s.name             = 'llama_cpp_dart'
  s.version          = '0.9.0-dev.10'
  s.summary          = 'Dart / Flutter FFI binding for llama.cpp'
  s.description      = 'High-level Dart and Flutter bindings for llama.cpp on iOS, macOS, and Android.'
  s.homepage         = 'https://github.com/netdur/llama_cpp_dart'
  s.license          = { :type => 'MIT', :file => '../LICENSE' }
  s.author           = { 'Adel Abdelaty' => 'netdur@gmail.com' }
  s.source           = { :path => '.' }

  s.ios.deployment_target = '14.0'
  s.osx.deployment_target = '11.0'

  s.vendored_frameworks = 'Llama.xcframework'
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'

  s.frameworks = 'Metal', 'MetalKit', 'Foundation', 'Accelerate'
  s.dependency 'Flutter'
end
