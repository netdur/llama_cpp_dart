Pod::Spec.new do |s|
  s.name             = 'llama_cpp_dart'
  s.version          = '0.9.0-dev.12'
  s.summary          = 'llama.cpp xcframework for iOS / macOS via llama_cpp_dart.'
  s.description      = <<-DESC
    Prebuilt llama.cpp + ggml + mtmd as an xcframework, vendored for use from
    Flutter apps that depend on the llama_cpp_dart binding.
  DESC
  s.homepage         = 'https://github.com/netdur/llama_cpp_dart'
  s.license          = { :type => 'MIT' }
  s.author           = { 'netdur' => 'noreply@netdur.dev' }
  s.source           = { :http => 'https://github.com/netdur/llama_cpp_dart/releases/download/v0.9.0-dev.12/llama-xcframework.zip' }

  s.ios.deployment_target = '14.0'
  s.osx.deployment_target = '11.0'

  s.vendored_frameworks = 'llama.xcframework'
  s.frameworks = 'Metal', 'MetalKit', 'Foundation', 'Accelerate'
end
