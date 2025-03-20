Pod::Spec.new do |s|
    s.name             = 'llama_cpp_dart'
    s.version          = '0.0.1'
    s.summary          = 'Flutter plugin for llama.cpp'
    s.description      = <<-DESC
  A Flutter plugin wrapper for llama.cpp to run LLM models locally.
                         DESC
    s.homepage         = 'https://github.com/netdur/llama_cpp_dart'
    s.license          = { :type => 'MIT', :file => '../LICENSE' }
    s.author           = { 'Your Name' => 'your-email@example.com' }
    s.source           = { :path => '.' }
    s.source_files     = 'Classes/**/*'
    s.public_header_files = 'Classes/**/*.h'
    s.dependency 'FlutterMacOS'
    
    s.platform = :osx, '10.11'
    s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
    s.swift_version = '5.0'
    
    # Add dependency on the llama.cpp library if needed
    # s.vendored_libraries = 'Libraries/libllama.dylib'
    # s.xcconfig = { 'OTHER_LDFLAGS' => '-force_load $(PODS_ROOT)/llama_cpp_dart/Libraries/libllama.dylib' }
  end