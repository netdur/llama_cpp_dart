Pod::Spec.new do |s|
  s.name         = 'llama_cpp_dart'
  s.version      = '0.1.1'
  s.summary      = 'Dart binding for llama.cpp'
  s.description  = 'High-level Dart / Flutter bindings for llama.cpp.'
  s.homepage     = 'https://github.com/netdur/llama_cpp_dart'
  s.license      = { :type => 'MIT', :file => 'LICENSE' }
  s.authors      = { 'Adel Abdelaty' => 'netdur@gmail.com' }

  s.source       = { :path => '.' }

  s.platform     = :ios, '16.4'
  s.swift_version = '5.9'

  s.source_files = []

  s.vendored_frameworks = 'Llama.xcframework'

  s.dependency 'Flutter'
end
