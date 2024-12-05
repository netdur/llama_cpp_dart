Pod::Spec.new do |s|
    s.name             = 'llama_cpp_dart'
    s.version          = '0.0.8'
    s.summary          = 'A Flutter plugin for llama cpp'
    s.description      = <<-DESC
  A Flutter plugin for llama cpp
                         DESC
    s.homepage         = 'http://example.com'
    s.license          = { :type => 'MIT', :file => '../LICENSE' }
    s.author           = { 'Your Name' => 'your-email@example.com' }
    s.source           = { :path => '.' }
    s.source_files     = 'Classes/**/*'
    s.dependency 'Flutter'
    s.platform         = :ios, '12.0'
    s.swift_version    = '5.0'
  end