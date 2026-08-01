// swift-tools-version: 5.9

import PackageDescription

// Keep this URL and checksum pinned to the same immutable release asset.
//
// This is necessarily updated *after* the tag it references: CI builds and
// uploads llama-xcframework.zip from the tag, and only then is its checksum
// known. Do not re-tag to pick this commit up — that would rebuild and replace
// the asset, changing the checksum and invalidating this pin again. The value
// that reaches consumers is the one in the published pub.dev package.
let package = Package(
    name: "llama_cpp_dart",
    platforms: [
        .iOS("14.0"),
        .macOS("12.0"),
    ],
    products: [
        .library(
            name: "llama-cpp-dart",
            targets: ["llama_cpp_dart"]
        ),
    ],
    dependencies: [
        .package(name: "FlutterFramework", path: "../FlutterFramework"),
    ],
    targets: [
        .target(
            name: "llama_cpp_dart",
            dependencies: [
                .product(
                    name: "FlutterFramework",
                    package: "FlutterFramework"
                ),
                "llama",
            ]
        ),
        .binaryTarget(
            name: "llama",
            url: "https://github.com/netdur/llama_cpp_dart/releases/download/v0.9.0-dev.12/llama-xcframework.zip",
            checksum: "e50f977d7ffbbd49557b50d04575c44ea2e279b7d50b1a7c574ead41840b7c1d"
        ),
    ]
)
