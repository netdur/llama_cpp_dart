// swift-tools-version: 5.9

import PackageDescription

// Keep this URL and checksum pinned to the same immutable release asset.
// The dev.10 URL is suitable for this spike; point the production manifest at
// the next release created by the symlink-preserving `ditto` packaging step.
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
            url: "https://github.com/netdur/llama_cpp_dart/releases/download/v0.9.0-dev.10/llama-xcframework.zip",
            checksum: "58ced0d358281a6d12664ffa27fc70b7299c5cd72397b2607f911544084d12d8"
        ),
    ]
)
