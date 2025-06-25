## Building Binaries

The project includes llama.cpp as a submodule and requires native binaries for various platforms. Follow these steps to build the binaries:

### Prerequisites

- Xcode with Command Line Tools
- A valid Apple Developer Team ID

### Build Process

1. Make sure you've initialized the submodules:
   ```bash
   git submodule update --remote src/llama.cpp
   git submodule update --init --recursive
   ```

2. Navigate to the `darwin` directory:
   ```bash
   cd darwin
   ```

3. Create a custom build script (`build.sh`) with these contents:
   ```bash
   #!/bin/bash
   
   # Exit on error
   set -e
   
   # Get the directory where the script is located
   script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   
   # Function to build and fix rpaths for a platform
   build_platform() {
       local platform=$1
       local dev_team=$2
       
       echo "Building for ${platform}..."
       bash "${script_dir}/run_build.sh" src/llama.cpp $dev_team ${platform}
       
       cd "${script_dir}"
       echo "Fixing rpaths for ${platform}..."
       bash "fix_rpath.sh" ${platform}
       cd "${script_dir}/.."
   }
   
   # Replace YOUR_DEVELOPER_TEAM_ID with your actual Apple Developer Team ID
   # Uncomment the platforms you need to build
   build_platform "MAC_ARM64" "YOUR_DEVELOPER_TEAM_ID"
   # build_platform "OS64" "YOUR_DEVELOPER_TEAM_ID"
   # build_platform "SIMULATOR64" "YOUR_DEVELOPER_TEAM_ID"
   # build_platform "SIMULATORARM64" "YOUR_DEVELOPER_TEAM_ID"
   # build_platform "MAC_CATALYST_ARM64" "YOUR_DEVELOPER_TEAM_ID"
   
   echo "Build completed successfully for all platforms."
   ```

4. Make your build script executable:
   ```bash
   chmod +x build.sh
   ```

5. Run the build script:
   ```bash
   ./darwin/build.sh
   ```

### Available Platforms

You can build for various Apple platforms by uncommenting the relevant lines in the build script:

- `MAC_ARM64` - For Apple Silicon Macs
- `OS64` - For iOS devices with arm64 architecture
- `SIMULATOR64` - For iOS Simulator on Intel Macs
- `SIMULATORARM64` - For iOS Simulator on Apple Silicon Macs
- `MAC_CATALYST_ARM64` - For Mac Catalyst on Apple Silicon

### Troubleshooting

- If you encounter "code signature invalid" errors, make sure your Developer Team ID is correct
- For deployment target issues, you may need to modify the `run_build.sh` script to set a higher iOS version (13.0 or above)
- If libraries fail to load at runtime, check the output of:
  ```bash
  DYLD_PRINT_LIBRARIES=1 DYLD_PRINT_RPATHS=1 dart example/simple.dart
  ```

### Notes

- The built libraries will be placed in the `bin/[PLATFORM]` directory
- The build process automatically handles fixing rpaths and code signing for each platform
- Model compatibility depends on the llama.cpp version - ensure you're using a version that supports your target models
