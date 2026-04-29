import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'bindings.dart';
import 'library_loader.dart';

/// What kind of compute device a backend represents. Mirrors
/// `ggml_backend_dev_type` so callers can branch on hardware class
/// without parsing the device name.
enum BackendDeviceType {
  /// Standard CPU (x86_64, arm64).
  cpu,

  /// Discrete or external GPU.
  gpu,

  /// Integrated GPU (CPU-shared memory, e.g. Apple unified memory,
  /// Intel iGPU, Adreno via OpenCL).
  igpu,

  /// Specialized accelerator — Hexagon NPU, Apple Neural Engine, etc.
  accel,

  /// Meta-backend used internally by ggml-backend's scheduler.
  meta,
}

/// One compute device available to llama.cpp at runtime. Read by
/// `LlamaEngine.devices` after engine spawn.
///
/// Whether a given device actually ran the most recent generation is
/// not directly exposed — the ggml-backend scheduler distributes ops
/// across devices, and a model might have layers on Hexagon and CPU
/// simultaneously. The list here tells you what's available; absence
/// of an entry means the backend didn't load (e.g. no `libOpenCL.so`
/// on the device manifest, or libggml-hexagon.so missing from the
/// app's lib path).
final class BackendDevice {
  /// Short identifier — e.g. `CPU`, `Metal`, `HTP0`, `OpenCL0`.
  final String name;

  /// Human-readable description — e.g.
  /// `Apple M1 Max`, `Hexagon (HTP v73)`, `Adreno 740`.
  final String description;

  /// Hardware class.
  final BackendDeviceType type;

  /// Free memory in bytes the backend reports it can allocate. `0` if
  /// the backend doesn't report memory (CPU usually returns `0/0`).
  final int memoryFree;

  /// Total memory in bytes the backend reports.
  final int memoryTotal;

  /// Name of the registered backend that owns this device — e.g. `CPU`,
  /// `Metal`, `Hexagon`, `OpenCL`. Several devices can share a backend
  /// (the Hexagon backend exposes one `HTP*` device per HTP variant
  /// the runtime decided to load).
  final String registryName;

  const BackendDevice({
    required this.name,
    required this.description,
    required this.type,
    required this.memoryFree,
    required this.memoryTotal,
    required this.registryName,
  });

  /// True if this device is the kind apps usually care about for
  /// acceleration (anything other than the plain CPU device).
  bool get isAccelerator => type != BackendDeviceType.cpu;

  Map<String, Object?> toJson() => {
        'name': name,
        'description': description,
        'type': type.name,
        'memory_free': memoryFree,
        'memory_total': memoryTotal,
        'registry_name': registryName,
      };

  @override
  String toString() {
    final mem = memoryTotal > 0
        ? ' [${(memoryFree / 1024 / 1024).toStringAsFixed(0)} / '
            '${(memoryTotal / 1024 / 1024).toStringAsFixed(0)} MiB free]'
        : '';
    return '$name (${type.name}, $registryName): $description$mem';
  }
}

/// Static helpers for enumerating ggml-backend devices.
final class LlamaBackends {
  LlamaBackends._();

  /// Enumerate every device the loaded backends exposed. Call after
  /// `LlamaLibrary.load(...)` (which triggers `ggml_backend_load_all`)
  /// — before that the list is empty.
  static List<BackendDevice> list() {
    final b = LlamaLibrary.bindings;
    final n = b.ggml_backend_dev_count();
    if (n <= 0) return const <BackendDevice>[];

    final devices = <BackendDevice>[];
    final freePtr = calloc<Size>();
    final totalPtr = calloc<Size>();
    try {
      for (var i = 0; i < n; i++) {
        final dev = b.ggml_backend_dev_get(i);
        if (dev == nullptr) continue;

        final namePtr = b.ggml_backend_dev_name(dev);
        final descPtr = b.ggml_backend_dev_description(dev);
        final typeRaw = b.ggml_backend_dev_type$1(dev);

        freePtr.value = 0;
        totalPtr.value = 0;
        b.ggml_backend_dev_memory(dev, freePtr, totalPtr);

        final reg = b.ggml_backend_dev_backend_reg(dev);
        final regNamePtr =
            reg == nullptr ? nullptr : b.ggml_backend_reg_name(reg);

        devices.add(BackendDevice(
          name: _readCString(namePtr),
          description: _readCString(descPtr),
          type: _typeFromRaw(typeRaw),
          memoryFree: freePtr.value,
          memoryTotal: totalPtr.value,
          registryName: _readCString(regNamePtr),
        ));
      }
    } finally {
      calloc.free(freePtr);
      calloc.free(totalPtr);
    }
    return List.unmodifiable(devices);
  }

  static String _readCString(Pointer<Char> ptr) {
    if (ptr == nullptr) return '';
    return ptr.cast<Utf8>().toDartString();
  }

  static BackendDeviceType _typeFromRaw(ggml_backend_dev_type raw) {
    switch (raw) {
      case ggml_backend_dev_type.GGML_BACKEND_DEVICE_TYPE_CPU:
        return BackendDeviceType.cpu;
      case ggml_backend_dev_type.GGML_BACKEND_DEVICE_TYPE_GPU:
        return BackendDeviceType.gpu;
      case ggml_backend_dev_type.GGML_BACKEND_DEVICE_TYPE_IGPU:
        return BackendDeviceType.igpu;
      case ggml_backend_dev_type.GGML_BACKEND_DEVICE_TYPE_ACCEL:
        return BackendDeviceType.accel;
      case ggml_backend_dev_type.GGML_BACKEND_DEVICE_TYPE_META:
        return BackendDeviceType.meta;
    }
  }
}
