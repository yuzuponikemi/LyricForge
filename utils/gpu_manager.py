"""
GPU and VRAM Management for LyricForge

This module handles device selection and memory management to prevent OOM errors
when running multiple models (Demucs, Whisper, LLM) on the same GPU.

Key features:
- Auto-detection of available compute devices (CUDA, MPS, CPU)
- VRAM usage monitoring
- Memory cleanup between pipeline stages
- Device selection based on availability and user preferences
"""

import gc
import os
import platform
from typing import Dict, Literal, Optional, Tuple

DeviceType = Literal["cuda", "mps", "cpu"]


class GPUManager:
    """
    Manages GPU resources and device selection.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize GPU manager.

        Args:
            config: Configuration dictionary with gpu settings
        """
        self.config = config or {}
        self.gpu_config = self.config.get("gpu", {})
        self.available_devices = self._detect_devices()
        self.current_device: Optional[DeviceType] = None

    def _detect_devices(self) -> Dict[str, bool]:
        """
        Detect available compute devices.

        Returns:
            Dictionary with device availability
        """
        devices = {"cuda": False, "mps": False, "cpu": True}

        # Check CUDA
        try:
            import torch

            devices["cuda"] = torch.cuda.is_available()
        except ImportError:
            pass

        # Check MPS (Apple Silicon)
        try:
            import torch

            if platform.system() == "Darwin":
                devices["mps"] = (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                )
        except ImportError:
            pass

        return devices

    def get_optimal_device(self, prefer_cpu: bool = False) -> DeviceType:
        """
        Get the optimal device for computation.

        Args:
            prefer_cpu: Force CPU usage even if GPU is available

        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if prefer_cpu or self.gpu_config.get("force_cpu", False):
            return "cpu"

        # Priority order: CUDA > MPS > CPU
        if self.available_devices["cuda"]:
            return "cuda"
        elif self.available_devices["mps"] and self.gpu_config.get("prefer_mps", True):
            return "mps"
        else:
            return "cpu"

    def get_device_info(self) -> Dict[str, any]:
        """
        Get information about the current device.

        Returns:
            Dictionary with device information
        """
        info = {
            "available_devices": self.available_devices,
            "current_device": self.current_device,
        }

        try:
            import torch

            if self.available_devices["cuda"]:
                info["cuda_device_count"] = torch.cuda.device_count()
                if torch.cuda.device_count() > 0:
                    info["cuda_device_name"] = torch.cuda.get_device_name(0)
                    info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
                    info["cuda_memory_reserved"] = torch.cuda.memory_reserved(0)
                    info["cuda_memory_total"] = torch.cuda.get_device_properties(
                        0
                    ).total_memory
        except ImportError:
            pass

        return info

    def clear_gpu_memory(self) -> None:
        """
        Clear GPU memory by running garbage collection and emptying cache.

        This should be called between pipeline stages to free up VRAM.
        """
        # Run Python garbage collection
        gc.collect()

        try:
            import torch

            # Clear PyTorch cache
            if self.available_devices["cuda"]:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif self.available_devices["mps"]:
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                if hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()
        except ImportError:
            pass

    def get_memory_usage(self) -> Optional[Tuple[int, int]]:
        """
        Get current GPU memory usage.

        Returns:
            Tuple of (used_memory, total_memory) in bytes, or None if not available
        """
        try:
            import torch

            if self.available_devices["cuda"] and torch.cuda.device_count() > 0:
                allocated = torch.cuda.memory_allocated(0)
                total = torch.cuda.get_device_properties(0).total_memory
                return (allocated, total)
        except ImportError:
            pass

        return None

    def check_memory_availability(self, required_mb: int) -> bool:
        """
        Check if enough GPU memory is available.

        Args:
            required_mb: Required memory in megabytes

        Returns:
            True if enough memory is available or using CPU
        """
        if not self.available_devices["cuda"]:
            return True  # CPU has no VRAM limit

        memory_info = self.get_memory_usage()
        if memory_info is None:
            return True  # Can't check, assume it's OK

        allocated, total = memory_info
        available = total - allocated
        required_bytes = required_mb * 1024 * 1024

        max_usage = self.gpu_config.get("max_vram_usage", 0.8)
        max_allowed = total * max_usage

        return (allocated + required_bytes) <= max_allowed

    def set_cuda_visible_devices(self) -> None:
        """
        Set CUDA_VISIBLE_DEVICES environment variable if specified in config.
        """
        cuda_devices = self.gpu_config.get("cuda_visible_devices")
        if cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    def get_compute_type(self, device: DeviceType) -> str:
        """
        Get the appropriate compute type for the device.

        Args:
            device: Device type

        Returns:
            Compute type string (e.g., 'float16', 'int8', 'float32')
        """
        if device == "cpu":
            return "int8"  # CPU is typically faster with int8
        elif device == "cuda":
            return "float16"  # CUDA supports float16 efficiently
        elif device == "mps":
            return "float16"  # MPS also supports float16
        else:
            return "float32"  # Default fallback


def create_gpu_manager(config: Dict) -> GPUManager:
    """
    Factory function to create a GPU manager.

    Args:
        config: Configuration dictionary

    Returns:
        GPUManager instance
    """
    manager = GPUManager(config)
    manager.set_cuda_visible_devices()
    return manager


# Context manager for automatic memory cleanup
class GPUMemoryContext:
    """
    Context manager for automatic GPU memory cleanup.

    Usage:
        with GPUMemoryContext(gpu_manager):
            # Your GPU-intensive code here
            model.process()
        # Memory is automatically cleaned up here
    """

    def __init__(self, gpu_manager: GPUManager):
        """
        Initialize context.

        Args:
            gpu_manager: GPUManager instance
        """
        self.gpu_manager = gpu_manager

    def __enter__(self) -> GPUManager:
        """Enter context."""
        return self.gpu_manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and clean up memory."""
        self.gpu_manager.clear_gpu_memory()
        return False


def print_device_info(gpu_manager: GPUManager) -> None:
    """
    Print device information in a readable format.

    Args:
        gpu_manager: GPUManager instance
    """
    info = gpu_manager.get_device_info()

    print("=== GPU Manager Device Information ===")
    print(f"Available devices: {info['available_devices']}")
    print(f"Current device: {info.get('current_device', 'Not set')}")

    if "cuda_device_count" in info:
        print(f"\nCUDA devices: {info['cuda_device_count']}")
        if info['cuda_device_count'] > 0:
            print(f"Device name: {info.get('cuda_device_name', 'Unknown')}")
            if "cuda_memory_total" in info:
                total_gb = info['cuda_memory_total'] / (1024**3)
                allocated_gb = info['cuda_memory_allocated'] / (1024**3)
                reserved_gb = info['cuda_memory_reserved'] / (1024**3)
                print(f"Total memory: {total_gb:.2f} GB")
                print(f"Allocated: {allocated_gb:.2f} GB")
                print(f"Reserved: {reserved_gb:.2f} GB")

    print("=" * 40)
