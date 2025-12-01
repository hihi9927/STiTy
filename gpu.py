#!/usr/bin/env python3
"""
Simple GPU status checker
Shows if GPU is available and being used, with device information
"""

import sys

def check_gpu():
    """Check GPU availability and usage"""
    try:
        import torch

        print("="*70)
        print("GPU Status Check")
        print("="*70)

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA Available: {cuda_available}")

        if cuda_available:
            # Number of GPUs
            gpu_count = torch.cuda.device_count()
            print(f"Number of GPUs: {gpu_count}")

            # Current GPU
            current_device = torch.cuda.current_device()
            print(f"\nCurrent GPU Device: {current_device}")

            # GPU details
            for i in range(gpu_count):
                print(f"\n--- GPU {i} ---")
                print(f"Name: {torch.cuda.get_device_name(i)}")

                # Memory info
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)  # GB

                print(f"Total Memory: {total_memory:.2f} GB")
                print(f"Allocated Memory: {allocated_memory:.2f} GB")
                print(f"Reserved Memory: {reserved_memory:.2f} GB")
                print(f"Free Memory: {total_memory - reserved_memory:.2f} GB")

            # Test GPU with a simple operation
            print("\n" + "="*70)
            print("Testing GPU with simple tensor operation...")
            print("="*70)

            test_tensor = torch.randn(1000, 1000).cuda()
            result = test_tensor @ test_tensor.T

            print(f"✓ GPU is working! Test tensor on device: {result.device}")
            print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")

            # Clean up
            del test_tensor
            del result
            torch.cuda.empty_cache()

        else:
            print("\n⚠ No GPU available. Running on CPU.")
            print("Possible reasons:")
            print("  - No NVIDIA GPU installed")
            print("  - CUDA not installed")
            print("  - PyTorch installed without CUDA support")

        print("\n" + "="*70)

    except ImportError:
        print("Error: PyTorch is not installed.")
        print("Install with: pip install torch")
        sys.exit(1)
    except Exception as e:
        print(f"Error checking GPU: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    check_gpu()
