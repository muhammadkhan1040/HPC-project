# Neural Network Acceleration on GPUs: MNIST Classification

This project explores the acceleration of a neural network for MNIST digit classification using GPU optimization techniques. The implementation progresses from a baseline CPU version to highly optimized GPU versions leveraging CUDA and Tensor Cores.

## Project Overview
- **Objective**: Optimize a neural network for MNIST classification using GPU acceleration.
- **Dataset**: MNIST (70,000 grayscale images of handwritten digits, 28×28 pixels).
- **Neural Network Architecture**: 
  - Input Layer: 784 neurons (one per pixel).
  - Hidden Layer: 128 neurons (expanded to 256 in later versions).
  - Output Layer: 10 neurons (one per digit class).

## Implementations
### 1. Baseline CPU Implementation (V1)
- **Description**: Sequential implementation in C with no parallelization.
- **Performance**: 
  - Training Time: 18 seconds.
  - Test Accuracy: ~92-94%.

### 2. Naive GPU Implementation (V2)
- **Description**: Initial GPU port using CUDA with parallelized matrix operations.
- **Performance**: 
  - Training Time: 35.23 seconds (slower than V1 due to overheads).
  - Test Accuracy: ~93%.

### 3. Optimized GPU Implementation (V3)
- **Description**: Enhanced with memory optimizations, tiled matrix multiplication, and improved kernel configurations.
- **Performance**: 
  - Training Time: 8.62 seconds (V3), 21.1 seconds (V3-accurate).
  - Test Accuracy: ~92% (V3), ~93% (V3-accurate).

### 4. Tensor Core Implementation (V4)
- **Description**: Utilizes FP16 computation and Tensor Cores for mixed-precision training.
- **Performance**: 
  - Training Time: 4.23 seconds.
  - Test Accuracy: ~92%.

## Performance Summary
| Version       | Training Time (s) | Speedup vs V1 | Accuracy |
|---------------|-------------------|---------------|----------|
| V1 (CPU)      | 18.00             | 1.00x         | ~93%     |
| V2 (Naive GPU)| 35.23             | 0.51x         | ~93%     |
| V3            | 8.62              | 2.09x         | ~92%     |
| V3-accurate   | 21.10             | 0.85x         | ~93%     |
| V4 (FP16)     | 4.23              | 4.25x         | ~92%     |

## Key Findings
- Proper GPU optimizations (memory management, kernel configurations) are crucial for performance.
- Tensor Cores (FP16 computation) provide the highest speedup (~4.25x vs CPU).
- Trade-offs exist between speed and accuracy in optimized versions.

## Future Work
- Explore advanced architectures (e.g., CNNs).
- Experiment with additional precision modes (TF32, BF16).
- Extend to multi-GPU training.

## Authors
- Ubaida Tariq (22i-1155)
- Muhammad Khan (22i-1040)
- Rihab Rabbani (22i-1345)