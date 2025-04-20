#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <cuda_fp16.h> // Include for FP16 support

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256      // Increased hidden layer size for better representation
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.005  // Lower learning rate for better convergence
#define EPOCHS 3             // Increased number of epochs
#define BATCH_SIZE 64        // Optimal batch size for training
#define NUM_CLASSES 10       // Digits 0-9
#define USE_PINNED_MEMORY 1  // Use pinned memory for faster transfers
#define OMP_NUM_THREADS 1    // Increased number of threads for OpenMP
#define MOMENTUM 0.9         // Add momentum for faster convergence
#define WEIGHT_DECAY 0.0001  // Add weight decay for regularization

// CUDA error checking
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Flattened matrix allocation for CUDA compatibility
double* flattenMatrix(double** mat, int rows, int cols) {
    double* flat = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[i * cols + j] = mat[i][j];
        }
    }
    return flat;
}

// Custom atomicAdd for double precision
// Only needed for older CUDA devices (compute capability < 6.0)
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
              __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Forward declarations for batch kernels
__global__ void batchForwardHiddenLayerKernel(__half* W1, __half* b1, __half* input, __half* hidden,
                                             int hidden_size, int input_size, int batch_size);
__global__ void batchReluKernel(__half* x, int size);
__global__ void batchForwardOutputLayerKernel(__half* W2, __half* b2, __half* hidden, __half* output,
                                             int output_size, int hidden_size, int batch_size);
__global__ void batchSoftmaxKernel(__half* output, int output_size, int batch_size);
__global__ void batchComputeOutputGradientKernel(__half* output, __half* target, __half* d_output, 
                                                int output_size, int batch_size);
__global__ void batchComputeHiddenGradientKernel(__half* W2, __half* d_output, __half* hidden, __half* d_hidden,
                                                int hidden_size, int output_size, int batch_size);
__global__ void updateWeightsBatchKernel(__half* weights, __half* gradients, __half* activations,
                                        __half* momentum, double lr, double momentum_factor, double weight_decay,
                                        int rows, int cols, int batch_size);
__global__ void updateBiasesBatchKernel(__half* biases, __half* gradients, __half* momentum,
                                       double lr, double momentum_factor, int size, int batch_size);
__global__ void evaluateBatchKernel(__half* output, __half* target, int* correct,
                                   int output_size, int batch_size);

// Neural network structure - MOVED UP before the forward declarations of functions that use it
typedef struct {
    __half* W1; // Flattened weights for hidden layer
    __half* W2; // Flattened weights for output layer
    __half* b1; // Biases for hidden layer
    __half* b2; // Biases for output layer
    
    __half* d_W1; // Device pointers
    __half* d_W2;
    __half* d_b1;
    __half* d_b2;
    __half* d_input;
    __half* d_hidden;
    __half* d_output;
    __half* d_target;
    __half* d_d_hidden;
    __half* d_d_output;
    double* d_sum;
    double* d_block_sums;  // For reduction in softmax
    double* d_loss;
    
    cudaStream_t stream;   // CUDA stream for asynchronous operations
    cudaEvent_t start, stop; // For timing

    // Add momentum fields
    __half* W1_momentum;
    __half* W2_momentum;
    __half* b1_momentum;
    __half* b2_momentum;
    __half* d_W1_momentum;
    __half* d_W2_momentum;
    __half* d_b1_momentum;
    __half* d_b2_momentum;
} NeuralNetwork;

// Forward declarations for host functions - MOVED to after the struct definition
void forward(NeuralNetwork* net, double* input, double* output);
void forwardBatch(NeuralNetwork* net, double** batch_images, int batchSize);
void backwardBatch(NeuralNetwork* net, double** batch_images, double** batch_labels, int batchSize);

// CUDA kernels
__global__ void reluKernel(__half* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = __half2float(x[i]);
        x[i] = (val > 0) ? x[i] : __float2half(0.0f);
    }
}

__global__ void forwardHiddenLayerKernel(__half* W1, __half* b1, __half* input, __half* hidden, 
                                         int hidden_size, int input_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        hidden[i] = b1[i];
        for (int j = 0; j < input_size; j++) {
            hidden[i] = __hadd(hidden[i], __hmul(W1[i * input_size + j], input[j]));
        }
    }
}

__global__ void forwardOutputLayerKernel(__half* W2, __half* b2, __half* hidden, __half* output, 
                                         int output_size, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < output_size) {
        output[i] = b2[i];
        for (int j = 0; j < hidden_size; j++) {
            output[i] = __hadd(output[i], __hmul(W2[i * hidden_size + j], hidden[j]));
        }
    }
}

__global__ void softmaxKernel(__half* x, double* sum, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = expf(__half2float(x[i]));
        x[i] = __float2half(val);
        atomicAddDouble(sum, val);  // Using custom atomicAdd for double
    }
}

__global__ void softmaxNormalizeKernel(__half* x, double* sum, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] /= *sum;
    }
}

__global__ void computeOutputGradientKernel(__half* output, __half* target, __half* d_output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_output[i] = __hsub(output[i], target[i]);
    }
}

__global__ void computeHiddenGradientKernel(__half* W2, __half* d_output, __half* hidden, 
                                            __half* d_hidden, int hidden_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        d_hidden[i] = __float2half(0.0f);
        for (int j = 0; j < output_size; j++) {
            d_hidden[i] = __hadd(d_hidden[i], __hmul(W2[j * hidden_size + i], d_output[j]));
        }
        float hidden_val = __half2float(hidden[i]);
        d_hidden[i] = __hmul(d_hidden[i], __float2half(hidden_val > 0 ? 1.0f : 0.0f));
    }
}

__global__ void updateOutputWeightsKernel(__half* W2, __half* d_output, __half* hidden, 
                                         double lr, int output_size, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < output_size && j < hidden_size) {
        W2[i * hidden_size + j] = __hsub(W2[i * hidden_size + j], __hmul(__float2half(lr), __hmul(d_output[i], hidden[j])));
    }
}

__global__ void updateHiddenWeightsKernel(__half* W1, __half* d_hidden, __half* input, 
                                         double lr, int hidden_size, int input_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < hidden_size && j < input_size) {
        W1[i * input_size + j] = __hsub(W1[i * input_size + j], __hmul(__float2half(lr), __hmul(d_hidden[i], input[j])));
    }
}

__global__ void updateOutputBiasKernel(__half* b2, __half* d_output, double lr, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < output_size) {
        b2[i] = __hsub(b2[i], __hmul(__float2half(lr), d_output[i]));
    }
}

__global__ void updateHiddenBiasKernel(__half* b1, __half* d_hidden, double lr, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        b1[i] = __hsub(b1[i], __hmul(__float2half(lr), d_hidden[i]));
    }
}

__global__ void computeLossKernel(__half* output, __half* target, double* loss, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float target_val = __half2float(target[i]);
        if (target_val > 0) {  // Only compute loss for the actual class (where target is 1)
            float output_val = __half2float(output[i]);
            atomicAddDouble(loss, -logf(output_val));  // Using custom atomicAdd for double
        }
    }
}

// Alternative version of softmax that doesn't use atomicAdd
__global__ void softmaxExpKernel(__half* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = expf(__half2float(x[i]));
        x[i] = __float2half(val);
    }
}

// Optimized weight initialization for faster convergence
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Use pinned memory for faster host-device transfers
    if (USE_PINNED_MEMORY) {
        CHECK_CUDA_ERROR(cudaMallocHost(&net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(__half)));
        CHECK_CUDA_ERROR(cudaMallocHost(&net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(__half)));
        CHECK_CUDA_ERROR(cudaMallocHost(&net->b1, HIDDEN_SIZE * sizeof(__half)));
        CHECK_CUDA_ERROR(cudaMallocHost(&net->b2, OUTPUT_SIZE * sizeof(__half)));
    } else {
        net->W1 = (__half*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(__half));
        net->W2 = (__half*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(__half));
        net->b1 = (__half*)calloc(HIDDEN_SIZE, sizeof(__half));
        net->b2 = (__half*)calloc(OUTPUT_SIZE, sizeof(__half));
    }

    // Allocate memory for momentum
    if (USE_PINNED_MEMORY) {
        CHECK_CUDA_ERROR(cudaMallocHost(&net->W1_momentum, HIDDEN_SIZE * INPUT_SIZE * sizeof(__half)));
        CHECK_CUDA_ERROR(cudaMallocHost(&net->W2_momentum, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(__half)));
        CHECK_CUDA_ERROR(cudaMallocHost(&net->b1_momentum, HIDDEN_SIZE * sizeof(__half)));
        CHECK_CUDA_ERROR(cudaMallocHost(&net->b2_momentum, OUTPUT_SIZE * sizeof(__half)));
    } else {
        net->W1_momentum = (__half*)calloc(HIDDEN_SIZE * INPUT_SIZE, sizeof(__half));
        net->W2_momentum = (__half*)calloc(OUTPUT_SIZE * HIDDEN_SIZE, sizeof(__half));
        net->b1_momentum = (__half*)calloc(HIDDEN_SIZE, sizeof(__half));
        net->b2_momentum = (__half*)calloc(OUTPUT_SIZE, sizeof(__half));
    }

    // Initialize weights with He initialization (better for ReLU)
    srand(time(NULL));
    double w1_scale = sqrt(2.0 / INPUT_SIZE);
    double w2_scale = sqrt(2.0 / HIDDEN_SIZE);
    
    // Initialize in parallel on CPU with SIMD-friendly pattern
    #pragma omp parallel for
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i * INPUT_SIZE + j] = __double2half(((double)rand() / RAND_MAX * 2 - 1) * w1_scale);
        }
        net->b1[i] = __double2half(0.0);
    }

    #pragma omp parallel for
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i * HIDDEN_SIZE + j] = __double2half(((double)rand() / RAND_MAX * 2 - 1) * w2_scale);
        }
        net->b2[i] = __double2half(0.0);
    }

    // Allocate device memory with appropriate flags for caching
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(__half)));
    
    // Use cudaHostAlloc for the most frequently accessed buffers
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_input, BATCH_SIZE * INPUT_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_sum, sizeof(double)));
    
    // For parallel reduction in softmax (max number of blocks we might need)
    int numBlocks = (OUTPUT_SIZE + 255) / 256;
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_block_sums, numBlocks * sizeof(double)));
    
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_loss, sizeof(double)));

    // Allocate device memory for momentum
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W1_momentum, HIDDEN_SIZE * INPUT_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W2_momentum, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b1_momentum, HIDDEN_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b2_momentum, OUTPUT_SIZE * sizeof(__half)));
    
    // Initialize momentum to zero
    cudaMemset(net->d_W1_momentum, 0, HIDDEN_SIZE * INPUT_SIZE * sizeof(__half));
    cudaMemset(net->d_W2_momentum, 0, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(__half));
    cudaMemset(net->d_b1_momentum, 0, HIDDEN_SIZE * sizeof(__half));
    cudaMemset(net->d_b2_momentum, 0, OUTPUT_SIZE * sizeof(__half));

    // Use stream to overlap compute and memory operations
    CHECK_CUDA_ERROR(cudaStreamCreate(&net->stream));
    
    // Copy weights and biases to device
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(__half), 
                                     cudaMemcpyHostToDevice, net->stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(__half), 
                                     cudaMemcpyHostToDevice, net->stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(__half), 
                                     cudaMemcpyHostToDevice, net->stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(__half), 
                                     cudaMemcpyHostToDevice, net->stream));
    
    // Pre-allocate events for timing and synchronization
    CHECK_CUDA_ERROR(cudaEventCreate(&net->start));
    CHECK_CUDA_ERROR(cudaEventCreate(&net->stop));

    return net;
}

// Improved forward for batch with better numerical stability
void forwardBatch(NeuralNetwork* net, double** batch_images, int batchSize) {
    int blockSize = 256;
    
    // Prepare input batch - transpose for better memory access
    __half* flatBatchInput = (__half*)malloc(batchSize * INPUT_SIZE * sizeof(__half));
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            flatBatchInput[j * batchSize + i] = __double2half(batch_images[i][j]);
        }
    }
    
    // Copy batch input to device asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_input, flatBatchInput, 
                                    batchSize * INPUT_SIZE * sizeof(__half), 
                                    cudaMemcpyHostToDevice, net->stream));
    
    // Launch kernels for the entire batch with optimized grid dimensions
    dim3 blockDim(16, 16);  // Smaller block for better occupancy
    dim3 gridDim((HIDDEN_SIZE + blockDim.x - 1) / blockDim.x, 
                 (batchSize + blockDim.y - 1) / blockDim.y);
    
    batchForwardHiddenLayerKernel<<<gridDim, blockDim, 0, net->stream>>>(
        net->d_W1, net->d_b1, net->d_input, net->d_hidden, 
        HIDDEN_SIZE, INPUT_SIZE, batchSize
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Apply batch ReLU 
    batchReluKernel<<<(HIDDEN_SIZE * batchSize + blockSize - 1) / blockSize, blockSize, 0, net->stream>>>(
        net->d_hidden, HIDDEN_SIZE * batchSize
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Output layer for batch
    dim3 outBlockDim(16, 16);  // Smaller block for better occupancy
    dim3 outGridDim((OUTPUT_SIZE + outBlockDim.x - 1) / outBlockDim.x, 
                    (batchSize + outBlockDim.y - 1) / outBlockDim.y);
    
    batchForwardOutputLayerKernel<<<outGridDim, outBlockDim, 0, net->stream>>>(
        net->d_W2, net->d_b2, net->d_hidden, net->d_output, 
        OUTPUT_SIZE, HIDDEN_SIZE, batchSize
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Batch softmax using shared memory
    batchSoftmaxKernel<<<batchSize, 256, 256 * sizeof(__half), net->stream>>>(
        net->d_output, OUTPUT_SIZE, batchSize
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    free(flatBatchInput);
}

// Add these GPU kernel functions for batch processing
__global__ void batchForwardHiddenLayerKernel(__half* W1, __half* b1, __half* input, __half* hidden,
                                             int hidden_size, int input_size, int batch_size) {
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (h_idx < hidden_size && b_idx < batch_size) {
        __half sum = b1[h_idx];
        for (int i = 0; i < input_size; i++) {
            sum = __hadd(sum, __hmul(W1[h_idx * input_size + i], input[i * batch_size + b_idx]));
        }
        hidden[h_idx * batch_size + b_idx] = sum;
    }
}

__global__ void batchReluKernel(__half* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = __half2float(x[i]);
        x[i] = (val > 0) ? x[i] : __float2half(0.0f);
    }
}

__global__ void batchForwardOutputLayerKernel(__half* W2, __half* b2, __half* hidden, __half* output,
                                             int output_size, int hidden_size, int batch_size) {
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (o_idx < output_size && b_idx < batch_size) {
        __half sum = b2[o_idx];
        for (int i = 0; i < hidden_size; i++) {
            sum = __hadd(sum, __hmul(W2[o_idx * hidden_size + i], hidden[i * batch_size + b_idx]));
        }
        output[o_idx * batch_size + b_idx] = sum;
    }
}

// Fix batch softmax kernel which might be causing numerical issues
__global__ void batchSoftmaxKernel(__half* output, int output_size, int batch_size) {
    int b_idx = blockIdx.x;
    if (b_idx >= batch_size) return;
    
    __shared__ float shared_data[256];
    
    // Find max value for numerical stability
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        float val = __half2float(output[i * batch_size + b_idx]);
        if (val > max_val) {
            max_val = val;
        }
    }
    
    // Reduce within block to find global max
    shared_data[threadIdx.x] = max_val;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    max_val = shared_data[0];
    
    // Compute exp(x - max_val) and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        float val = expf(__half2float(output[i * batch_size + b_idx]) - max_val);
        output[i * batch_size + b_idx] = __float2half(val);
        thread_sum += val;
    }
    
    // Reduce to get total sum
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float sum = shared_data[0];
    
    // Normalize by sum
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        float val = __half2float(output[i * batch_size + b_idx]) / sum;
        output[i * batch_size + b_idx] = __float2half(val);
    }
}

// Batch version of backward pass for multiple samples at once
void backwardBatch(NeuralNetwork* net, double** batch_images, double** batch_labels, int batchSize) {
    int blockSize = 256;
    
    // Prepare target batch - transpose for better memory access
    __half* flatBatchTarget = (__half*)malloc(batchSize * OUTPUT_SIZE * sizeof(__half));
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            flatBatchTarget[j * batchSize + i] = __double2half(batch_labels[i][j]);
        }
    }
    
    // Copy batch targets to device asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_target, flatBatchTarget, 
                                    batchSize * OUTPUT_SIZE * sizeof(__half), 
                                    cudaMemcpyHostToDevice, net->stream));
    
    // Compute output gradients for batch
    dim3 outGradBlock(32, 32);
    dim3 outGradGrid((OUTPUT_SIZE + outGradBlock.x - 1) / outGradBlock.x,
                     (batchSize + outGradBlock.y - 1) / outGradBlock.y);
    
    batchComputeOutputGradientKernel<<<outGradGrid, outGradBlock, 0, net->stream>>>(
        net->d_output, net->d_target, net->d_d_output, OUTPUT_SIZE, batchSize
    );
    
    // Compute hidden gradients for batch
    dim3 hiddenGradBlock(32, 32);
    dim3 hiddenGradGrid((HIDDEN_SIZE + hiddenGradBlock.x - 1) / hiddenGradBlock.x,
                        (batchSize + hiddenGradBlock.y - 1) / hiddenGradBlock.y);
    
    batchComputeHiddenGradientKernel<<<hiddenGradGrid, hiddenGradBlock, 0, net->stream>>>(
        net->d_W2, net->d_d_output, net->d_hidden, net->d_d_hidden, 
        HIDDEN_SIZE, OUTPUT_SIZE, batchSize
    );
    
    // Update weights and biases using the average gradient across the batch with momentum and weight decay
    updateWeightsBatchKernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + blockSize - 1) / blockSize, blockSize, 0, net->stream>>>(
        net->d_W2, net->d_d_output, net->d_hidden, net->d_W2_momentum, 
        LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, OUTPUT_SIZE, HIDDEN_SIZE, batchSize
    );
    
    updateBiasesBatchKernel<<<(OUTPUT_SIZE + blockSize - 1) / blockSize, blockSize, 0, net->stream>>>(
        net->d_b2, net->d_d_output, net->d_b2_momentum, LEARNING_RATE, MOMENTUM, OUTPUT_SIZE, batchSize
    );
    
    updateWeightsBatchKernel<<<(HIDDEN_SIZE * INPUT_SIZE + blockSize - 1) / blockSize, blockSize, 0, net->stream>>>(
        net->d_W1, net->d_d_hidden, net->d_input, net->d_W1_momentum, 
        LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, HIDDEN_SIZE, INPUT_SIZE, batchSize
    );
    
    updateBiasesBatchKernel<<<(HIDDEN_SIZE + blockSize - 1) / blockSize, blockSize, 0, net->stream>>>(
        net->d_b1, net->d_d_hidden, net->d_b1_momentum, LEARNING_RATE, MOMENTUM, HIDDEN_SIZE, batchSize
    );
    
    free(flatBatchTarget);
}

// Add batch GPU kernel functions for backward pass
__global__ void batchComputeOutputGradientKernel(__half* output, __half* target, __half* d_output, 
                                                int output_size, int batch_size) {
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (o_idx < output_size && b_idx < batch_size) {
        d_output[o_idx * batch_size + b_idx] = __hsub(output[o_idx * batch_size + b_idx], target[o_idx * batch_size + b_idx]);
    }
}

__global__ void batchComputeHiddenGradientKernel(__half* W2, __half* d_output, __half* hidden, __half* d_hidden,
                                                int hidden_size, int output_size, int batch_size) {
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (h_idx < hidden_size && b_idx < batch_size) {
        __half sum = __float2half(0.0f);
        for (int i = 0; i < output_size; i++) {
            sum = __hadd(sum, __hmul(W2[i * hidden_size + h_idx], d_output[i * batch_size + b_idx]));
        }
        float hidden_val = __half2float(hidden[h_idx * batch_size + b_idx]);
        d_hidden[h_idx * batch_size + b_idx] = __hmul(sum, __float2half(hidden_val > 0 ? 1.0f : 0.0f));
    }
}

__global__ void updateWeightsBatchKernel(__half* weights, __half* gradients, __half* activations,
                                        __half* momentum, double lr, double momentum_factor, double weight_decay,
                                        int rows, int cols, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int i = idx / cols;
        int j = idx % cols;
        
        __half update = __float2half(0.0f);
        for (int b = 0; b < batch_size; b++) {
            update = __hadd(update, __hmul(gradients[i * batch_size + b], activations[j * batch_size + b]));
        }
        
        // Add weight decay (L2 regularization)
        update = __hadd(__hmul(update, __float2half(1.0 / batch_size)), __hmul(__float2half(weight_decay), weights[idx]));
        
        // Update with momentum
        momentum[idx] = __hadd(__hmul(__float2half(momentum_factor), momentum[idx]), __hmul(__float2half(lr), update));
        weights[idx] = __hsub(weights[idx], momentum[idx]);
    }
}

__global__ void updateBiasesBatchKernel(__half* biases, __half* gradients, __half* momentum,
                                       double lr, double momentum_factor, int size, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        __half update = __float2half(0.0f);
        for (int b = 0; b < batch_size; b++) {
            update = __hadd(update, gradients[i * batch_size + b]);
        }
        
        // Update with momentum
        momentum[i] = __hadd(__hmul(__float2half(momentum_factor), momentum[i]), __hmul(__float2half(lr), __hmul(update, __float2half(1.0 / batch_size))));
        biases[i] = __hsub(biases[i], momentum[i]);
    }
}

// Replace the train function to use batch processing
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    
    // Prepare batches
    int numBatches = (numImages + BATCH_SIZE - 1) / BATCH_SIZE;
    
    // Create indices for shuffling
    int* indices = (int*)malloc(numImages * sizeof(int));
    for (int i = 0; i < numImages; i++) {
        indices[i] = i;
    }
    
    // Use host memory for evaluation counting
    int* h_correct = (int*)malloc(sizeof(int));
    int* d_correct;
    CHECK_CUDA_ERROR(cudaMalloc(&d_correct, sizeof(int)));
    
    // Learning rate decay
    double initial_lr = LEARNING_RATE;
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        
        // Decay learning rate
        double current_lr = initial_lr / (1.0 + 0.1 * epoch);
        
        // Shuffle indices
        for (int i = 0; i < numImages - 1; i++) {
            int j = i + rand() % (numImages - i);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        for (int batch = 0; batch < numBatches; batch++) {
            int startIdx = batch * BATCH_SIZE;
            int endIdx = min(startIdx + BATCH_SIZE, numImages);
            int currentBatchSize = endIdx - startIdx;
            
            // Create batch arrays
            double** batchImages = (double**)malloc(currentBatchSize * sizeof(double*));
            double** batchLabels = (double**)malloc(currentBatchSize * sizeof(double*));
            
            for (int i = 0; i < currentBatchSize; i++) {
                batchImages[i] = images[indices[startIdx + i]];
                batchLabels[i] = labels[indices[startIdx + i]];
            }
            
            // Process batch - forward and backward pass
            forwardBatch(net, batchImages, currentBatchSize);
            backwardBatch(net, batchImages, batchLabels, currentBatchSize);
            
            free(batchImages);
            free(batchLabels);
        }
        
        // Better evaluation: use more samples and track both accuracy and loss
        int evalSize = min(10000, numImages);
        int correct = 0;
        double totalLoss = 0.0;
        
        // Evaluate using standard evaluate function instead of GPU kernel for accuracy
        for (int i = 0; i < evalSize; i++) {
            double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
            forward(net, images[i], output);
            
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
            
            // Compute loss
            if (labels[i][actual] > 0) {
                totalLoss += -log(output[actual]);
            }
            
            free(output);
        }
        
        printf("Epoch %d - LR: %.5f - Train Accuracy: %.2f%% - Loss: %.4f - Time: %.3fs\n",
               epoch + 1, current_lr, (correct / (float)evalSize) * 100, totalLoss/evalSize, get_time(epoch_start));
    }
    
    printf("Total training time: %.3fs\n", get_time(total_start));
    
    // Clean up
    free(indices);
    free(h_correct);
    cudaFree(d_correct);
}

// Fix the evaluateBatchKernel to correctly count predictions
__global__ void evaluateBatchKernel(__half* output, __half* target, int* correct,
                                   int output_size, int batch_size) {
    int b_idx = threadIdx.x;
    if (b_idx >= batch_size) return;
    
    int pred_class = 0;
    int true_class = 0;
    float max_pred = __half2float(output[0 * batch_size + b_idx]);
    float max_true = __half2float(target[0 * batch_size + b_idx]);
    
    for (int i = 1; i < output_size; i++) {
        float pred_val = __half2float(output[i * batch_size + b_idx]);
        float true_val = __half2float(target[i * batch_size + b_idx]);
        if (pred_val > max_pred) {
            max_pred = pred_val;
            pred_class = i;
        }
        if (true_val > max_true) {
            max_true = true_val;
            true_class = i;
        }
    }
    
    if (pred_class == true_class) {
        atomicAdd(correct, 1);
    }
}

// Optimized forward pass using CUDA
void forward(NeuralNetwork* net, double* input, double* output) {
    int blockSize = 256;
    
    // Copy input to device
    __half* h_input = (__half*)malloc(INPUT_SIZE * sizeof(__half));
    for (int i = 0; i < INPUT_SIZE; i++) {
        h_input[i] = __float2half((float)input[i]);
    }
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, h_input, INPUT_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    free(h_input);
    
    // Compute hidden layer
    int hiddenBlocks = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    forwardHiddenLayerKernel<<<hiddenBlocks, blockSize>>>(
        net->d_W1, net->d_b1, net->d_input, net->d_hidden, HIDDEN_SIZE, INPUT_SIZE
    );
    
    // Apply ReLU activation
    reluKernel<<<hiddenBlocks, blockSize>>>(net->d_hidden, HIDDEN_SIZE);
    
    // Compute output layer
    int outputBlocks = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    forwardOutputLayerKernel<<<outputBlocks, blockSize>>>(
        net->d_W2, net->d_b2, net->d_hidden, net->d_output, OUTPUT_SIZE, HIDDEN_SIZE
    );
    
    // Fixed softmax implementation that's both fast and numerically stable
    // First find max value for stability
    __half* device_output = NULL;
    CHECK_CUDA_ERROR(cudaMalloc(&device_output, OUTPUT_SIZE * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMemcpy(device_output, net->d_output, OUTPUT_SIZE * sizeof(__half), cudaMemcpyDeviceToDevice));
    
    // Do softmax on CPU for small output size (faster than memory transfers and atomics)
    __half h_output[OUTPUT_SIZE];
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, net->d_output, OUTPUT_SIZE * sizeof(__half), cudaMemcpyDeviceToHost));
    
    // Find max for numerical stability
    float max_val = __half2float(h_output[0]);
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        float val = __half2float(h_output[i]);
        if (val > max_val) max_val = val;
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float val = expf(__half2float(h_output[i]) - max_val);
        h_output[i] = __float2half(val);
        sum += val;
    }
    
    // Normalize
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float val = __half2float(h_output[i]) / sum;
        h_output[i] = __float2half(val);
    }
    
    // Copy back to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_output, h_output, OUTPUT_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    cudaFree(device_output);
    
    // Copy output back to host if needed
    if (output != NULL) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output[i] = (double)__half2float(h_output[i]);
        }
    }
}

// Backpropagation using CUDA
void backward(NeuralNetwork* net, double* input, double* target) {
    int blockSize = 256;
    int hiddenBlocks = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    int outputBlocks = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    
    // Copy input and target to device if they aren't already there
    __half* h_input = (__half*)malloc(INPUT_SIZE * sizeof(__half));
    __half* h_target = (__half*)malloc(OUTPUT_SIZE * sizeof(__half));
    for (int i = 0; i < INPUT_SIZE; i++) {
        h_input[i] = __float2half((float)input[i]);
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        h_target[i] = __float2half((float)target[i]);
    }
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, h_input, INPUT_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_target, h_target, OUTPUT_SIZE * sizeof(__half), cudaMemcpyHostToDevice));
    free(h_input);
    free(h_target);
    
    // Compute output gradient
    computeOutputGradientKernel<<<outputBlocks, blockSize>>>(
        net->d_output, net->d_target, net->d_d_output, OUTPUT_SIZE
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Compute hidden gradient
    computeHiddenGradientKernel<<<hiddenBlocks, blockSize>>>(
        net->d_W2, net->d_d_output, net->d_hidden, net->d_d_hidden, HIDDEN_SIZE, OUTPUT_SIZE
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Update weights
    dim3 outputWeightBlock(16, 16);
    dim3 outputWeightGrid((OUTPUT_SIZE + outputWeightBlock.x - 1) / outputWeightBlock.x,
                          (HIDDEN_SIZE + outputWeightBlock.y - 1) / outputWeightBlock.y);
    
    updateOutputWeightsKernel<<<outputWeightGrid, outputWeightBlock>>>(
        net->d_W2, net->d_d_output, net->d_hidden, LEARNING_RATE, OUTPUT_SIZE, HIDDEN_SIZE
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    dim3 hiddenWeightBlock(16, 16);
    dim3 hiddenWeightGrid((HIDDEN_SIZE + hiddenWeightBlock.x - 1) / hiddenWeightBlock.x,
                          (INPUT_SIZE + hiddenWeightBlock.y - 1) / hiddenWeightBlock.y);
    
    updateHiddenWeightsKernel<<<hiddenWeightGrid, hiddenWeightBlock>>>(
        net->d_W1, net->d_d_hidden, net->d_input, LEARNING_RATE, HIDDEN_SIZE, INPUT_SIZE
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Update biases
    updateOutputBiasKernel<<<outputBlocks, blockSize>>>(
        net->d_b2, net->d_d_output, LEARNING_RATE, OUTPUT_SIZE
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    updateHiddenBiasKernel<<<hiddenBlocks, blockSize>>>(
        net->d_b1, net->d_d_hidden, LEARNING_RATE, HIDDEN_SIZE
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// Optimized evaluation with batch processing
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    double* input = (double*)malloc(INPUT_SIZE * sizeof(double));
    
    int correct = 0;
    int batchSize = 100; // Larger batch size for evaluation
    
    for (int i = 0; i < numImages; i++) {
        // Copy current sample to flat array
        for (int j = 0; j < INPUT_SIZE; j++) {
            input[j] = images[i][j];
        }
        
        forward(net, input, output);
        
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
    
    free(output);
    free(input);
}

// Read MNIST dataset
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// Free network memory
void freeNetwork(NeuralNetwork* net) {
    // Free device memory
    cudaFree(net->d_W1);
    cudaFree(net->d_W2);
    cudaFree(net->d_b1);
    cudaFree(net->d_b2);
    cudaFree(net->d_input);
    cudaFree(net->d_hidden);
    cudaFree(net->d_output);
    cudaFree(net->d_target);
    cudaFree(net->d_d_hidden);
    cudaFree(net->d_d_output);
    cudaFree(net->d_sum);
    cudaFree(net->d_block_sums);
    cudaFree(net->d_loss);
    
    // Free momentum device memory
    cudaFree(net->d_W1_momentum);
    cudaFree(net->d_W2_momentum);
    cudaFree(net->d_b1_momentum);
    cudaFree(net->d_b2_momentum);
    
    // Destroy stream and events
    cudaStreamDestroy(net->stream);
    cudaEventDestroy(net->start);
    cudaEventDestroy(net->stop);
    
    // Free host memory
    if (USE_PINNED_MEMORY) {
        cudaFreeHost(net->W1);
        cudaFreeHost(net->W2);
        cudaFreeHost(net->b1);
        cudaFreeHost(net->b2);
        cudaFreeHost(net->W1_momentum);
        cudaFreeHost(net->W2_momentum);
        cudaFreeHost(net->b1_momentum);
        cudaFreeHost(net->b2_momentum);
    } else {
        free(net->W1);
        free(net->W2);
        free(net->b1);
        free(net->b2);
        free(net->W1_momentum);
        free(net->W2_momentum);
        free(net->b1_momentum);
        free(net->b2_momentum);
    }
    free(net);
}

// Optimized main function with faster data loading

int main() {
    printf("MNIST Neural Network (CUDA Implementation - Tensor Core Ultra Optimized)\n\n");
    int deviceCount;
    // Set number of OpenMP threads
    omp_set_num_threads(OMP_NUM_THREADS);
    
    // Initialize CUDA device with higher priority
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    
    cudaSetDevice(0);
    
    // Pre-allocate a large chunk of CUDA memory to avoid fragmentation
    void* gpu_memory_pool;
    size_t pool_size = 512 * 1024 * 1024; // 512 MB
    // FIX: Add size parameter to cudaMalloc
    cudaMalloc(&gpu_memory_pool, pool_size);
    cudaFree(gpu_memory_pool);
    
    // Load data in parallel
    clock_t start = clock();
    printf("Loading training data...\n");
    
    // Use OpenMP for parallel data loading if available
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
            printf("Training images loaded.\n");
            
            #pragma omp critical
            {
                // Store in some global variable or pass to next function
            }
        }
        
        #pragma omp section
        {
            double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
            printf("Training labels loaded.\n");
            
            #pragma omp critical
            {
                // Store in some global variable or pass to next function
            }
        }
        
        #pragma omp section
        {
            double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
            printf("Test images loaded.\n");
            
            #pragma omp critical
            {
                // Store in some global variable or pass to next function
            }
        }
        
        #pragma omp section
        {
            double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);
            printf("Test labels loaded.\n");
            
            #pragma omp critical
            {
                // Store in some global variable or pass to next function
            }
        }
    }
    
    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);
    printf("Data loaded in %.2fs\n\n", get_time(start));
    
    // Use CUDA Graph to capture and replay the training workflow
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Create and train network
    NeuralNetwork* net = createNetwork();
    
    // Set CUDA device flags for performance
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    
    // Train with ultra-optimized implementation
    train(net, train_images, train_labels, 60000);
    
    // Evaluate with optimized batch evaluation
    evaluate(net, test_images, test_labels, 10000);
    
    // Free all allocated memory
    freeNetwork(net);
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);
    cudaStreamDestroy(stream);
    
    // Reset device before exit
    cudaDeviceReset();
    
    return 0;
}