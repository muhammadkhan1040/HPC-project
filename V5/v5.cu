#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <openacc.h> // Replace OpenMP with OpenACC

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256      // Increased hidden layer size for better representation
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.005  // Lower learning rate for better convergence
#define EPOCHS 3             // Increased number of epochs
#define NUM_CLASSES 10       // Digits 0-9
#define USE_PINNED_MEMORY 1  // Use pinned memory for faster transfers
#define ACC_NUM_GANGS 32    // Increased from 4 to 32 for better parallelization
#define BATCH_SIZE 128      // Increased from 64 to 128 for better throughput
#define MOMENTUM 0.9         // Add momentum for faster convergence
#define WEIGHT_DECAY 0.0001  // Add weight decay for regularization

// Add a flag to track OpenACC availability
#define USE_OPENACC 1  // Set to 0 to disable OpenACC completely

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
__global__ void batchForwardHiddenLayerKernel(double* W1, double* b1, double* input, double* hidden,
                                             int hidden_size, int input_size, int batch_size);
__global__ void batchReluKernel(double* x, int size);
__global__ void batchForwardOutputLayerKernel(double* W2, double* b2, double* hidden, double* output,
                                             int output_size, int hidden_size, int batch_size);
__global__ void batchSoftmaxKernel(double* output, int output_size, int batch_size);
__global__ void batchComputeOutputGradientKernel(double* output, double* target, double* d_output, 
                                                int output_size, int batch_size);
__global__ void batchComputeHiddenGradientKernel(double* W2, double* d_output, double* hidden, double* d_hidden,
                                                int hidden_size, int output_size, int batch_size);
__global__ void updateWeightsBatchKernel(double* weights, double* gradients, double* activations,
                                        double* momentum, double lr, double momentum_factor, double weight_decay,
                                        int rows, int cols, int batch_size);
__global__ void updateBiasesBatchKernel(double* biases, double* gradients, double* momentum,
                                       double lr, double momentum_factor, int size, int batch_size);
__global__ void evaluateBatchKernel(double* output, double* target, int* correct,
                                   int output_size, int batch_size);

// Neural network structure - MOVED UP before the forward declarations of functions that use it
typedef struct {
    double* W1; // Flattened weights for hidden layer
    double* W2; // Flattened weights for output layer
    double* b1; // Biases for hidden layer
    double* b2; // Biases for output layer
    
    double* d_W1; // Device pointers
    double* d_W2;
    double* d_b1;
    double* d_b2;
    double* d_input;
    double* d_hidden;
    double* d_output;
    double* d_target;
    double* d_d_hidden;
    double* d_d_output;
    double* d_sum;
    double* d_block_sums;  // For reduction in softmax
    double* d_loss;
    
    cudaStream_t stream;   // CUDA stream for asynchronous operations
    cudaEvent_t start, stop; // For timing

    // Add momentum fields
    double* W1_momentum;
    double* W2_momentum;
    double* b1_momentum;
    double* b2_momentum;
    double* d_W1_momentum;
    double* d_W2_momentum;
    double* d_b1_momentum;
    double* d_b2_momentum;
} NeuralNetwork;

// Forward declarations for host functions - MOVED to after the struct definition
void forward(NeuralNetwork* net, double* input, double* output);
void forwardBatch(NeuralNetwork* net, double** batch_images, int batchSize);
void backwardBatch(NeuralNetwork* net, double** batch_images, double** batch_labels, int batchSize);
// Add forward declaration for prepareBatchData
void prepareBatchData(double** batchImages, double** batchLabels, double* flatBatchInput, double* flatBatchTarget, int batchSize);

// CUDA kernels
__global__ void reluKernel(double* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

__global__ void forwardHiddenLayerKernel(double* W1, double* b1, double* input, double* hidden, 
                                         int hidden_size, int input_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        hidden[i] = b1[i];
        for (int j = 0; j < input_size; j++) {
            hidden[i] += W1[i * input_size + j] * input[j];
        }
    }
}

__global__ void forwardOutputLayerKernel(double* W2, double* b2, double* hidden, double* output, 
                                         int output_size, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < output_size) {
        output[i] = b2[i];
        for (int j = 0; j < hidden_size; j++) {
            output[i] += W2[i * hidden_size + j] * hidden[j];
        }
    }
}

__global__ void softmaxKernel(double* x, double* sum, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] = exp(x[i]);
        atomicAddDouble(sum, x[i]);  // Using custom atomicAdd for double
    }
}

__global__ void softmaxNormalizeKernel(double* x, double* sum, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] /= *sum;
    }
}

__global__ void computeOutputGradientKernel(double* output, double* target, double* d_output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_output[i] = output[i] - target[i];
    }
}

__global__ void computeHiddenGradientKernel(double* W2, double* d_output, double* hidden, 
                                            double* d_hidden, int hidden_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        d_hidden[i] = 0;
        for (int j = 0; j < output_size; j++) {
            d_hidden[i] += W2[j * hidden_size + i] * d_output[j];
        }
        d_hidden[i] *= (hidden[i] > 0);
    }
}

__global__ void updateOutputWeightsKernel(double* W2, double* d_output, double* hidden, 
                                         double lr, int output_size, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < output_size && j < hidden_size) {
        W2[i * hidden_size + j] -= lr * d_output[i] * hidden[j];
    }
}

__global__ void updateHiddenWeightsKernel(double* W1, double* d_hidden, double* input, 
                                         double lr, int hidden_size, int input_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < hidden_size && j < input_size) {
        W1[i * input_size + j] -= lr * d_hidden[i] * input[j];
    }
}

__global__ void updateOutputBiasKernel(double* b2, double* d_output, double lr, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < output_size) {
        b2[i] -= lr * d_output[i];
    }
}

__global__ void updateHiddenBiasKernel(double* b1, double* d_hidden, double lr, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        b1[i] -= lr * d_hidden[i];
    }
}

__global__ void computeLossKernel(double* output, double* target, double* loss, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size && target[i] > 0) {  // Only compute loss for the actual class (where target is 1)
        atomicAddDouble(loss, -log(output[i]));  // Using custom atomicAdd for double
    }
}

// Alternative version of softmax that doesn't use atomicAdd
__global__ void softmaxExpKernel(double* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] = exp(x[i]);
    }
}

__global__ void softmaxSumKernel(double* x, double* sum, int size) {
    // Use a parallel reduction to compute the sum
    __shared__ double sdata[256];  // Shared memory for the reduction
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load shared memory
    sdata[tid] = (i < size) ? x[i] : 0;
    __syncthreads();
    
    // Do reduction in shared memory
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        sum[blockIdx.x] = sdata[0];
    }
}

// Optimized weight initialization for faster convergence
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Use pinned memory for faster host-device transfers
    if (USE_PINNED_MEMORY) {
        CHECK_CUDA_ERROR(cudaMallocHost(&net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMallocHost(&net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMallocHost(&net->b1, HIDDEN_SIZE * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMallocHost(&net->b2, OUTPUT_SIZE * sizeof(double)));
    } else {
        net->W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
        net->W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
        net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
        net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    }

    // Allocate memory for momentum
    if (USE_PINNED_MEMORY) {
        CHECK_CUDA_ERROR(cudaMallocHost(&net->W1_momentum, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMallocHost(&net->W2_momentum, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMallocHost(&net->b1_momentum, HIDDEN_SIZE * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMallocHost(&net->b2_momentum, OUTPUT_SIZE * sizeof(double)));
    } else {
        net->W1_momentum = (double*)calloc(HIDDEN_SIZE * INPUT_SIZE, sizeof(double));
        net->W2_momentum = (double*)calloc(OUTPUT_SIZE * HIDDEN_SIZE, sizeof(double));
        net->b1_momentum = (double*)calloc(HIDDEN_SIZE, sizeof(double));
        net->b2_momentum = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    }

    // Initialize weights with He initialization (better for ReLU)
    srand(time(NULL));
    double w1_scale = sqrt(2.0 / INPUT_SIZE);
    double w2_scale = sqrt(2.0 / HIDDEN_SIZE);
    
    // Initialize in parallel on CPU with SIMD-friendly pattern
    #if USE_OPENACC
    #pragma acc parallel loop num_gangs(ACC_NUM_GANGS)
    #endif
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i * INPUT_SIZE + j] = ((double)rand() / RAND_MAX * 2 - 1) * w1_scale;
        }
        net->b1[i] = 0.0;
    }

    #if USE_OPENACC
    #pragma acc parallel loop num_gangs(ACC_NUM_GANGS)
    #endif
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i * HIDDEN_SIZE + j] = ((double)rand() / RAND_MAX * 2 - 1) * w2_scale;
        }
        net->b2[i] = 0.0;
    }

    // Allocate device memory with appropriate flags for caching
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(double)));
    
    // Use cudaHostAlloc for the most frequently accessed buffers
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_input, BATCH_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_sum, sizeof(double)));
    
    // For parallel reduction in softmax (max number of blocks we might need)
    int numBlocks = (OUTPUT_SIZE + 255) / 256;
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_block_sums, numBlocks * sizeof(double)));
    
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_loss, sizeof(double)));

    // Allocate device memory for momentum
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W1_momentum, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W2_momentum, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b1_momentum, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b2_momentum, OUTPUT_SIZE * sizeof(double)));
    
    // Initialize momentum to zero
    cudaMemset(net->d_W1_momentum, 0, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMemset(net->d_W2_momentum, 0, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMemset(net->d_b1_momentum, 0, HIDDEN_SIZE * sizeof(double));
    cudaMemset(net->d_b2_momentum, 0, OUTPUT_SIZE * sizeof(double));

    // Use stream to overlap compute and memory operations
    CHECK_CUDA_ERROR(cudaStreamCreate(&net->stream));
    
    // Copy weights and biases to device
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), 
                                     cudaMemcpyHostToDevice, net->stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), 
                                     cudaMemcpyHostToDevice, net->stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), 
                                     cudaMemcpyHostToDevice, net->stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), 
                                     cudaMemcpyHostToDevice, net->stream));
    
    // Pre-allocate events for timing and synchronization
    CHECK_CUDA_ERROR(cudaEventCreate(&net->start));
    CHECK_CUDA_ERROR(cudaEventCreate(&net->stop));

    return net;
}

// Improved forward for batch with better numerical stability
void forwardBatch(NeuralNetwork* net, double** batch_images, int batchSize) {
    int blockSize = 256;
    
    // Use the prepareBatchData function to prepare input batch with OpenACC
    double* flatBatchInput = (double*)malloc(batchSize * INPUT_SIZE * sizeof(double));
    prepareBatchData(batch_images, NULL, flatBatchInput, NULL, batchSize);
    
    // Copy batch input to device asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_input, flatBatchInput, 
                                    batchSize * INPUT_SIZE * sizeof(double), 
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
    batchSoftmaxKernel<<<batchSize, 256, 256 * sizeof(double), net->stream>>>(
        net->d_output, OUTPUT_SIZE, batchSize
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    free(flatBatchInput);
}

// Add these GPU kernel functions for batch processing
__global__ void batchForwardHiddenLayerKernel(double* W1, double* b1, double* input, double* hidden,
                                             int hidden_size, int input_size, int batch_size) {
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (h_idx < hidden_size && b_idx < batch_size) {
        double sum = b1[h_idx];
        for (int i = 0; i < input_size; i++) {
            sum += W1[h_idx * input_size + i] * input[i * batch_size + b_idx];
        }
        hidden[h_idx * batch_size + b_idx] = sum;
    }
}

__global__ void batchReluKernel(double* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

__global__ void batchForwardOutputLayerKernel(double* W2, double* b2, double* hidden, double* output,
                                             int output_size, int hidden_size, int batch_size) {
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (o_idx < output_size && b_idx < batch_size) {
        double sum = b2[o_idx];
        for (int i = 0; i < hidden_size; i++) {
            sum += W2[o_idx * hidden_size + i] * hidden[i * batch_size + b_idx];
        }
        output[o_idx * batch_size + b_idx] = sum;
    }
}

// Fix batch softmax kernel which might be causing numerical issues
__global__ void batchSoftmaxKernel(double* output, int output_size, int batch_size) {
    int b_idx = blockIdx.x;
    if (b_idx >= batch_size) return;
    
    __shared__ double shared_data[256];
    
    // Find max value for numerical stability
    double max_val = -INFINITY;
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        if (output[i * batch_size + b_idx] > max_val) {
            max_val = output[i * batch_size + b_idx];
        }
    }
    
    // Reduce within block to find global max
    shared_data[threadIdx.x] = max_val;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] = fmax(shared_data[threadIdx.x], shared_data[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    max_val = shared_data[0];
    
    // Compute exp(x - max_val) and sum
    double thread_sum = 0.0;
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        double val = exp(output[i * batch_size + b_idx] - max_val);
        output[i * batch_size + b_idx] = val;
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
    
    double sum = shared_data[0];
    
    // Normalize by sum
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        output[i * batch_size + b_idx] /= sum;
    }
}

// Batch version of backward pass for multiple samples at once
void backwardBatch(NeuralNetwork* net, double** batch_images, double** batch_labels, int batchSize) {
    int blockSize = 256;
    
    // Use the prepareBatchData function to prepare target batch with OpenACC
    double* flatBatchTarget = (double*)malloc(batchSize * OUTPUT_SIZE * sizeof(double));
    prepareBatchData(NULL, batch_labels, NULL, flatBatchTarget, batchSize);
    
    // Copy batch targets to device asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(net->d_target, flatBatchTarget, 
                                    batchSize * OUTPUT_SIZE * sizeof(double), 
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
__global__ void batchComputeOutputGradientKernel(double* output, double* target, double* d_output, 
                                                int output_size, int batch_size) {
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (o_idx < output_size && b_idx < batch_size) {
        d_output[o_idx * batch_size + b_idx] = output[o_idx * batch_size + b_idx] - target[o_idx * batch_size + b_idx];
    }
}

__global__ void batchComputeHiddenGradientKernel(double* W2, double* d_output, double* hidden, double* d_hidden,
                                                int hidden_size, int output_size, int batch_size) {
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (h_idx < hidden_size && b_idx < batch_size) {
        double sum = 0.0;
        for (int i = 0; i < output_size; i++) {
            sum += W2[i * hidden_size + h_idx] * d_output[i * batch_size + b_idx];
        }
        d_hidden[h_idx * batch_size + b_idx] = sum * (hidden[h_idx * batch_size + b_idx] > 0);
    }
}

__global__ void updateWeightsBatchKernel(double* weights, double* gradients, double* activations,
                                        double* momentum, double lr, double momentum_factor, double weight_decay,
                                        int rows, int cols, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int i = idx / cols;
        int j = idx % cols;
        
        double update = 0.0;
        for (int b = 0; b < batch_size; b++) {
            update += gradients[i * batch_size + b] * activations[j * batch_size + b];
        }
        
        // Add weight decay (L2 regularization)
        update = (update / batch_size) + weight_decay * weights[idx];
        
        // Update with momentum
        momentum[idx] = momentum_factor * momentum[idx] + lr * update;
        weights[idx] -= momentum[idx];
    }
}

__global__ void updateBiasesBatchKernel(double* biases, double* gradients, double* momentum,
                                       double lr, double momentum_factor, int size, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        double update = 0.0;
        for (int b = 0; b < batch_size; b++) {
            update += gradients[i * batch_size + b];
        }
        
        // Update with momentum
        momentum[i] = momentum_factor * momentum[i] + lr * (update / batch_size);
        biases[i] -= momentum[i];
    }
}

// Replace the train function to use batch processing
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    
    // Prepare batches
    int numBatches = (numImages + BATCH_SIZE - 1) / BATCH_SIZE;
    
    // Create indices for shuffling - use OpenACC to parallelize the initialization
    int* indices = (int*)malloc(numImages * sizeof(int));
    #if USE_OPENACC
    #pragma acc parallel loop num_gangs(ACC_NUM_GANGS)
    #endif
    for (int i = 0; i < numImages; i++) {
        indices[i] = i;
    }
    
    // Pre-allocate memory for batches to avoid repeated allocations
    double** batchImages = (double**)malloc(BATCH_SIZE * sizeof(double*));
    double** batchLabels = (double**)malloc(BATCH_SIZE * sizeof(double*));
    
    // Pre-allocate memory for flattened data
    double* flatBatchInput = (double*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(double));
    double* flatBatchTarget = (double*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(double));
    
    // Learning rate decay with warmup
    double initial_lr = LEARNING_RATE;
    double min_lr = LEARNING_RATE * 0.1;
    
    // Use pinned memory for faster transfers
    int* h_correct;
    cudaMallocHost(&h_correct, sizeof(int));
    int* d_correct;
    CHECK_CUDA_ERROR(cudaMalloc(&d_correct, sizeof(int)));
    
    // Create array for batch losses to track convergence
    double* batchLosses = (double*)malloc(numBatches * sizeof(double));
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        
        // Cosine decay learning rate schedule
        double progress = (double)epoch / EPOCHS;
        double current_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(progress * M_PI));
        
        // Shuffle indices using more efficient algorithm
        #if USE_OPENACC
        #pragma acc parallel loop num_gangs(ACC_NUM_GANGS)
        #endif
        for (int i = 0; i < numImages; i++) {
            int j = rand() % numImages;
            // Use XOR swap to avoid temporary variable
            if (i != j) {
                indices[i] ^= indices[j];
                indices[j] ^= indices[i];
                indices[i] ^= indices[j];
            }
        }
        
        // Loop over batches with potential for OpenACC parallelism
        double epochLoss = 0.0;
        #if USE_OPENACC
        #pragma acc data copy(epochLoss)
        #endif
        for (int batch = 0; batch < numBatches; batch++) {
            int startIdx = batch * BATCH_SIZE;
            int endIdx = min(startIdx + BATCH_SIZE, numImages);
            int currentBatchSize = endIdx - startIdx;
            
            // Create batch arrays without repeated mallocs
            for (int i = 0; i < currentBatchSize; i++) {
                batchImages[i] = images[indices[startIdx + i]];
                batchLabels[i] = labels[indices[startIdx + i]];
            }
            
            // Process batch - forward and backward pass
            forwardBatch(net, batchImages, currentBatchSize);
            backwardBatch(net, batchImages, batchLabels, currentBatchSize);
            
            // Track loss for this batch
            cudaMemcpy(&batchLosses[batch], net->d_loss, sizeof(double), cudaMemcpyDeviceToHost);
            epochLoss += batchLosses[batch];
        }
        
        // Measure accuracy - use optimized sampling for faster evaluation
        int evalSize = min(5000, numImages); // Reduced from 10000 to 5000 for faster epochs
        double evalInterval = numImages / (double)evalSize;
        int correct = 0;
        
        #if USE_OPENACC
        #pragma acc parallel loop reduction(+:correct) num_gangs(ACC_NUM_GANGS)
        #endif
        for (int i = 0; i < evalSize; i++) {
            // Sample at regular intervals rather than the beginning
            int idx = (int)(i * evalInterval) % numImages;
            double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
            forward(net, images[idx], output);
            
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[idx][j] > labels[idx][actual]) actual = j;
            }
            if (pred == actual) correct++;
            
            free(output);
        }
        
        // Print detailed progress
        printf("Epoch %d - LR: %.5f - Train Accuracy: %.2f%% - Loss: %.4f - Time: %.3fs\n",
               epoch + 1, current_lr, (correct / (float)evalSize) * 100, 
               epochLoss/numBatches, get_time(epoch_start));
    }
    
    printf("Total training time: %.3fs\n", get_time(total_start));
    
    // Clean up all allocated memory
    free(indices);
    free(batchImages);
    free(batchLabels);
    free(flatBatchInput);
    free(flatBatchTarget);
    free(batchLosses);
    cudaFreeHost(h_correct);
    cudaFree(d_correct);
}

// Fix the evaluateBatchKernel to correctly count predictions
__global__ void evaluateBatchKernel(double* output, double* target, int* correct,
                                   int output_size, int batch_size) {
    int b_idx = threadIdx.x;
    if (b_idx >= batch_size) return;
    
    int pred_class = 0;
    int true_class = 0;
    double max_pred = output[0 * batch_size + b_idx];
    double max_true = target[0 * batch_size + b_idx];
    
    for (int i = 1; i < output_size; i++) {
        if (output[i * batch_size + b_idx] > max_pred) {
            max_pred = output[i * batch_size + b_idx];
            pred_class = i;
        }
        if (target[i * batch_size + b_idx] > max_true) {
            max_true = target[i * batch_size + b_idx];
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
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
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
    double* device_output = NULL;
    CHECK_CUDA_ERROR(cudaMalloc(&device_output, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpy(device_output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToDevice));
    
    // Do softmax on CPU for small output size (faster than memory transfers and atomics)
    double host_output[OUTPUT_SIZE];
    CHECK_CUDA_ERROR(cudaMemcpy(host_output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Find max for numerical stability
    double max_val = host_output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (host_output[i] > max_val) max_val = host_output[i];
    }
    
    // Compute exp(x - max) and sum
    double sum = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        host_output[i] = exp(host_output[i] - max_val);
        sum += host_output[i];
    }
    
    // Normalize
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        host_output[i] /= sum;
    }
    
    // Copy back to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_output, host_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    cudaFree(device_output);
    
    // Copy output back to host if needed
    if (output != NULL) {
        memcpy(output, host_output, OUTPUT_SIZE * sizeof(double));
    }
}

// Backpropagation using CUDA
void backward(NeuralNetwork* net, double* input, double* target) {
    int blockSize = 256;
    int hiddenBlocks = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    int outputBlocks = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    
    // Copy input and target to device if they aren't already there
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_target, target, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
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
    double* flatInput = (double*)malloc(INPUT_SIZE * sizeof(double));
    double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    
    int correct = 0;
    int batchSize = 100; // Larger batch size for evaluation
    
    // Process in parallel without using OpenACC on the 2D arrays directly
    #if USE_OPENACC
    #pragma acc parallel loop num_gangs(ACC_NUM_GANGS) reduction(+:correct)
    #endif
    for (int i = 0; i < numImages; i++) {
        // Copy current sample to flat array - avoid OpenACC here
        for (int j = 0; j < INPUT_SIZE; j++) {
            flatInput[j] = images[i][j];
        }
        
        // Forward pass
        forward(net, flatInput, output);
        
        // Find prediction and actual class
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        
        if (pred == actual) correct++;
    }
    
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
    
    free(flatInput);
    free(output);
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
    
    // Read and process images one row at a time to avoid OpenACC errors
    unsigned char* pixels = (unsigned char*)malloc(INPUT_SIZE * sizeof(unsigned char));
    
    for (int i = 0; i < numImages; i++) {
        size_t read_result = fread(pixels, sizeof(unsigned char), INPUT_SIZE, file);
        if (read_result != INPUT_SIZE) {
            fprintf(stderr, "Error: Failed to read MNIST image data\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        
        // Process this image with optimized OpenACC
        #if USE_OPENACC
        #pragma acc parallel loop vector vector_length(256) num_gangs(ACC_NUM_GANGS)
        #endif
        for (int j = 0; j < INPUT_SIZE; j++) {
            images[i][j] = pixels[j] / 255.0;
        }
    }
    
    free(pixels);
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
    
    unsigned char* rawLabels = (unsigned char*)malloc(numLabels);
    size_t read_result = fread(rawLabels, sizeof(unsigned char), numLabels, file);
    if (read_result != numLabels) {
        fprintf(stderr, "Error: Failed to read MNIST label data\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    
    // Process labels one row at a time to avoid OpenACC errors
    for (int i = 0; i < numLabels; i++) {
        #if USE_OPENACC
        #pragma acc parallel loop num_gangs(ACC_NUM_GANGS) vector independent
        #endif
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == rawLabels[i]) ? 1.0 : 0.0;
        }
    }
    
    free(rawLabels);
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

// Improved batch data preparation with OpenACC - optimized for speed
void prepareBatchData(double** batchImages, double** batchLabels, double* flatBatchInput, double* flatBatchTarget, int batchSize) {
    // Check if input/target is NULL and skip processing
    if (batchImages != NULL && flatBatchInput != NULL) {
        #if USE_OPENACC
        #pragma acc data present_or_create(flatBatchInput[0:batchSize*INPUT_SIZE])
        #pragma acc parallel loop collapse(2) num_gangs(ACC_NUM_GANGS) vector_length(128)
        #endif
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                flatBatchInput[j * batchSize + i] = batchImages[i][j];
            }
        }
    }
    
    if (batchLabels != NULL && flatBatchTarget != NULL) {
        #if USE_OPENACC
        #pragma acc data present_or_create(flatBatchTarget[0:batchSize*OUTPUT_SIZE])
        #pragma acc parallel loop collapse(2) num_gangs(ACC_NUM_GANGS) vector_length(32)
        #endif
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                flatBatchTarget[j * batchSize + i] = batchLabels[i][j];
            }
        }
    }
}

// Alternative implementation for main() with robust OpenACC handling
int main() {
    printf("MNIST Neural Network (CUDA Implementation with OpenACC - Ultra Optimized)\n\n");
    
    // Initialize CUDA device with higher priority
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    
    // Set to high performance modes
    cudaSetDevice(0);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // Prefer L1 cache for better performance
    
    // Initialize OpenACC with proper variable declaration
    #if USE_OPENACC
    int openacc_available = 1; // Declare and initialize the missing variable
    printf("Initializing OpenACC for accelerated data processing...\n");
    acc_init(acc_device_host);
    #endif
    
    // Load data with OpenACC acceleration
    clock_t start = clock();
    printf("Loading training data...\n");
    
    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    printf("Training images loaded.\n");
    
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    printf("Training labels loaded.\n");
    
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    printf("Test images loaded.\n");
    
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);
    printf("Test labels loaded.\n");
    
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
    
    // Shutdown OpenACC only if we successfully initialized it
    #if USE_OPENACC
    if (openacc_available) {
        // No explicit shutdown - just let the runtime handle it
        printf("OpenACC runtime cleanup handled automatically\n");
    }
    #endif
    
    return 0;
}