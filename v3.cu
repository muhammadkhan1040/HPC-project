#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9
#define TILE_WIDTH 16

// For kernel timing
#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_stop;    \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_stop);

#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);

#define TIMER_END(t)                            \
  cudaEventRecord(t##_stop);                    \
  cudaEventSynchronize(t##_stop);               \
  float t##_time;                               \
  cudaEventElapsedTime(&t##_time, t##_start, t##_stop); \
  printf("Kernel %-30s: %8.3f ms\n", #t, t##_time);

#define TIMER_DESTROY(t)              \
  cudaEventDestroy(t##_start);        \
  cudaEventDestroy(t##_stop);

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

// Activation and forward kernels
__global__ void reluKernel(double* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

__global__ void optimizedForwardHiddenLayerKernel(double* W1, double* b1, double* input, 
                                                double* hidden, int hidden_size, int input_size) {
    __shared__ double input_tile[TILE_WIDTH];
    __shared__ double weight_tile[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    double sum = 0.0;
    
    if (row < hidden_size) {
        sum = b1[row];
        
        for (int p = 0; p < (input_size + TILE_WIDTH - 1) / TILE_WIDTH; ++p) {
            if (p * TILE_WIDTH + tx < input_size && ty == 0) {
                input_tile[tx] = input[p * TILE_WIDTH + tx];
            }
            
            if (p * TILE_WIDTH + tx < input_size) {
                weight_tile[ty][tx] = W1[row * input_size + p * TILE_WIDTH + tx];
            } else {
                weight_tile[ty][tx] = 0.0;
            }
            
            __syncthreads();
            
            for (int k = 0; k < TILE_WIDTH; k++) {
                if (p * TILE_WIDTH + k < input_size) {
                    sum += weight_tile[ty][k] * input_tile[k];
                }
            }
            
            __syncthreads();
        }
        
        hidden[row] = sum;
    }
}

__global__ void forwardOutputLayerKernel(double* W2, double* b2, double* hidden, 
                                       double* output, int output_size, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < output_size) {
        output[i] = b2[i];
        for (int j = 0; j < hidden_size; j++) {
            output[i] += W2[i * hidden_size + j] * hidden[j];
        }
    }
}

// Softmax and gradient kernels
__global__ void softmaxKernel(double* x, double* sum, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] = exp(x[i]);
        atomicAddDouble(sum, x[i]);
    }
}

__global__ void softmaxNormalizeKernel(double* x, double* sum, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] /= *sum;
    }
}

__global__ void computeOutputGradientKernel(double* output, double* target, 
                                          double* d_output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_output[i] = output[i] - target[i];
    }
}

// Backpropagation kernels
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

// Weight update kernels
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

// Bias update kernels
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

// Loss computation kernel
__global__ void computeLossKernel(double* output, double* target, double* loss, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size && target[i] > 0) {
        atomicAddDouble(loss, -log(output[i]));
    }
}

// Alternative softmax implementation
__global__ void softmaxExpKernel(double* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        x[i] = exp(x[i]);
    }
}

__global__ void softmaxSumKernel(double* x, double* sum, int size) {
    __shared__ double sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? x[i] : 0;
    __syncthreads();
    
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        sum[blockIdx.x] = sdata[0];
    }
}


// Neural network structure
typedef struct {
    // Host pointers
    double* W1;  // Weights input->hidden [HIDDEN_SIZE x INPUT_SIZE]
    double* W2;  // Weights hidden->output [OUTPUT_SIZE x HIDDEN_SIZE]
    double* b1;  // Biases hidden layer [HIDDEN_SIZE]
    double* b2;  // Biases output layer [OUTPUT_SIZE]
    
    // Device pointers
    double* d_W1;
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
    double* d_block_sums;
    double* d_loss;
} NeuralNetwork;

// Create and initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!net) {
        fprintf(stderr, "Failed to allocate network\n");
        exit(EXIT_FAILURE);
    }

    // Host memory allocation
    net->W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    net->W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    // Xavier initialization
    srand(time(NULL));
    double w1_scale = sqrt(2.0 / INPUT_SIZE);
    double w2_scale = sqrt(2.0 / HIDDEN_SIZE);
    
    // Initialize weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i * INPUT_SIZE + j] = ((double)rand() / RAND_MAX * 2 - 1) * w1_scale;
        }
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i * HIDDEN_SIZE + j] = ((double)rand() / RAND_MAX * 2 - 1) * w2_scale;
        }
    }

    // Device memory allocation
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_input, INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_output, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_target, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_d_output, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_sum, sizeof(double)));
    
    // For parallel reduction in softmax
    int numBlocks = (OUTPUT_SIZE + 255) / 256;
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_block_sums, numBlocks * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_loss, sizeof(double)));

    // Copy initialized weights to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    return net;
}

// Free network resources
void freeNetwork(NeuralNetwork* net) {
    if (!net) return;

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

    // Free host memory
    free(net->W1);
    free(net->W2);
    free(net->b1);
    free(net->b2);
    free(net);
}



// Optimized forward pass using CUDA with timing
void forward(NeuralNetwork* net, double* input, double* output) {
    // Set up timers
    TIMER_CREATE(forward_hidden);
    TIMER_CREATE(relu);
    TIMER_CREATE(forward_output);
    TIMER_CREATE(softmax);
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    // Compute hidden layer with optimized kernel
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);  // Renamed from blockDim to avoid conflict
    dim3 grid_dim((HIDDEN_SIZE + TILE_WIDTH - 1) / TILE_WIDTH);  // Renamed from gridDim
    
    TIMER_START(forward_hidden);
    optimizedForwardHiddenLayerKernel<<<grid_dim, block_dim>>>(
        net->d_W1, net->d_b1, net->d_input, net->d_hidden, HIDDEN_SIZE, INPUT_SIZE
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    TIMER_END(forward_hidden);
    
    // Apply ReLU activation
    int block_size = 256;  // Renamed from blockSize
    int hidden_blocks = (HIDDEN_SIZE + block_size - 1) / block_size;
    
    TIMER_START(relu);
    reluKernel<<<hidden_blocks, block_size>>>(net->d_hidden, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    TIMER_END(relu);
    
    // Compute output layer
    int output_blocks = (OUTPUT_SIZE + block_size - 1) / block_size;
    
    TIMER_START(forward_output);
    forwardOutputLayerKernel<<<output_blocks, block_size>>>(
        net->d_W2, net->d_b2, net->d_hidden, net->d_output, OUTPUT_SIZE, HIDDEN_SIZE
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    TIMER_END(forward_output);
    
    // Softmax implementation on CPU (for numerical stability)
    TIMER_START(softmax);
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
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    TIMER_END(softmax);
    
    // Copy output back to host if requested
    if (output != NULL) {
        memcpy(output, host_output, OUTPUT_SIZE * sizeof(double));
    }
    
    // Clean up timers
    TIMER_DESTROY(forward_hidden);
    TIMER_DESTROY(relu);
    TIMER_DESTROY(forward_output);
    TIMER_DESTROY(softmax);
}




// Backpropagation using CUDA with timing
void backward(NeuralNetwork* net, double* input, double* target) {
    // Set up timers
    TIMER_CREATE(output_gradient);
    TIMER_CREATE(hidden_gradient);
    TIMER_CREATE(update_output_weights);
    TIMER_CREATE(update_hidden_weights);
    TIMER_CREATE(update_output_bias);
    TIMER_CREATE(update_hidden_bias);
    
    int block_size = 256;
    int hidden_blocks = (HIDDEN_SIZE + block_size - 1) / block_size;
    int output_blocks = (OUTPUT_SIZE + block_size - 1) / block_size;
    
    // Copy input and target to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_target, target, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    // 1. Compute output gradient
    TIMER_START(output_gradient);
    computeOutputGradientKernel<<<output_blocks, block_size>>>(
        net->d_output, net->d_target, net->d_d_output, OUTPUT_SIZE
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    TIMER_END(output_gradient);
    
    // 2. Compute hidden gradient
    TIMER_START(hidden_gradient);
    computeHiddenGradientKernel<<<hidden_blocks, block_size>>>(
        net->d_W2, net->d_d_output, net->d_hidden, net->d_d_hidden, HIDDEN_SIZE, OUTPUT_SIZE
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    TIMER_END(hidden_gradient);
    
    // 3. Update output weights
    dim3 output_weight_block(16, 16);
    dim3 output_weight_grid(
        (OUTPUT_SIZE + output_weight_block.x - 1) / output_weight_block.x,
        (HIDDEN_SIZE + output_weight_block.y - 1) / output_weight_block.y
    );
    
    TIMER_START(update_output_weights);
    updateOutputWeightsKernel<<<output_weight_grid, output_weight_block>>>(
        net->d_W2, net->d_d_output, net->d_hidden, LEARNING_RATE, OUTPUT_SIZE, HIDDEN_SIZE
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    TIMER_END(update_output_weights);
    
    // 4. Update hidden weights
    dim3 hidden_weight_block(16, 16);
    dim3 hidden_weight_grid(
        (HIDDEN_SIZE + hidden_weight_block.x - 1) / hidden_weight_block.x,
        (INPUT_SIZE + hidden_weight_block.y - 1) / hidden_weight_block.y
    );
    
    TIMER_START(update_hidden_weights);
    updateHiddenWeightsKernel<<<hidden_weight_grid, hidden_weight_block>>>(
        net->d_W1, net->d_d_hidden, net->d_input, LEARNING_RATE, HIDDEN_SIZE, INPUT_SIZE
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    TIMER_END(update_hidden_weights);
    
    // 5. Update output bias
    TIMER_START(update_output_bias);
    updateOutputBiasKernel<<<output_blocks, block_size>>>(
        net->d_b2, net->d_d_output, LEARNING_RATE, OUTPUT_SIZE
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    TIMER_END(update_output_bias);
    
    // 6. Update hidden bias
    TIMER_START(update_hidden_bias);
    updateHiddenBiasKernel<<<hidden_blocks, block_size>>>(
        net->d_b1, net->d_d_hidden, LEARNING_RATE, HIDDEN_SIZE
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    TIMER_END(update_hidden_bias);
    
    // Clean up timers
    TIMER_DESTROY(output_gradient);
    TIMER_DESTROY(hidden_gradient);
    TIMER_DESTROY(update_output_weights);
    TIMER_DESTROY(update_hidden_weights);
    TIMER_DESTROY(update_output_bias);
    TIMER_DESTROY(update_hidden_bias);
}


// Batch training implementation with timing
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    // Pre-allocate memory
    double* flatInput = (double*)malloc(INPUT_SIZE * sizeof(double));
    double* flatTarget = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    
    // Create indices array for shuffling
    int* indices = (int*)malloc(numImages * sizeof(int));
    for (int i = 0; i < numImages; i++) {
        indices[i] = i;
    }
    
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        
        // Shuffle indices
        for (int i = 0; i < numImages - 1; i++) {
            int j = i + rand() % (numImages - i);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        // Process in mini-batches
        int batchSize = BATCH_SIZE;
        int numBatches = (numImages + batchSize - 1) / batchSize;
        
        printf("\n--- Epoch %d Training Details ---\n", epoch + 1);
        
        for (int batch = 0; batch < numBatches; batch++) {
            int startIdx = batch * batchSize;
            int endIdx = (startIdx + batchSize < numImages) ? startIdx + batchSize : numImages;
            
            if (batch == 0) {
                printf("\nBatch %d/%d Kernel Timings:\n", batch + 1, numBatches);
            }
            
            for (int i = startIdx; i < endIdx; i++) {
                int idx = indices[i];
                
                // Prepare sample
                memcpy(flatInput, images[idx], INPUT_SIZE * sizeof(double));
                memcpy(flatTarget, labels[idx], OUTPUT_SIZE * sizeof(double));
                
                // Forward pass
                if (batch == 0 && i == startIdx) {
                    forward(net, flatInput, output);
                } else {
                    // Fast forward pass without timing
                    CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, flatInput, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
                    
                    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
                    dim3 grid_dim((HIDDEN_SIZE + TILE_WIDTH - 1) / TILE_WIDTH);
                    optimizedForwardHiddenLayerKernel<<<grid_dim, block_dim>>>(
                        net->d_W1, net->d_b1, net->d_input, net->d_hidden, HIDDEN_SIZE, INPUT_SIZE);
                    
                    int block_size = 256;
                    int hidden_blocks = (HIDDEN_SIZE + block_size - 1) / block_size;
                    reluKernel<<<hidden_blocks, block_size>>>(net->d_hidden, HIDDEN_SIZE);
                    
                    int output_blocks = (OUTPUT_SIZE + block_size - 1) / block_size;
                    forwardOutputLayerKernel<<<output_blocks, block_size>>>(
                        net->d_W2, net->d_b2, net->d_hidden, net->d_output, OUTPUT_SIZE, HIDDEN_SIZE);
                    
                    // Softmax on CPU
                    double host_output[OUTPUT_SIZE];
                    CHECK_CUDA_ERROR(cudaMemcpy(host_output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
                    
                    double max_val = host_output[0];
                    for (int j = 1; j < OUTPUT_SIZE; j++) {
                        if (host_output[j] > max_val) max_val = host_output[j];
                    }
                    
                    double sum = 0.0;
                    for (int j = 0; j < OUTPUT_SIZE; j++) {
                        host_output[j] = exp(host_output[j] - max_val);
                        sum += host_output[j];
                    }
                    
                    for (int j = 0; j < OUTPUT_SIZE; j++) {
                        host_output[j] /= sum;
                    }
                    
                    CHECK_CUDA_ERROR(cudaMemcpy(net->d_output, host_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
                    memcpy(output, host_output, OUTPUT_SIZE * sizeof(double));
                }
                
                // Backward pass
                if (batch == 0 && i == startIdx) {
                    backward(net, flatInput, flatTarget);
                } else {
                    // Fast backward pass without timing
                    CHECK_CUDA_ERROR(cudaMemcpy(net->d_target, flatTarget, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
                    
                    int block_size = 256;
                    int output_blocks = (OUTPUT_SIZE + block_size - 1) / block_size;
                    computeOutputGradientKernel<<<output_blocks, block_size>>>(
                        net->d_output, net->d_target, net->d_d_output, OUTPUT_SIZE);
                    
                    int hidden_blocks = (HIDDEN_SIZE + block_size - 1) / block_size;
                    computeHiddenGradientKernel<<<hidden_blocks, block_size>>>(
                        net->d_W2, net->d_d_output, net->d_hidden, net->d_d_hidden, HIDDEN_SIZE, OUTPUT_SIZE);
                    
                    dim3 weight_block(16, 16);
                    dim3 output_weight_grid(
                        (OUTPUT_SIZE + weight_block.x - 1) / weight_block.x,
                        (HIDDEN_SIZE + weight_block.y - 1) / weight_block.y);
                    updateOutputWeightsKernel<<<output_weight_grid, weight_block>>>(
                        net->d_W2, net->d_d_output, net->d_hidden, LEARNING_RATE, OUTPUT_SIZE, HIDDEN_SIZE);
                    
                    dim3 hidden_weight_grid(
                        (HIDDEN_SIZE + weight_block.x - 1) / weight_block.x,
                        (INPUT_SIZE + weight_block.y - 1) / weight_block.y);
                    updateHiddenWeightsKernel<<<hidden_weight_grid, weight_block>>>(
                        net->d_W1, net->d_d_hidden, net->d_input, LEARNING_RATE, HIDDEN_SIZE, INPUT_SIZE);
                    
                    updateOutputBiasKernel<<<output_blocks, block_size>>>(
                        net->d_b2, net->d_d_output, LEARNING_RATE, OUTPUT_SIZE);
                    
                    updateHiddenBiasKernel<<<hidden_blocks, block_size>>>(
                        net->d_b1, net->d_d_hidden, LEARNING_RATE, HIDDEN_SIZE);
                }
                
                // Calculate accuracy
                int predicted = 0;
                double max_prob = output[0];
                for (int j = 1; j < OUTPUT_SIZE; j++) {
                    if (output[j] > max_prob) {
                        max_prob = output[j];
                        predicted = j;
                    }
                }
                
                int actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (flatTarget[j] == 1.0) {
                        actual = j;
                        break;
                    }
                }
                
                if (predicted == actual) {
                    correct++;
                }
                
                // Calculate loss
                loss -= log(output[actual]);
            }
            
        }
        
        // Epoch statistics
        loss /= numImages;
        double accuracy = (double)correct / numImages * 100.0;
        double epoch_time = get_time(epoch_start);
        
        printf("\n--- Epoch %d Summary ---\n", epoch + 1);
        printf("Average Loss: %.6f\n", loss);
        printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, numImages);
        printf("Epoch Time: %.3f seconds\n", epoch_time);
    }
    
    // Training complete
    double total_time = get_time(total_start);
    printf("\n--- Training Complete ---\n");
    printf("Total Training Time: %.3f seconds\n", total_time);
    
    // Free resources
    free(flatInput);
    free(flatTarget);
    free(output);
    free(indices);
}

// Evaluate model on test data
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    printf("\n--- Evaluating Model on Test Set ---\n");
    
    double* flatInput = (double*)malloc(INPUT_SIZE * sizeof(double));
    double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    
    int correct = 0;
    double loss = 0.0;
    
    clock_t start = clock();
    
    for (int i = 0; i < numImages; i++) {
        memcpy(flatInput, images[i], INPUT_SIZE * sizeof(double));
        
        // Forward pass without timing
        CHECK_CUDA_ERROR(cudaMemcpy(net->d_input, flatInput, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
        
        dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
        dim3 grid_dim((HIDDEN_SIZE + TILE_WIDTH - 1) / TILE_WIDTH);
        optimizedForwardHiddenLayerKernel<<<grid_dim, block_dim>>>(
            net->d_W1, net->d_b1, net->d_input, net->d_hidden, HIDDEN_SIZE, INPUT_SIZE);
        
        int block_size = 256;
        int hidden_blocks = (HIDDEN_SIZE + block_size - 1) / block_size;
        reluKernel<<<hidden_blocks, block_size>>>(net->d_hidden, HIDDEN_SIZE);
        
        int output_blocks = (OUTPUT_SIZE + block_size - 1) / block_size;
        forwardOutputLayerKernel<<<output_blocks, block_size>>>(
            net->d_W2, net->d_b2, net->d_hidden, net->d_output, OUTPUT_SIZE, HIDDEN_SIZE);
        
        // Softmax on CPU
        double host_output[OUTPUT_SIZE];
        CHECK_CUDA_ERROR(cudaMemcpy(host_output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        
        double max_val = host_output[0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (host_output[j] > max_val) max_val = host_output[j];
        }
        
        double sum = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            host_output[j] = exp(host_output[j] - max_val);
            sum += host_output[j];
        }
        
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            host_output[j] /= sum;
        }
        
        memcpy(output, host_output, OUTPUT_SIZE * sizeof(double));
        
        // Get prediction
        int predicted = 0;
        double max_prob = output[0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (output[j] > max_prob) {
                max_prob = output[j];
                predicted = j;
            }
        }
        
        // Get actual label
        int actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (labels[i][j] == 1.0) {
                actual = j;
                break;
            }
        }
        
        if (predicted == actual) {
            correct++;
        }
        
        // Calculate loss
        loss -= log(output[actual]);
        
        // Progress reporting
        if ((i + 1) % 1000 == 0 || i == numImages - 1) {
            printf("Processed %d/%d test images\n", i + 1, numImages);
        }
    }
    
    // Test results
    loss /= numImages;
    double accuracy = (double)correct / numImages * 100.0;
    double eval_time = get_time(start);
    
    printf("\n--- Test Results ---\n");
    printf("Test Loss: %.6f\n", loss);
    printf("Test Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, numImages);
    printf("Evaluation Time: %.3f seconds\n", eval_time);
    
    // Clean up
    free(flatInput);
    free(output);
}


// Load MNIST dataset
void loadMNIST(const char* imagesFile, const char* labelsFile, 
    double*** images, double*** labels, int* numImages) {
FILE* imgFile = fopen(imagesFile, "rb");
FILE* lblFile = fopen(labelsFile, "rb");

if (imgFile == NULL || lblFile == NULL) {
fprintf(stderr, "Error opening MNIST files\n");
exit(EXIT_FAILURE);
}

// Read and verify image file header
unsigned char imgHeader[16];
if (fread(imgHeader, sizeof(unsigned char), 16, imgFile) != 16) {
fprintf(stderr, "Error reading image file header\n");
exit(EXIT_FAILURE);
}

// Read and verify label file header
unsigned char lblHeader[8];
if (fread(lblHeader, sizeof(unsigned char), 8, lblFile) != 8) {
fprintf(stderr, "Error reading label file header\n");
exit(EXIT_FAILURE);
}

// Extract counts (big-endian to little-endian conversion)
int numImg = (imgHeader[4] << 24) | (imgHeader[5] << 16) | 
      (imgHeader[6] << 8) | imgHeader[7];
int numLbl = (lblHeader[4] << 24) | (lblHeader[5] << 16) | 
      (lblHeader[6] << 8) | lblHeader[7];

if (numImg != numLbl) {
fprintf(stderr, "Error: Image count (%d) != Label count (%d)\n", numImg, numLbl);
exit(EXIT_FAILURE);
}

*numImages = numImg;
*images = (double**)malloc(numImg * sizeof(double*));
*labels = (double**)malloc(numImg * sizeof(double*));

unsigned char pixelBuffer[INPUT_SIZE];
unsigned char label;

for (int i = 0; i < numImg; i++) {
// Read image
if (fread(pixelBuffer, sizeof(unsigned char), INPUT_SIZE, imgFile) != INPUT_SIZE) {
  fprintf(stderr, "Error reading image %d\n", i);
  exit(EXIT_FAILURE);
}

// Allocate and normalize image
(*images)[i] = (double*)malloc(INPUT_SIZE * sizeof(double));
for (int j = 0; j < INPUT_SIZE; j++) {
  (*images)[i][j] = pixelBuffer[j] / 255.0;
}

// Read and one-hot encode label
if (fread(&label, sizeof(unsigned char), 1, lblFile) != 1) {
  fprintf(stderr, "Error reading label %d\n", i);
  exit(EXIT_FAILURE);
}

(*labels)[i] = (double*)calloc(OUTPUT_SIZE, sizeof(double));
(*labels)[i][label] = 1.0;
}

fclose(imgFile);
fclose(lblFile);
printf("Successfully loaded %d MNIST images\n", numImg);
}

int main() {
printf("CUDA Neural Network for MNIST\n");
printf("===========================================\n");

// Display CUDA device info
int deviceCount;
cudaGetDeviceCount(&deviceCount);
if (deviceCount == 0) {
fprintf(stderr, "No CUDA devices found\n");
return EXIT_FAILURE;
}

printf("Found %d CUDA device(s):\n", deviceCount);
for (int i = 0; i < deviceCount; i++) {
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, i);
printf("Device %d: %s\n", i, prop.name);
printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
}

// Load MNIST data
double** trainImages;
double** trainLabels;
int numTrainImages;

printf("\nLoading training data...\n");
loadMNIST("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 
   &trainImages, &trainLabels, &numTrainImages);

double** testImages;
double** testLabels;
int numTestImages;

printf("Loading test data...\n");
loadMNIST("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 
   &testImages, &testLabels, &numTestImages);

// Create and train network
printf("\nInitializing neural network...\n");
NeuralNetwork* net = createNetwork();
printf("Network architecture: %d -> %d -> %d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

printf("\nStarting training for %d epochs...\n", EPOCHS);
train(net, trainImages, trainLabels, numTrainImages);

// Evaluate
evaluate(net, testImages, testLabels, numTestImages);

// Cleanup
freeNetwork(net);

for (int i = 0; i < numTrainImages; i++) {
free(trainImages[i]);
free(trainLabels[i]);
}
free(trainImages);
free(trainLabels);

for (int i = 0; i < numTestImages; i++) {
free(testImages[i]);
free(testLabels[i]);
}
free(testImages);
free(testLabels);

printf("\nProgram completed successfully.\n");
return EXIT_SUCCESS;
}