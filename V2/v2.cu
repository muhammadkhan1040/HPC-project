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

// Neural network structure
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
} NeuralNetwork;

// Optimized weight initialization for faster convergence
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Allocate host memory
    net->W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    net->W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    // Initialize weights with Xavier initialization for better convergence
    srand(time(NULL));
    double w1_scale = sqrt(2.0 / INPUT_SIZE);
    double w2_scale = sqrt(2.0 / HIDDEN_SIZE);
    
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

    // Allocate device memory
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
    
    // For parallel reduction in softmax (max number of blocks we might need)
    int numBlocks = (OUTPUT_SIZE + 255) / 256;
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_block_sums, numBlocks * sizeof(double)));
    
    CHECK_CUDA_ERROR(cudaMalloc(&net->d_loss, sizeof(double)));

    // Copy weights and biases to device
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    return net;
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

// Batch training implementation to reduce data transfer overhead
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
        
        // Shuffle indices for better convergence
        for (int i = 0; i < numImages - 1; i++) {
            int j = i + rand() % (numImages - i);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        // Process data in mini-batches
        int batchSize = BATCH_SIZE;
        int numBatches = (numImages + batchSize - 1) / batchSize;
        
        for (int batch = 0; batch < numBatches; batch++) {
            int startIdx = batch * batchSize;
            int endIdx = min(startIdx + batchSize, numImages);
            
            for (int i = startIdx; i < endIdx; i++) {
                int idx = indices[i];
                
                // Copy current sample to flat arrays
                memcpy(flatInput, images[idx], INPUT_SIZE * sizeof(double));
                memcpy(flatTarget, labels[idx], OUTPUT_SIZE * sizeof(double));
                
                // Forward pass
                forward(net, flatInput, output);
                
                // Backward pass
                backward(net, flatInput, flatTarget);
                
                // Compute loss & accuracy on CPU
                double sampleLoss = 0.0;
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    if (flatTarget[k] > 0) {
                        // Add small epsilon to avoid log(0)
                        sampleLoss -= log(output[k] > 1e-10 ? output[k] : 1e-10);
                    }
                }
                loss += sampleLoss;
                
                int pred = 0, actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (output[j] > output[pred]) pred = j;
                    if (flatTarget[j] > flatTarget[actual]) actual = j;
                }
                if (pred == actual) correct++;
            }
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
    
    free(flatInput);
    free(flatTarget);
    free(output);
    free(indices);
}

// Optimized evaluation with batch processing
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    double* flatInput = (double*)malloc(INPUT_SIZE * sizeof(double));
    double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    
    int correct = 0;
    int batchSize = 100; // Larger batch size for evaluation
    
    for (int i = 0; i < numImages; i++) {
        // Copy current sample to flat array
        memcpy(flatInput, images[i], INPUT_SIZE * sizeof(double));
        
        forward(net, flatInput, output);
        
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
    
    // Free host memory
    free(net->W1);
    free(net->W2);
    free(net->b1);
    free(net->b2);
    free(net);
}

// Optimized main function
int main() {
    printf("MNIST Neural Network (CUDA Implementation - Optimized)\n\n");
    
    // Initialize CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    
    cudaSetDevice(0); // Use first CUDA device
    
    // Load data with pre-allocation
    clock_t start = clock();
    printf("Loading training data...\n");
    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    printf("Loading test data...\n");
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);
    printf("Data loaded in %.2fs\n\n", get_time(start));
    
    // Create and train network
    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    // Free all allocated memory
    freeNetwork(net);
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);
    
    return 0;
}