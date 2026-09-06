#include <stdio.h>
#include <stdlib.h>

// 包住每个 CUDA API 调用，出错立刻报出文件、行号和原因。
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                    cudaGetErrorName(err_), __FILE__, __LINE__,           \
                    cudaGetErrorString(err_));                            \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

// kernel 启动本身没有返回值，要靠这两句查它的错误。
#define CUDA_CHECK_KERNEL()                        \
    do {                                           \
        CUDA_CHECK(cudaGetLastError());            \
        CUDA_CHECK(cudaDeviceSynchronize());       \
    } while (0)

// 基于 cudaEvent 的计时器，量的是 GPU 上的耗时（毫秒）。
struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};


__global__ void saxpy_kernel(const float *x, float *y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < n; i += stride) {
        y[i] = 2.0f * x[i] + y[i];
    }
}

int main(int argc, char *argv[]) {    
    int n = atoi(argv[1]);
    
    // n = 0 特判
    if (n == 0) {
        printf("SUM=0\n");
        return 0;
    }
    
    size_t bytes = (size_t)n * sizeof(float);
    
    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    
    if (!h_x || !h_y) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }
    
    for (int i = 0; i < n; i++) {
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = (i % 1024) - 512;
    }
    
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
    
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    
    GpuTimer timer;
    timer.start();
    
    saxpy_kernel<<<blocks_per_grid, threads_per_block>>>(d_x, d_y, n);
    CUDA_CHECK_KERNEL();
    
    float kernel_time_ms = timer.stop_ms();
    
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)h_y[i];
    }
    
    printf("SUM=%.0f n=%d time=%.3f ms\n", sum, n, kernel_time_ms);
    
    free(h_x);
    free(h_y);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    
    return 0;
}