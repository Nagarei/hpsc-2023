#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cooperative_groups.h>

template<size_t N, unsigned int RANGE>
__global__ void setup_bucket(int* arr, unsigned int* bucket)
{
  __shared__ unsigned int s_bucket[RANGE];
  constexpr int blockDim_x = 1024;//blockDim.x
  const int idx = blockIdx.x * blockDim_x + threadIdx.x;
  for (int i = threadIdx.x; i < RANGE; i += blockDim_x) {
    s_bucket[i] = 0;
  }
  

  const int v = idx < N ? arr[idx] : 0;
  __syncthreads();
  if (idx < N) { atomicAdd(&s_bucket[v], 1); }

  __syncthreads();
  for (int i = threadIdx.x; i < RANGE; i += blockDim_x) {
    //bucket_block[i * gridDim.x + blockIdx.x] =  s_bucket[i];
    atomicAdd(&bucket[i], s_bucket[i]);
  }
}

//template<size_t N, unsigned char RANGE>
//__global__ void sum_bucket(unsigned int* bucket, unsigned int* bucket_block) {
//  __shared__ unsigned int s_bucket[RANGE];
//  constexpr int blockDim_x = 1024;//blockDim.x
//  const int idx = blockIdx.x * blockDim_x + threadIdx.x;
//}


template<size_t N, unsigned int RANGE>
__global__ void cumulative_bucket(unsigned int* bucket)
{
  //累積和をとる
  //auto cg = cooperative_groups::this_grid();
  constexpr int blockDim_x = 1024;//blockDim.x
  const int idx = threadIdx.x;
  const int me = idx<<1;

  __shared__ unsigned int s_bucket[RANGE];
  for (int i = threadIdx.x; i < RANGE; i += blockDim_x) {
    s_bucket[i] = bucket[i];
  }

  //up-sweep
  size_t s = 1;
  for (;s < std::min<unsigned int>(RANGE, 32<<1); s <<= 1) {
    for (int offset = 0; offset < RANGE; offset += (blockDim_x<<1)) {
      const int i = offset + me;
      if (i < RANGE && (i + 1) % s == 0) {
        s_bucket[i] = s_bucket[i] + s_bucket[i - s];
      }
    }
    __syncwarp();
  }
  for (;s < RANGE; s <<= 1) {
    for (int offset = 0; offset < RANGE; offset += (blockDim_x<<1)) {
      const int i = offset + me;
      if (i < RANGE && (i + 1) % s == 0) {
        s_bucket[i] = s_bucket[i] + s_bucket[i - s];
      }
    }
    __syncthreads();
  }
  //down-sweep
  s_bucket[RANGE - 1] = 0;
  s >>= 1;
  for (; s > (32<<1); s >>= 1) {
    for (int offset = 0; offset < RANGE; offset += (blockDim_x<<1)) {
      const int i = offset + me;
      if (i < RANGE && (i + 1) % s == 0) {
        const auto tmp = s_bucket[i - s];
        s_bucket[i - s] = s_bucket[i];
        s_bucket[i] = s_bucket[i] + tmp;
      }
    }
    __syncthreads();
  }
  for (; s > 0; s >>= 1) {
    for (int offset = 0; offset < RANGE; offset += (blockDim_x<<1)) {
      const int i = offset + me;
      if (i < RANGE && (i + 1) % s == 0) {
        const auto tmp = s_bucket[i - s];
        s_bucket[i - s] = s_bucket[i];
        s_bucket[i] = s_bucket[i] + tmp;
      }
    }
    __syncwarp();
  }

  //write back
  for (int i = threadIdx.x; i < RANGE; i += blockDim_x) {
    bucket[i] = s_bucket[i];
  }
}

template<size_t N, unsigned int RANGE>
__global__ void output_bucket(int* arr, unsigned int* bucket)
{
  //auto cg = cooperative_groups::this_grid();
  constexpr int blockDim_x = 1024;//blockDim.x
  const int idx = blockIdx.x * blockDim_x + threadIdx.x;

  __shared__ unsigned int s_bucket[RANGE];
  for (int i = threadIdx.x; i < RANGE; i += blockDim_x) {
    s_bucket[i] = bucket[i];
  }
  __syncthreads();

  if (idx < N) {
    int ok = 0, ng = RANGE;  
    while (ng - ok > 1) {
      int mid = (ng - ok)/2;
      if (s_bucket[mid] <= idx) {
        ok = mid;
      } else {
        ng = mid;
      }
    }
    arr[idx] = ok;
  }
}

inline constexpr unsigned int get_min2pow(unsigned int value, unsigned int work = 1) {
    return ((value <= work) ? (work) : (get_min2pow(value, work * 2)));
}
template<size_t N, unsigned int RANGE_>
void bucket_sort(int* begin) {
  constexpr unsigned int RANGE = get_min2pow(RANGE_);
  int* d_arr = nullptr;
  unsigned int* d_bucket = nullptr;
  constexpr int block_num = (N+1023)/1024;
  //cudaMalloc(&d_arr, N * sizeof(int));
  //cudaMalloc(&d_bucket, RANGE * sizeof(unsigned int));
  //cudaMemcpyAsync(d_arr, begin, N * sizeof(int), cudaMemcpyHostToDevice);
  //cudaMemsetAsync(d_bucket, 0, RANGE * sizeof(unsigned int));

  //setup_bucket<N,RANGE><<<block_num,1024>>>(d_arr, d_bucket);
  //cumulative_bucket<N,RANGE><<<1,1024>>>(d_bucket);
  //output_bucket<N,RANGE><<<block_num,1024>>>(d_arr, d_bucket);

  cudaFuncAttributes attr;attr.maxThreadsPerBlock = 0;
  cudaFuncGetAttributes(&attr, setup_bucket<N,RANGE>);
  printf("%d\n", attr.maxThreadsPerBlock);attr.maxThreadsPerBlock = 0;
  cudaFuncGetAttributes(&attr, cumulative_bucket<N,RANGE>);
  printf("%d\n", attr.maxThreadsPerBlock);attr.maxThreadsPerBlock = 0;
  cudaFuncGetAttributes(&attr, output_bucket<N,RANGE>);
  printf("%d\n", attr.maxThreadsPerBlock);

  //cudaMemcpy(begin, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemset(begin, -1, 5 * sizeof(int));
  cudaDeviceSynchronize();
}

int main() {
  constexpr int n = 50;
  constexpr int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  bucket_sort<n,range>(key.data());

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
