#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 10;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
  int*const bucket_arr = bucket.data();
  std::vector<int> offset(range,0);
  int*const offset_arr = offset.data();
  #pragma omp parallel
  {
    #pragma omp for reduction (+:bucket_arr[0:n])
    for (int i=0; i<n; i++)
      bucket_arr[key[i]]++;
    #pragma omp single
    for (int i=1; i<range; i++) 
      offset_arr[i] = offset_arr[i-1] + bucket_arr[i-1];
    #pragma omp for
    for (int i=0; i<range; i++) {
      int begin = offset[i];
      //#pragma omp for
      for (int j = 0; j < bucket[i]; ++j) {
        key[begin + j] = i;
      }
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
