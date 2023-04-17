#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

void merge(std::vector<int>& vec, std::vector<int>& tmp, int begin, int mid, int end) {
  int left = begin;
  int right = mid+1;
  for (int i=begin; i<=end; i++) { 
    if (left > mid)
      tmp[i] = vec[right++];
    else if (right > end)
      tmp[i] = vec[left++];
    else if (vec[left] <= vec[right])
      tmp[i] = vec[left++];
    else
      tmp[i] = vec[right++]; 
  }
  for (int i=begin; i<=end; i++) { 
    vec[i] = tmp[i];
  }
}

void merge_sort_par(std::vector<int>& vec, std::vector<int>& tmp, int begin, int end) {
  if(begin < end) {
    int mid = (begin + end) / 2;
    #pragma omp task shared(vec,tmp) firstprivate(begin,end,mid)
    merge_sort_par(vec, tmp, begin, mid);
    #pragma omp task shared(vec,tmp) firstprivate(begin,end,mid)
    merge_sort_par(vec, tmp, mid+1, end);
    #pragma omp taskwait
    merge(vec, tmp, begin, mid, end);
  }
}
void merge_sort(std::vector<int>& vec, int begin, int end) {
  std::vector<int> tmp(vec.size());
  #pragma omp parallel shared(vec,tmp) firstprivate(begin,end)
  #pragma omp single
  merge_sort_par(vec, tmp, begin, end);
}

int main() {
  int n = 2000000;
  std::vector<int> vec(n);
  for (int i=0; i<n; i++) {
    vec[i] = rand() % (10 * n);
    if(i<10 || n-10 < i) printf("%d ",vec[i]);
  }
  //std::vector<int> vec2 = vec;std::sort(vec2.begin(), vec2.end());
  printf("\n");
  merge_sort(vec, 0, n-1);
  for (int i=0; i<n; i++) {
    if(i<10 || n-10 < i) printf("%d ",vec[i]);
  }
  printf("\n");

  //verify
  bool ok = true;
  for (int i=1; i<n; ++i) {
    if(vec[i-1]>vec[i]) { ok = false; printf("%d > %d\n", vec[i-1], vec[i]); break; }
  }
  //ok &= (vec == vec2);
  printf("verify: %s", ok ? "ok" : "NG");
  printf("\n\n");
}
