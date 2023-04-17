#include <cstdio>
#include <cstdlib>
#include <vector>

void merge(std::vector<int>& vec, int begin, int mid, int end) {
  std::vector<int> tmp(end-begin+1);
  int left = begin;
  int right = mid+1;
  for (int i=0; i<tmp.size(); i++) { 
    if (left > mid)
      tmp[i] = vec[right++];
    else if (right > end)
      tmp[i] = vec[left++];
    else if (vec[left] <= vec[right])
      tmp[i] = vec[left++];
    else
      tmp[i] = vec[right++]; 
  }
  for (int i=0; i<tmp.size(); i++) 
    vec[begin+i] = tmp[i];
}

void merge_sort_par(std::vector<int>& vec, int begin, int end) {
  if(begin < end) {
    int mid = (begin + end) / 2;
    #pragma omp task shared(vec) firstprivate(begin,end,mid)
    merge_sort_par(vec, begin, mid);
    #pragma omp task shared(vec) firstprivate(begin,end,mid)
    merge_sort_par(vec, mid+1, end);
    #pragma omp taskwait
    merge(vec, begin, mid, end);
  }
}
void merge_sort(std::vector<int>& vec, int begin, int end) {
  #pragma omp parallel shared(vec) firstprivate(begin,end)
  merge_sort_par(vec, begin, end);
}

int main() {
  int n = 200000;
  std::vector<int> vec(n);
  for (int i=0; i<n; i++) {
    vec[i] = rand() % (10 * n);
    if(i<10 || n-10 < i) printf("%d ",vec[i]);
  }
  printf("\n");
  merge_sort(vec, 0, n-1);
  for (int i=0; i<n; i++) {
    if(i<10 || n-10 < i) printf("%d ",vec[i]);
  }
  printf("\n\n");
}
