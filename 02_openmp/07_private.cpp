#include <cstdio>

int main() {
  int tmp[4] = {1,1,1,1};
  int a[1] = {1}, *b = tmp, c[1] = {-1};
  
#pragma omp parallel num_threads(4)
  {
#pragma omp for private(a)
    for(int i=0; i<4; i++)
      printf("%d ",++a[0]);
#pragma omp single
    printf("\n");
#pragma omp for firstprivate(b)
    for(int i=0; i<4; i++)
      printf("%d ",++*b);
#pragma omp single
    printf("\n");
#pragma omp for lastprivate(c)
    for(int i=0; i<4; i++)
      printf("%d ",++*c);
#pragma omp single
    printf("\n");
  }
  printf("%d %d %d\n",*a,*b,*c);
}
