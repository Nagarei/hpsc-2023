#include <iostream>
#include <omp.h>

void func() {
   #pragma omp parallel
  std::cout << "func hello\n";
}

int main() {
  func();

  omp_set_num_threads(3);
#pragma omp parallel
//num_threads(2)
  std::cout << "hello\n";

  func();
}
