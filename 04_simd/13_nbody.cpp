#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m256 xvec = _mm256_loadu_ps(x);
  __m256 yvec = _mm256_loadu_ps(y);
  __m256 mvec = _mm256_loadu_ps(m);
  //__m256i index_vec = _m256i_set_si256(0,1,2,3,4,5,6,7);
  for(int i=0; i<N; i++) {
    //static_assert(N==8,"simd");
    __m256 rx = _mm256_sub_ps(_mm256_set1_ps(x[i]), xvec);
    __m256 ry = _mm256_sub_ps(_mm256_set1_ps(y[i]), yvec);
    __m256 r2 = _mm256_add_ps(_mm256_mul_ps(rx, rx), _mm256_mul_ps(ry, ry));
    __m256 rcp_r = _mm256_rsqrt_ps(r2);
    __m256 tmp = _mm256_mul_ps(_mm256_div_ps(mvec, r2), rcp_r);
    //__m256 mask = _mm256_cmpep_si256(index_vec, _mm256_set1_si256(i));
    //const int mask = 1 << i;
    //tmp = _mm256_blend_ps(tmp, _mm256_setzero_ps(), mask);
    #define DO_MASK(IND) else if (i == (IND)) { tmp = _mm256_blend_ps(tmp, _mm256_setzero_ps(), 1<<(IND)); }
    if(0){} DO_MASK(0) DO_MASK(1) DO_MASK(2) DO_MASK(3) DO_MASK(4) DO_MASK(5) DO_MASK(6) DO_MASK(7) 
    #undef DO_MASK
    __m256 fxadd = _mm256_mul_ps(rx, tmp);
    __m256 fyadd = _mm256_mul_ps(ry, tmp);
    //reduction
    // https://qiita.com/beru/items/fff00c19968685dada68#__m256-%E5%9E%8B%E3%81%AE%E5%8D%98%E7%B2%BE%E5%BA%A6%E6%B5%AE%E5%8B%95%E5%B0%8F%E6%95%B0%E7%82%B9%E6%95%B08%E8%A6%81%E7%B4%A0%E3%81%AE%E5%90%88%E8%A8%88%E5%87%A6%E7%90%862%E5%A4%89%E6%95%B0%E3%81%BE%E3%81%A8%E3%82%81%E3%81%A6
    // https://qiita.com/fukushima1981/items/1cd5fa6ea1caac1ddb58   
    __m256 lo_xy = _mm256_permute2f128_ps(fxadd, fyadd, 0x20);
    __m256 hi_xy = _mm256_permute2f128_ps(fxadd, fyadd, 0x31);
    __m256 sum = _mm256_add_ps(lo_xy, hi_xy);
    sum = _mm256_add_ps(sum, _mm256_permute_ps(sum, _MM_SHUFFLE(3, 2, 3, 2)));
    sum = _mm256_add_ps(sum, _mm256_permute_ps(sum, _MM_SHUFFLE(3, 2, 1, 1)));

    float tmpbuffer[8];
    _mm256_storeu_ps(tmpbuffer, sum);
    fx[i] -= tmpbuffer[0];
    fy[i] -= tmpbuffer[4];
  }
  for(int i=0; i<N; i++) {
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
