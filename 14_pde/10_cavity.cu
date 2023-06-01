#include <iostream>
#include <vector>
#include <array>
#include <cmath>

constexpr int nx = 41;
constexpr int ny = 41;
constexpr int nt = 500;
constexpr int nit = 50;

constexpr double range_l = 0;
constexpr double range_r = 2;
constexpr double dx = (range_r - range_l) / static_cast<double>(nx - 1);
constexpr double dy = (range_r - range_l) / static_cast<double>(ny - 1);
constexpr double dt = 0.1;
constexpr double rho = 1;
constexpr double nu = 0.2;
  
template<typename T>
inline constexpr T pow2(const T& v) {
  return v*v;
}


using MATRIX = std::array<std::array<double, nx>, ny>;
void pyplot_array(const MATRIX& m) {
    auto convert = [](double d)->double {
        if (std::isfinite(d)) { return d; }
        if (std::signbit(d)) { return -1e18; }
        return 1e18;
    };
    std::cout << "np.array([\n";
    for(int h = 0; h < ny; ++h){
        std::cout << '[';
        for(auto& vv : m[h]){ std::cout << convert(vv) << ','; }
        std::cout << "],";
    }
    std::cout << "])\n";
}
void pyplot(const MATRIX& u, const MATRIX&  v, const MATRIX& p) {
    //TODO
    std::cout << "u=";
    pyplot_array(u);

    std::cout << "v=";
    pyplot_array(v);

    std::cout << "p=";
    pyplot_array(p);

    std::cout << R"(
plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
plt.pause(.01)
plt.clf()

)";
}

int main()
{
  //init pyplot
  std::cout << R"(
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2, )" << nx << R"()
y = np.linspace(0, 2, )" << ny << R"()
X, Y = np.meshgrid(x, y)

)";

  
  static std::array<double, nx> x;
  for(int i = 0; i < nx; ++i){ x[i] = dx * i; }
  static std::array<double, ny> y;
  for(int i = 0; i < ny; ++i){ y[i] = dy * i; }
  
  static std::array<std::array<double, nx>, ny> u_[2],v_[2],p_[2],b;
  for(int n = 0; n < nt; ++n) {
      //un = u.copy();
      //vn = v.copy();
      auto& u = u_[n%2];
      auto& un = u_[(n%2)^1];
      auto& v = v_[n%2];
      auto& vn = v_[(n%2)^1];
      
      for(int h = 1; h < ny-1; ++h){
          for(int w = 1; w < nx-1; ++w){
              b[h][w] = rho * (1 / dt *\
                      ((un[h][w+1] - un[h][w-1]) / (2 * dx) + (vn[h+1][w] - vn[h-1][w]) / (2 * dy)) -
                      pow2((un[h][w+1] - un[h][w-1]) / (2 * dx)) - 2 * ((un[h+1][w] - un[h-1][w]) / (2 * dy) *
                       (vn[h][w+1] - vn[h][w-1]) / (2 * dx)) - pow2((vn[h+1][w] - vn[h-1][w]) / (2 * dy)));
          }
      }
      //
      for(int it = 0; n < nit; ++n){
          auto& p = p_[(n+it)%2];
          auto& pn = p_[((n+it)%2)^1];
          //pn = p.copy();
          for(int h = 1; h < ny-1; ++h){
              for(int w = 1; w < nx-1; ++w){
                  p[h][w] = (pow2(dy) * (pn[h][w+1] + pn[h][w-1]) +
                             pow2(dx) * (pn[h+1][w] + pn[h-1][w]) -
                             b[h][w] * pow2(dx * dy))
                            / (2 * pow2(dx * dy));
              }
          }
      	  for(int h = 0; h < ny; ++h){ p[h][nx-1] = p[h][nx-2]; }
          for(int w = 0; w < nx; ++w){ p[0][w] = p[1][w]; }
          for(int h = 0; h < ny; ++h){ p[h][0] = p[h][1]; }
          for(int w = 0; w < nx; ++w){ p[ny-1][w] = 0; }
      }
      auto& p = p_[((n+nit)%2)^1];
      for(int h = 1; h < ny-1; ++h){
          for(int w = 1; w < nx-1; ++w){
              u[h][w] = un[h][w] - un[h][w] * dt / dx * (un[h][w] - un[h][w-1])
                                 - un[h][w] * dt / dy * (un[h][w] - un[h-1][w])
                                 - dt / (2 * rho * dx) * (p[h][w+1] - p[h][w-1])
                                 + nu * dt / pow2(dx) * (un[h][w+1] - 2 * un[h][w] + un[h][w-1])
                                 + nu * dt / pow2(dy) * (un[h+1][w] - 2 * un[h][w] + un[h-1][w]);
              v[h][w] = vn[h][w] - vn[h][w] * dt / dx * (vn[h][w] - vn[h][w-1])
                                 - vn[h][w] * dt / dy * (vn[h][w] - vn[h-1][w])
                                 - dt / (2 * rho * dx) * (p[h+1][w] - p[h-1][w])
                                 + nu * dt / pow2(dx) * (vn[h][w+1] - 2 * vn[h][w] + vn[h][w-1])
                                 + nu * dt / pow2(dy) * (vn[h+1][w] - 2 * vn[h][w] + vn[h-1][w]);
          }
      }
      for(int h = 0; h < ny; ++h){
          u[h][0]  = 0;
          u[h][nx-1] = 0;
      }
      for(int w = 0; w < nx; ++w){
          u[0][w]  = 0;
          u[ny-1][w] = 1;
      }
      for(int w = 0; w < nx; ++w){
          v[0][w]  = 0;
          v[ny-1][w] = 0;
      }
      for(int h = 0; h < ny; ++h){
          v[h][0]  = 0;
          v[h][nx-1] = 0;
      }
      
      pyplot(p,u,v);
  }

  //finalize pyplot
  std::cout << "plt.show()" << std::endl;
   
  return 0;
}
