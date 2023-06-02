#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cassert>

constexpr int nx = 41;
constexpr int ny = 41;
constexpr int nt = 500;
constexpr int nit = 50;

constexpr double range_l = 0;
constexpr double range_r = 2;
constexpr double dx = (range_r - range_l) / static_cast<double>(nx - 1);
constexpr double dy = (range_r - range_l) / static_cast<double>(ny - 1);
constexpr double dt = 0.01;
constexpr double rho = 1;
constexpr double nu = 0.02;

template<typename T>
inline constexpr T pow2(const T& v) {
	return v * v;
}


using MATRIX = std::array<std::array<double, nx>, ny>;
std::ofstream pyplot_out{ "pyplot.py" };
void pyplot_init() {
	pyplot_out << R"(
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2, )" << nx << R"()
y = np.linspace(0, 2, )" << ny << R"()
X, Y = np.meshgrid(x, y)

)";
}
void pyplot_array(const MATRIX& m) {
	auto convert = [](double d)->double {
		if (std::isfinite(d)) { return d; }
		if (std::signbit(d)) { return -1e18; }
		return 1e18;
	};
	pyplot_out << "np.array([";
	for (int h = 0; h < ny; ++h) {
		pyplot_out << '[';
		for (auto& vv : m[h]) {
			pyplot_out
				<< convert(vv) << ',';
		}
		pyplot_out << "],";
	}
	pyplot_out << "])\n";
}
void pyplot(const MATRIX& u, const MATRIX& v, const MATRIX& p) {

	pyplot_out << "u=";
	pyplot_array(u);

	pyplot_out << "v=";
	pyplot_array(v);

	pyplot_out << "p=";
	pyplot_array(p);

	pyplot_out << R"(
plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
plt.pause(.001)
plt.clf()
del u
del v
del p
)";
}

void step1(double* out_b, double* prev_u, double* prev_v, int idx) {
	//const int idx = blockIdx.x * blockDim_x + threadIdx.x;
	const auto un = reinterpret_cast<const double(*)[nx]>(prev_u);
	const auto vn = reinterpret_cast<const double(*)[nx]>(prev_v);
	const auto b = reinterpret_cast<double(*)[nx]>(out_b);
	const int h = 1 + (idx / (nx - 2));
	const int w = 1 + (idx % (nx - 2));

	assert(1 <= h && h < ny - 1);
	assert(1 <= w && w < nx - 1);
	if (ny - 1 <= h) { return; }
	b[h][w] = rho * (1 / dt * \
		((un[h][w + 1] - un[h][w - 1]) / (2 * dx) + (vn[h + 1][w] - vn[h - 1][w]) / (2 * dy)) -
		pow2((un[h][w + 1] - un[h][w - 1]) / (2 * dx)) - 2 * ((un[h + 1][w] - un[h - 1][w]) / (2 * dy) *
			(vn[h][w + 1] - vn[h][w - 1]) / (2 * dx)) - pow2((vn[h + 1][w] - vn[h - 1][w]) / (2 * dy)));
}

void step2(double* out_p, double* prev_p, double* now_b, int idx) {
	//const int idx = blockIdx.x * blockDim_x + threadIdx.x;
	const auto p = reinterpret_cast<double(*)[nx]>(out_p);
	const auto pn = reinterpret_cast<const double(*)[nx]>(prev_p);
	const auto b = reinterpret_cast<const double(*)[nx]>(now_b);
	int h = (idx / nx);
	int w = (idx % nx);

	assert(0 <= h && h < ny);
	assert(0 <= w && w < nx);
	if (ny <= h) { return; }
	auto& out = p[h][w];
	if (w == nx - 1) { w = nx - 2; }
	else if (w == 0) { w = 1; }
	if (h == 0) { h = 1; }

	if (h == ny - 1) {
		out = 0;
	}
	else {
		assert(1 <= h && h < ny - 1);
		assert(1 <= w && w < nx - 1);
		out = (
			pow2(dy) * (pn[h][w + 1] + pn[h][w - 1]) +
			pow2(dx) * (pn[h + 1][w] + pn[h - 1][w]) -
			b[h][w] * pow2(dx * dy)
			) / (2 * (pow2(dx) + pow2(dy)));
	}
}

void step3(double* out_u, double* out_v, double* prev_u, double* prev_v, double* now_p, int idx) {
	//const int idx = blockIdx.x * blockDim_x + threadIdx.x;
	const auto u = reinterpret_cast<double(*)[nx]>(out_u);
	const auto v = reinterpret_cast<double(*)[nx]>(out_v);
	const auto un = reinterpret_cast<const double(*)[nx]>(prev_u);
	const auto vn = reinterpret_cast<const double(*)[nx]>(prev_v);
	const auto p = reinterpret_cast<const double(*)[nx]>(now_p);
	int h = (idx / nx);
	int w = (idx % nx);

	assert(0 <= h && h < ny);
	assert(0 <= w && w < nx);
	if (ny <= h) { return; }
	if (h == ny - 1) {
		u[h][w] = 1;
		v[h][w] = 0;
	}
	else if (h == 0 || w == 0 || w == nx - 1) {
		u[h][w] = 0;
		v[h][w] = 0;
	}
	else {
		assert(1 <= h && h < ny - 1);
		assert(1 <= w && w < nx - 1);
		u[h][w] = un[h][w]
			- un[h][w] * dt / dx * (un[h][w] - un[h][w - 1])
			- un[h][w] * dt / dy * (un[h][w] - un[h - 1][w])
			- dt / (2 * rho * dx) * (p[h][w + 1] - p[h][w - 1])
			+ nu * dt / pow2(dx) * (un[h][w + 1] - 2 * un[h][w] + un[h][w - 1])
			+ nu * dt / pow2(dy) * (un[h + 1][w] - 2 * un[h][w] + un[h - 1][w]);
		v[h][w] = vn[h][w]
			- vn[h][w] * dt / dx * (vn[h][w] - vn[h][w - 1])
			- vn[h][w] * dt / dy * (vn[h][w] - vn[h - 1][w])
			- dt / (2 * rho * dx) * (p[h + 1][w] - p[h - 1][w])
			+ nu * dt / pow2(dx) * (vn[h][w + 1] - 2 * vn[h][w] + vn[h][w - 1])
			+ nu * dt / pow2(dy) * (vn[h + 1][w] - 2 * vn[h][w] + vn[h - 1][w]);
	}
}

void run()
{
	//init pyplot
	pyplot_init();

	//static std::array<double, nx> x;
	//for (int i = 0; i < nx; ++i) { x[i] = dx * i; }
	//static std::array<double, ny> y;
	//for (int i = 0; i < ny; ++i) { y[i] = dy * i; }

	static std::array<std::array<double, nx>, ny> u_[2], v_[2], p_[2], b;
	for (int n = 0; n < nt; ++n) {
		//un = u.copy();
		//vn = v.copy();
		auto& u = u_[n % 2];
		auto& un = u_[(n % 2) ^ 1];
		auto& v = v_[n % 2];
		auto& vn = v_[(n % 2) ^ 1];

		for (int h = 1; h < ny - 1; ++h) {
			for (int w = 1; w < nx - 1; ++w) {
				step1(b.data()->data(), un.data()->data(), vn.data()->data(), (h - 1) * (nx - 2) + (w - 1));
			}
		}
		//
		for (int it = 0; it < nit; ++it) {
			auto& p = p_[(n + it) % 2];
			auto& pn = p_[((n + it) % 2) ^ 1];
			//pn = p.copy();
			for (int h = 0; h < ny; ++h) {
				for (int w = 0; w < nx; ++w) {
					step2(p.data()->data(), pn.data()->data(), b.data()->data(), h * nx + w);
				}
			}
		}
		auto& p = p_[((n + nit) % 2) ^ 1];
		for (int h = 0; h < ny; ++h) {
			for (int w = 0; w < nx; ++w) {
				step3(u.data()->data(), v.data()->data(), un.data()->data(), vn.data()->data(), p.data()->data(), h * nx + w);
			}
		}

		pyplot_out << "plt.title(\"" << n << "\")\n";
		pyplot(u, v, p);
	}

	//finalize pyplot
	pyplot_out << "plt.show()" << std::endl;
}

int main()
{
	run();
	system("python3 pyplot.py");
	return 0;
}
