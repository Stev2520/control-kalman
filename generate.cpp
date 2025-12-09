#include <fstream>
#include <array>
int main()
{
    const double t0 = 0, t1 = 10, step = 0.01;
    const size_t n = (t1 - t0) / step;
    std::array<double, n> times;
    times[0] = t0;      times[n - 1] = t1; 
    for (size_t i = 0; ++i < n; times[i] = t0 + i * step);
    std::ofstream out("times.bin", std::ios::binary);
    out.write((char*)times.data(), n * sizeof(double));
    return 0;
}