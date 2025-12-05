#include <fstream>
#include <iostream>
#include <string>
#include "kalman.hpp"
#include "models.hpp"
int main()
{
    size_t n;
    std::ifstream times("times.bin", std::ios::binary);
    std::ofstream exact("exact.bin", std::ios::binary),
                  noisy("noisy.bin", std::ios::binary),
                  filteredCKF("CKF.bin", std::ios::binary),
                  filteredSRCF("SRCF.bin", std::ios::binary);
    kalman::CKF ckf(Eigen::Vector2d::Zero(2), Eigen::Matrix2d::Identity());
    kalman::SRCF srcf(Eigen::Vector2d::Zero(2), Eigen::Matrix2d::Identity());
    times.read((char*)&n, sizeof(size_t));
    double t_prev, t, dt;
    Eigen::VectorXd x_exact =  Eigen::Vector2d::Zero(), y_exact(2);
    Eigen::MatrixXd A, B, C, D, Q, R;
    Eigen::VectorXd u, w, v, y;
    times.read((char*)&t_prev, sizeof(double));
    while (--n)
    {
        times.read((char*)&t, sizeof(double));
        dt = t - t_prev;
        A = model2::A(dt);
        B = model2::B(dt);
        C = model2::C(t);
        D = model2::D(dt);
        Q = model2::Q(t);
        R = model2::R(t);
        u = model2::u(t);
        w = model2::w(t);
        v = model2::v(t);
        x_exact = A * x_exact + D * u;
        y_exact = C * x_exact;
        y = y_exact + v;
        exact.write((char*)x_exact.data(), sizeof(double) * x_exact.size());
        noisy.write((char*)y.data(), sizeof(double) * y.size());
        ckf.step(A, B, C, D, Q, R, u, y);
        srcf.step(A, B, C, D, Q, R, u, y);
        filteredCKF.write((char*)ckf.state().data(), sizeof(double) * ckf.state().size());
        filteredSRCF.write((char*)srcf.state().data(), sizeof(double) * srcf.state().size());
        t_prev = t;
    }
    return 0;
}