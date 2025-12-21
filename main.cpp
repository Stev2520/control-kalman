#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include "kalman.hpp"
#include "models.hpp"
int main()
{
    const size_t n = 1500;
    const double dt = .01,
                 cQ[] = {.001, .01, .1, 1, 10, 1000},
                 cR[] = {.001, .01, .1, 1, 10, 1000};
    //std::ifstream times("times.bin", std::ios::binary);
    std::ofstream exact("exact.bin", std::ios::binary),
                  noisy("noisy.bin", std::ios::binary),
                  filteredCKF("CKF.bin", std::ios::binary),
                  filteredSRCF("SRCF.bin", std::ios::binary),
                  metricsCKF("ARMSEckf.txt"),
                  metricsSRCF("ARMSEsrcf.txt"),
                  timingsCKF("timingsCKF.txt"),
                  timingsSRCF("timingsSRCF.txt");
    //times.read((char*)&n, sizeof(size_t));
    double t_prev = t0, t, dt;
    Eigen::VectorXd x_exact = Eigen::Vector2d::Zero(), y_exact(2);
    Eigen::MatrixXd A, B, C, D, Q, R;
    Eigen::VectorXd u, w, v, y;
    kalman::CKF ckf(x_exact, Eigen::Matrix2d::Identity());
    kalman::SRCF srcf(x_exact, Eigen::Matrix2d::Identity());
    std::chrono::_V2::system_clock::time_point start, end;
    //times.read((char*)&t_prev, sizeof(double));
    for (unsigned char idx_r = 0; idx_r < 6; ++idx_r)
    {
        R = model2::R() * cR[idx_r];
        for (unsigned char idx_q = 0; idx_q < 6; ++idx_q)
        {
            unsigned long long timeCKF = 0, timeSRCF = 0;
            Q = model2::Q() * cQ[idx_q];
            x_exact = Eigen::Vector2d::Zero();
            ckf.initialize(x_exact, Eigen::Matrix2d::Identity());
            srcf.initialize(x_exact, Eigen::Matrix2d::Identity());
            double lossCKF = 0, lossSRCF = 0;
            for (size_t i = 0; i < n; ++i)
            {
                t = i * dt;
                A = model2::A(dt);
                B = model2::B(dt);
                C = model2::C(t);
                D = model2::D(dt);
                u = model2::u(t);
                w = model2::w(t);
                v = model2::v(t);
                x_exact = A * x_exact + D * u;
                y_exact = C * x_exact;
                y = y_exact + v;
                exact.write((char*)x_exact.data(), sizeof(double) * x_exact.size());
                noisy.write((char*)y.data(), sizeof(double) * y.size());
                start = std::chrono::high_resolution_clock::now();
                ckf.step(A, B, C, D, Q, R, u, y);
                end = std::chrono::high_resolution_clock::now();
                timeCKF = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                start = std::chrono::high_resolution_clock::now();
                srcf.step(A, B, C, D, Q, R, u, y);
                end = std::chrono::high_resolution_clock::now();
                auto error = ckf.state() - x_exact;
                lossCKF += error.squaredNorm();
                error = srcf.state() - x_exact;
                lossSRCF += error.squaredNorm();
                filteredCKF.write((char*)ckf.state().data(), sizeof(double) * ckf.state().size());
                filteredSRCF.write((char*)srcf.state().data(), sizeof(double) * srcf.state().size());
                t_prev = t;
            }
            metricsCKF << std::fixed << std::setprecision(15) << std::sqrt(lossCKF / (n << 1)) << "\t";
            timingsCKF << timeCKF << "\t";
            metricsSRCF << std::fixed << std::setprecision(15) << std::sqrt(lossSRCF / (n << 1)) << "\t";
            timingsSRCF << timeSRCF << "\t";
        }
        metricsCKF << "\n";
        timingsCKF << "\n";
        metricsSRCF << "\n";
        timingsSRCF << "\n";
    }
    /*for (size_t i = 0; i < n; ++i)
    {
        times.read((char*)&t, sizeof(double));
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
    }*/
    return 0;
}