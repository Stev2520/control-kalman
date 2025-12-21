#pragma once
#include <Eigen/Dense>
#include <random>
#include <functional>
namespace kalman_noise
{
    class NoiseGenerator
    {
    private:
        std::default_random_engine generator;
        std::normal_distribution<double> distribution;
        
    public:
        NoiseGenerator(int seed = std::random_device{}())
            : generator(seed), distribution(0.0, 1.0) {}
        
        double gaussian() { return distribution(generator); }
        
        Eigen::VectorXd gaussianVector(int size)
        {
            Eigen::VectorXd res(size);
            for (int i = 0; i < size; ++i) res(i) = distribution(generator);
            return res;
        }
        
        Eigen::VectorXd noiseWithCovariance(const Eigen::MatrixXd& covariance)
        {
            return covariance.llt().matrixL() * gaussianVector(covariance.rows());
        }
    };
    
    static NoiseGenerator noise_gen;
}
namespace model2
{
    const double L_phi = 1, L_p = 1, L_delta = 1, g = 9.80665;
    Eigen::MatrixXd Q(double t = 0)
    {
        static const Eigen::MatrixXd Q_const = []() {
            Eigen::VectorXd q(1);
            q(0) = 4e-4;
            return q;
        }();
        return Q_const;
    }
    Eigen::MatrixXd R(double t = 0)
    {
        static const Eigen::MatrixXd R_const = []() {
            Eigen::MatrixXd r(2, 2);
            r << 1.2e-5, 0.0,
                 0.0, 7.6e-7;
            return r;
        }();
        return R_const;
    }
    Eigen::MatrixXd A(const double dt)
    {
        Eigen::MatrixXd a(2, 2);
        static const double coeff = 20. / 19.;
        const double exp1 = std::exp(-.05 * dt), exp2 = std::exp(-dt);
        a << (exp1 - .05 * exp2) * coeff, (exp1 - exp2) * coeff,
             .05 * (exp2 - exp1) * coeff, (exp2 - .05 * exp1) * coeff;
        return a;
    }
    Eigen::MatrixXd B(const double dt)
    {
        static const double coeff = 160. / 19.;
        const double exp1 = std::exp(-.05 * dt), exp2 = std::exp(-dt);
        Eigen::VectorXd b(2);
        b << (19 - 20 * exp1 + exp2) * coeff, (exp1 - exp2) * coeff;
        return b;
    }
    Eigen::MatrixXd C(const double t)
    {
        static const Eigen::MatrixXd C_const = []() {
            Eigen::MatrixXd c = Eigen::MatrixXd::Zero(2, 2);
            c(0, 1) = 1;
            c(1, 0) = 9.80665;
            return c;
        }();
        return C_const;
    }
    Eigen::MatrixXd D(const double dt)
    {
        static const double coeff = 20. / 19.;
        const double exp1 = std::exp(-.05 * dt), exp2 = std::exp(-dt);
        Eigen::VectorXd d(2);
        d << (19 - 20 * exp1 + exp2) * coeff, (exp1 - exp2) * coeff;
        return d;
    }
    Eigen::VectorXd u(const double t)
    {
        Eigen::VectorXd res(1);
        res(0) = 0.1 * sin(t * 0.5);
        return res;
    }
    Eigen::VectorXd w(const double t) { return kalman_noise::noise_gen.noiseWithCovariance(Q(t)); }
    Eigen::VectorXd v(const double t) { return kalman_noise::noise_gen.noiseWithCovariance(R(t)); }
}