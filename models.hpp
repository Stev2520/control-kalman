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
namespace model0
{
    const double b = 1;
    Eigen::MatrixXd Q(double t = 0.0) 
    {
        static const Eigen::MatrixXd Q_const = []() {
            Eigen::MatrixXd q(2, 2);
            q << 0.1, 0.0,
                 0.0, 0.2;
            return q;
        }();
        return Q_const;  // Always returns the same matrix
    }
    
    Eigen::MatrixXd R(double t = 0.0) 
    {
        static const Eigen::MatrixXd R_const = []() {
            Eigen::MatrixXd r(2, 2);
            r << 1.0, 0.0,
                 0.0, 1.0;
            return r;
        }();
        return R_const;
    }
    Eigen::MatrixXd A(const double dt)
    {
        Eigen::MatrixXd res = Eigen::MatrixXd::Identity(2, 2);
        res(0, 1) = dt;
        return res;
    }
    Eigen::MatrixXd B(const double dt)
    {
        Eigen::VectorXd res(2);
        res(0) = res(1) = b * dt;
        res(0) *= .5 * dt;
        return res;
    }
    Eigen::MatrixXd C(const double t)
    {
        return Eigen::MatrixXd::Identity(2, 2);
    }
    Eigen::MatrixXd D(const double dt)
    {
        Eigen::VectorXd res(2);
        res(0) = res(1) = dt;
        res(0) *= .5 * dt;
        return res;
    }
    Eigen::VectorXd u(const double t)
    {
        Eigen::VectorXd res(1);
        return res;
    }
    Eigen::VectorXd w(const double t)
    {
        Eigen::VectorXd res(1);
        return res;
    }
    Eigen::VectorXd v(const double t)
    {
        Eigen::VectorXd res(1);
        return res;
    }
}
namespace model1    //aeroplane from practice
{
    const double b = 1;
    Eigen::MatrixXd Q(double t = 0.0) 
    {
        static const Eigen::MatrixXd Q_const = []() {
            Eigen::MatrixXd q(2, 2);
            q << 0.05, 0.01,   // Angle and angular velocity process noise
                 0.01, 0.1;    // (with some correlation)
            return q;
        }();
        return Q_const;
    }
    
    Eigen::MatrixXd R(double t = 0.0) 
    {
        static const Eigen::MatrixXd R_const = []() {
            Eigen::MatrixXd r(2, 2);
            r << 0.5, 0.0,
                 0.0, 0.5;
            return r;
        }();
        return R_const;
    }
    Eigen::MatrixXd A(const double dt)
    {
        Eigen::MatrixXd res = Eigen::MatrixXd::Identity(2, 2);
        res(0, 1) = dt;
        return res;
    }
    Eigen::MatrixXd B(const double dt)
    {
        Eigen::VectorXd res(2);
        res << .5 * b * dt * dt, b * dt;
        return res;
    }
    Eigen::MatrixXd C(const double t)
    {
        return Eigen::MatrixXd::Identity(2, 2);
    }
    Eigen::MatrixXd D(const double dt)
    {
        Eigen::VectorXd res(2);
        res << .5 * dt * dt, dt;
        return res;
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

namespace model2
{
    const double L_phi = 1, L_p = 1, L_delta = 1, g = 9.80665;
    Eigen::MatrixXd Q(double t = 0.0) 
    {
        static const Eigen::MatrixXd Q_const = []() {
            Eigen::MatrixXd q(2, 2);
            q << 0.05, 0.01,   // Angle and angular velocity process noise
                 0.01, 0.1;    // (with some correlation)
            return q;
        }();
        return Q_const;
    }
    Eigen::MatrixXd R(double t = 0.0) 
    {
        static const Eigen::MatrixXd R_const = []() {
            Eigen::MatrixXd r(2, 2);
            r << 0.5, 0.0,
                 0.0, 0.5;
            return r;
        }();
        return R_const;
    }
    Eigen::MatrixXd A(const double dt)
    {
        Eigen::MatrixXd res = Eigen::MatrixXd::Identity(2, 2);
        res(0, 1) = dt;
        res(1, 0) = L_phi * dt;
        res(1, 1) += L_p * dt;
        return res;
    }
    Eigen::MatrixXd B(const double dt)
    {
        Eigen::VectorXd res(2);
        res(0) = 0;
        res(1) = 1;
        return res;
    }
    Eigen::MatrixXd C(const double t)
    {
        Eigen::MatrixXd res = Eigen::MatrixXd::Zero(2, 2);
        res(0, 1) = 1;
        res(1, 0) = g;
        return res;
    }
    Eigen::MatrixXd D(const double dt)
    {
        Eigen::VectorXd res = Eigen::VectorXd::Zero(2);
        res(1) = L_delta * dt;
        return res;
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