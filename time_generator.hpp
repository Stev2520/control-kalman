#ifndef CONTROL_KALMAN_TIME_GENERATOR_HPP
#define CONTROL_KALMAN_TIME_GENERATOR_HPP
#pragma once
#include <vector>
#include <random>
#include <fstream>

namespace time_generator {
    enum class TimeMode {
        UNIFORM,      // Равномерный шаг
        VARIABLE,     // Переменный шаг (имитация сбоев/прерываний)
        RANDOM_JITTER // Случайные отклонения от равномерного шага
    };

    class TimeGenerator {
    private:
        std::default_random_engine gen_;
        std::uniform_real_distribution<double> uniform_dist_;
        std::normal_distribution<double> normal_dist_;

    public:
        TimeGenerator(int seed = std::random_device{}())
                : gen_(seed)
                , uniform_dist_(0.001, 0.05)  // Шаг от 1мс до 50мс
                , normal_dist_(0.0, 0.005)    // Случайные отклонения
        {}

        // Генерация временной сетки
        std::vector<double> generate(size_t n_steps,
                                     double dt_base = 0.01,
                                     TimeMode mode = TimeMode::UNIFORM)
        {
            std::vector<double> times;
            times.reserve(n_steps);
            double t = 0.0;
            times.push_back(t);
            for (size_t i = 1; i < n_steps; ++i) {
                double dt = dt_base;
                switch (mode) {
                    case TimeMode::UNIFORM:
                        // Равномерный шаг
                        break;

                    case TimeMode::VARIABLE:
                        // Переменный шаг (имитация сбоев связи)
                        if (i % 100 == 0) {
                            dt = 0.05;  // Задержка каждые 100 шагов
                        } else if (i % 50 == 0) {
                            dt = 0.002; // Быстрый шаг каждые 50 шагов
                        }
                        break;

                    case TimeMode::RANDOM_JITTER:
                        // Случайные отклонения от равномерного шага
                        dt = dt_base * (1.0 + normal_dist_(gen_));
                        break;
                }
                t += dt;
                times.push_back(t);
            }
            return times;
        }

        // Сохранение времен в бинарный файл
        void saveToFile(const std::vector<double>& times,
                        const std::string& filename)
        {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }
            size_t n = times.size();
            file.write((char*)&n, sizeof(size_t));
            for (const auto& t : times) {
                file.write((char*)&t, sizeof(double));
            }
            file.close();
        }

        // Загрузка времен из бинарного файла
        std::vector<double> loadFromFile(const std::string& filename) {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open file: " + filename);
            }

            size_t n;
            file.read((char*)&n, sizeof(size_t));
            std::vector<double> times(n);
            for (size_t i = 0; i < n; ++i) {
                file.read((char*)&times[i], sizeof(double));
            }
            return times;
        }
    };
} // namespace time_generatorv

#endif // CONTROL_KALMAN_TIME_GENERATOR_HPP
