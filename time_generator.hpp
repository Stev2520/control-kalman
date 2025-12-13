/**
 * @file control_kalman_time_generator.hpp
 * @brief Генератор временных меток для фильтра Калмана
 * @author FAST_DEVELOPMENT (NORREYLL)
 * @date 2025
 * @version 2.0
 *
 * @copyright MIT License
 *
 * Данный заголовочный файл содержит класс TimeGenerator для генерации
 * временных сеток с различными режимами, используемыми в задачах
 * оценки состояния с помощью фильтра Калмана.
 */

#ifndef CONTROL_KALMAN_TIME_GENERATOR_HPP
#define CONTROL_KALMAN_TIME_GENERATOR_HPP

#pragma once

#include <vector>
#include <random>
#include <fstream>

/**
 * @namespace time_generator
 * @brief Пространство имен для генерации временных меток
 *
 * Содержит классы и перечисления для работы с генерацией временных сеток
 * различного типа, используемых в системах управления и фильтрации.
 */
namespace time_generator {
    /**
     * @enum TimeMode
     * @brief Режимы генерации временных меток
     *
     * Определяет различные стратегии генерации временных интервалов
     * для моделирования различных сценариев работы систем реального времени.
     */
    enum class TimeMode {
        UNIFORM,      /**< Равномерный шаг по времени */
        VARIABLE,     /**< Переменный шаг (имитация сбоев/прерываний) */
        RANDOM_JITTER /**< Случайные отклонения от равномерного шага */
    };

    /**
     * @class TimeGenerator
     * @brief Генератор временных сеток
     *
     * Класс для генерации последовательностей временных меток с различными
     * характеристиками. Поддерживает сохранение и загрузку временных сеток
     * в бинарные файлы.
     *
     * @warning При использовании режима RANDOM_JITTER временные интервалы
     * могут стать отрицательными при больших отклонениях.
     */
    class TimeGenerator {
    private:
        std::default_random_engine gen_;                 /**< Генератор случайных чисел */
        std::uniform_real_distribution<double> uniform_dist_; /**< Равномерное распределение */
        std::normal_distribution<double> normal_dist_;   /**< Нормальное распределение */

    public:
        /**
         * @brief Конструктор класса TimeGenerator
         *
         * Инициализирует генератор случайных чисел и распределения.
         *
         * @param seed Начальное значение для генератора случайных чисел
         *             (по умолчанию используется случайное устройство)
         *
         * @note Равномерное распределение настроено на интервал [0.001, 0.05],
         *       что соответствует шагам от 1 мс до 50 мс.
         * @note Нормальное распределение настроено со средним 0 и
         *       стандартным отклонением 0.005.
         */
        explicit TimeGenerator(int seed = std::random_device{}())
                : gen_(seed)
                , uniform_dist_(0.001, 0.05)  // Шаг от 1мс до 50мс
                , normal_dist_(0.0, 0.005)    // Случайные отклонения
        {}

        /**
         * @brief Генерирует временную сетку
         *
         * Создает вектор временных меток в соответствии с выбранным режимом.
         *
         * @param n_steps Количество временных меток для генерации
         * @param dt_base Базовый временной шаг (по умолчанию 0.01 секунды)
         * @param mode Режим генерации (по умолчанию UNIFORM)
         * @return std::vector<double> Вектор временных меток
         *
         * @exception std::invalid_argument Если n_steps равен 0
         *
         * @details Режимы генерации:
         *          - UNIFORM: постоянный шаг dt_base
         *          - VARIABLE: имитация сбоев связи - задержки каждые 100 шагов,
         *                      быстрые шаги каждые 50 шагов
         *          - RANDOM_JITTER: базовый шаг с нормально распределенными отклонениями
         *
         * @code{.cpp}
         * TimeGenerator tg;
         * auto times = tg.generate(1000, 0.01, TimeMode::RANDOM_JITTER);
         * @endcode
         */
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

        /**
         * @brief Сохраняет временные метки в бинарный файл
         *
         * Сохраняет вектор временных меток в бинарный файл для последующего
         * использования или анализа.
         *
         * @param times Вектор временных меток для сохранения
         * @param filename Имя файла для сохранения
         *
         * @exception std::runtime_error Если не удалось открыть файл
         *
         * @note Формат файла:
         *       1. size_t n - количество элементов
         *       2. double t0, t1, ..., tn-1 - временные метки
         *
         * @see loadFromFile
         */
        static void saveToFile(const std::vector<double>& times,
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

        /**
         * @brief Загружает временные метки из бинарного файла
         *
         * Загружает вектор временных меток из бинарного файла,
         * сохраненного методом saveToFile().
         *
         * @param filename Имя файла для загрузки
         * @return std::vector<double> Вектор загруженных временных меток
         *
         * @exception std::runtime_error Если не удалось открыть файл
         * @exception std::runtime_error Если произошла ошибка чтения
         *
         * @note Формат файла должен соответствовать формату saveToFile()
         *
         * @see saveToFile
         */
        static std::vector<double> loadFromFile(const std::string& filename)
        {
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
} // namespace time_generator

#endif // CONTROL_KALMAN_TIME_GENERATOR_HPP
