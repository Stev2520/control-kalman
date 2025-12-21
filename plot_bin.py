import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import os
from matplotlib.patches import Ellipse

class KalmanVisualizer:
    def __init__(self, data_path='./cmake-build-debug/data/verhaegen_test6_easy'):
        self.data_path = data_path
        self.df = None
        self.times = None
        self.metrics_text = ""
        self.comparison_text = ""
        self.load_data()

    def load_data(self):
        """Загрузка данных из CSV файлов или TXT файлов"""
        print(f"Загрузка данных из: {self.data_path}")

        # Проверяем существование папки
        if not os.path.exists(self.data_path):
            print(f"✗ Папка не найдена: {self.data_path}")
            print("Попытка найти альтернативные пути...")

            # Ищем папки с данными
            possible_paths = [
                './data/verhaegen_test6_easy',
                '../data/verhaegen_test6_easy',
                '../../data/verhaegen_test6_easy',
                f'../{self.data_path}',
                f'../../{self.data_path}'
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    self.data_path = path
                    print(f"✓ Найдена альтернативная папка: {path}")
                    break
            else:
                print("✗ Не найдено ни одной папки с данными")
                return

        print(f"✓ Папка найдена: {self.data_path}")
        print("Содержимое папки:")
        for f in os.listdir(self.data_path)[:10]:  # Показываем первые 10 файлов
            print(f"  - {f}")
        if len(os.listdir(self.data_path)) > 10:
            print(f"  ... и ещё {len(os.listdir(self.data_path)) - 10} файлов")

        # Пытаемся загрузить CSV файл
        csv_path = os.path.join(self.data_path, "simulation_data.csv")
        if os.path.exists(csv_path):
            print(f"\n✓ Найден CSV файл: {csv_path}")
            try:
                self.df = pd.read_csv(csv_path)
                print(f"✓ Загружено {len(self.df)} строк")
                print(f"  Колонки: {list(self.df.columns)}")

                # Переименовываем колонки если нужно
                if 'time' in self.df.columns:
                    print("✓ Используются правильные имена колонок")
                elif 'Time(s)' in self.df.columns:
                    self.df = self.df.rename(columns={'Time(s)': 'time'})
                    print("✓ Переименована колонка Time(s) -> time")

                expected_columns = ['meas_gyro_exact', 'meas_accel_exact', 'meas_gyro_noisy', 'meas_accel_noisy']
                available_columns = [col for col in expected_columns if col in self.df.columns]

                if len(available_columns) == len(expected_columns):
                    print("✓ Обнаружены новые имена колонок (гироскоп/акселерометр)")
                elif 'meas_phi_exact' in self.df.columns or 'meas_p_exact' in self.df.columns:
                    print("⚠ Обнаружены старые имена колонок (phi/p)")
                    # Автоматическое переименование старых колонок
                    self.rename_old_columns()

                # Проверяем наличие нужных колонок
                required_cols = ['true_phi', 'true_p', 'ckf_phi', 'ckf_p', 'srcf_phi', 'srcf_p']
                missing_cols = [col for col in required_cols if col not in self.df.columns]
                if missing_cols:
                    print(f"⚠ Отсутствуют колонки: {missing_cols}")

                    # Пытаемся найти альтернативные имена
                    col_mapping = {}
                    for col in missing_cols:
                        # Ищем похожие имена
                        possible_names = [
                            col,
                            col.replace('_', ' '),
                            col.upper(),
                            col.replace('phi', 'Phi').replace('p', 'P')
                        ]
                        for possible in possible_names:
                            if possible in self.df.columns:
                                col_mapping[possible] = col
                                break

                    if col_mapping:
                        self.df = self.df.rename(columns=col_mapping)
                        print(f"✓ Переименованы колонки: {col_mapping}")

            except Exception as e:
                print(f"✗ Ошибка загрузки CSV: {e}")
                self.df = None

        comparison_path = os.path.join(self.data_path, "comparison.csv")
        if os.path.exists(comparison_path):
            print(f"\n✓ Найден файл сравнений: {comparison_path}")
            try:
                self.comparison_df = pd.read_csv(comparison_path)
                print(f"✓ Загружено {len(self.comparison_df)} строк сравнений")
                print(f"  Колонки: {list(self.comparison_df.columns)}")

                # Анализ данных сравнений
                self.analyze_comparison_data()
            except Exception as e:
                print(f"✗ Ошибка загрузки comparison.csv: {e}")

        # Если CSV не найден, пробуем загрузить TXT файлы
        if self.df is None:
            txt_path = os.path.join(self.data_path, "main_data.txt")
            if os.path.exists(txt_path):
                print(f"\n✓ Найден TXT файл: {txt_path}")
                try:
                    self.df = self.load_from_txt(txt_path)
                    print(f"✓ Загружено {len(self.df)} строк из TXT")
                except Exception as e:
                    print(f"✗ Ошибка загрузки TXT: {e}")
                    self.df = None

        self.load_metrics_files()

        # Если данные всё ещё не загружены, создаём тестовые
        if self.df is None:
            print("\n⚠ Данные не найдены, создаю тестовые данные...")
            self.create_test_data()
        else:
            self.analyze_data_structure()



    def load_from_txt(self, filepath):
        """Загрузка данных из текстового файла с фиксированной шириной"""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Пропускаем заголовки
        data_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('=', '-', 'KALMAN', 'Time(s)')):
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        # Парсим числа
                        row_data = [float(p) for p in parts[:8]]
                        data_lines.append(row_data)
                    except ValueError:
                        continue

        # Создаем DataFrame
        df = pd.DataFrame(data_lines, columns=[
            'time', 'true_phi', 'true_p', 'control',
            'ckf_phi', 'ckf_p', 'srcf_phi', 'srcf_p'
        ])

        return df

    def rename_old_columns(self):
        """Переименование старых колонок в новые"""
        rename_map = {}

        # Старые имена -> Новые имена
        if 'meas_phi_exact' in self.df.columns:
            rename_map['meas_phi_exact'] = 'meas_gyro_exact'  # было phi, теперь гироскоп (p)
        if 'meas_p_exact' in self.df.columns:
            rename_map['meas_p_exact'] = 'meas_accel_exact'   # было p, теперь акселерометр (g*sin(phi))
        if 'meas_phi_noisy' in self.df.columns:
            rename_map['meas_phi_noisy'] = 'meas_gyro_noisy'
        if 'meas_p_noisy' in self.df.columns:
            rename_map['meas_p_noisy'] = 'meas_accel_noisy'

        if rename_map:
            self.df = self.df.rename(columns=rename_map)
            print(f"✓ Переименованы колонки: {rename_map}")

    def analyze_data_structure(self):
        """Анализ структуры загруженных данных"""
        print("\n" + "="*50)
        print("АНАЛИЗ СТРУКТУРЫ ДАННЫХ")
        print("="*50)

        g = 9.80665  # ускорение свободного падения

        # Проверяем измерения
        if 'meas_gyro_exact' in self.df.columns and 'true_p' in self.df.columns:
            gyro_diff = (self.df['meas_gyro_exact'] - self.df['true_p']).abs()
            print(f"Гироскоп (meas_gyro_exact):")
            print(f"  Отличается от true_p на: {gyro_diff.mean():.15e} ± {gyro_diff.std():.15e}")
            print(f"  Диапазон: {self.df['meas_gyro_exact'].min():.15f} ... {self.df['meas_gyro_exact'].max():.15f}")

        if 'meas_accel_exact' in self.df.columns and 'true_phi' in self.df.columns:
            true_accel = g * np.sin(self.df['true_phi'])
            accel_diff = (self.df['meas_accel_exact'] - true_accel).abs()
            print(f"\nАкселерометр (meas_accel_exact):")
            print(f"  Отличается от g·sin(φ) на: {accel_diff.mean():.15e} ± {accel_diff.std():.15e}")
            print(f"  Диапазон: {self.df['meas_accel_exact'].min():.15f} ... {self.df['meas_accel_exact'].max():.15f}")

        # Проверяем шум
        if 'meas_gyro_exact' in self.df.columns and 'meas_gyro_noisy' in self.df.columns:
            gyro_noise = (self.df['meas_gyro_noisy'] - self.df['meas_gyro_exact']).abs()
            print(f"\nШум гироскопа (meas_gyro_noisy - meas_gyro_exact):")
            print(f"  Средний: {gyro_noise.mean():.15e}, Макс: {gyro_noise.max():.15e}")

        if 'meas_accel_exact' in self.df.columns and 'meas_accel_noisy' in self.df.columns:
            accel_noise = (self.df['meas_accel_noisy'] - self.df['meas_accel_exact']).abs()
            print(f"Шум акселерометра (meas_accel_noisy - meas_accel_exact):")
            print(f"  Средний: {accel_noise.mean():.15e}, Макс: {accel_noise.max():.15e}")

        print("="*50)

    def analyze_comparison_data(self):
        """Анализ данных сравнений"""
        if self.comparison_df is None:
            return

        print("\n" + "="*50)
        print("АНАЛИЗ ДАННЫХ СРАВНЕНИЙ")
        print("="*50)

        # Основные статистики
        print(f"Средняя ошибка CKF: {self.comparison_df['CKF_Error'].mean():.15e}")
        print(f"Средняя ошибка SRCF: {self.comparison_df['SRCF_Error'].mean():.15e}")
        print(f"Отношение SRCF/CKF: {self.comparison_df['SRCF_Error'].mean() / self.comparison_df['CKF_Error'].mean():.15f}")

        # Нормы ковариаций
        print(f"\nНорма ковариации CKF: {self.comparison_df['CKF_Cov_Norm'].mean():.15e}")
        print(f"Норма ковариации SRCF: {self.comparison_df['SRCF_Cov_Norm'].mean():.15e}")
        print(f"Отношение норм SRCF/CKF: {self.comparison_df['SRCF_Cov_Norm'].mean() / self.comparison_df['CKF_Cov_Norm'].mean():.15f}")

        # Числа обусловленности
        if 'CKF_Cond_Number' in self.comparison_df.columns and 'SRCF_Cond_Number' in self.comparison_df.columns:
            print(f"\nЧисло обусловленности CKF: {self.comparison_df['CKF_Cond_Number'].mean():.15e}")
            print(f"Число обусловленности SRCF: {self.comparison_df['SRCF_Cond_Number'].mean():.15e}")
            print(f"Отношение cond SRCF/CKF: {self.comparison_df['SRCF_Cond_Number'].mean() / self.comparison_df['CKF_Cond_Number'].mean():.15f}")

        print("="*50)


    def load_metrics_files(self):
        """Загрузка файлов с метриками"""
        try:
            metrics_path = os.path.join(self.data_path, "filter_metrics.txt")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics_text = f.read()
                print("✓ Загружены метрики фильтров")

            comparison_path = os.path.join(self.data_path, "comparison_details.txt")
            if os.path.exists(comparison_path):
                with open(comparison_path, 'r') as f:
                    self.comparison_text = f.read()
                print("✓ Загружены данные сравнения")
        except Exception as e:
            print(f"⚠ Ошибка загрузки метрик: {e}")

    def create_test_data(self):
        """Создание тестовых данных если файлы отсутствуют"""
        print("Создание тестовых данных...")
        n = 100
        t = np.linspace(0, 2, n)  # 2 секунды как в ваших данных

        # Более реалистичные данные на основе ваших CSV
        true_phi = 1.0 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        true_p = 2.0 + 0.3 * np.cos(2 * np.pi * 0.3 * t)

        g = 9.80665

        # Точные измерения (гироскоп и акселерометр)
        gyro_exact = true_p  # Гироскоп измеряет угловую скорость p
        accel_exact = g * np.sin(true_phi)  # Акселерометр измеряет g*sin(phi)

        # Зашумленные измерения
        gyro_noisy = gyro_exact + np.random.normal(0, 0.05, n)
        accel_noisy = accel_exact + np.random.normal(0, 0.1, n)

        # Оценки фильтров
        ckf_phi = true_phi + np.random.normal(0, 0.01, n)
        ckf_p = true_p + np.random.normal(0, 0.02, n)

        # SRCF делаем ИДЕНТИЧНЫМ CKF
        srcf_phi = ckf_phi.copy()
        srcf_p = ckf_p.copy()

        self.df = pd.DataFrame({
            'time': t,
            'true_phi': true_phi,
            'true_p': true_p,
            'meas_gyro_exact': gyro_exact,
            'meas_accel_exact': accel_exact,
            'meas_gyro_noisy': gyro_noisy,
            'meas_accel_noisy': accel_noisy,
            'control': np.sin(0.5 * t) * 0.05,  # Пример управления
            'ckf_phi': ckf_phi,
            'ckf_p': ckf_p,
            'srcf_phi': srcf_phi,
            'srcf_p': srcf_p
        })

        print("✓ Созданы тестовые данные с новыми именами колонок")

    def calculate_errors(self):
        """Расчет ошибок фильтров"""
        if self.df is None:
            print("⚠ Нет данных для расчета ошибок")
            return

        # Проверяем наличие нужных колонок
        required_cols = ['true_phi', 'true_p', 'ckf_phi', 'ckf_p', 'srcf_phi', 'srcf_p']
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            print(f"⚠ Отсутствуют колонки для расчета ошибок: {missing_cols}")

            # Если нет измерений, создаем их из истинных значений с шумом
            if 'meas_gyro_noisy' not in self.df.columns and 'true_phi' in self.df.columns:
                self.df['meas_gyro_noisy'] = self.df['true_phi'] + np.random.normal(0, 0.05, len(self.df))
                self.df['meas_accel_noisy'] = self.df['true_p'] + np.random.normal(0, 0.1, len(self.df))
                print("✓ Созданы искусственные измерения")

            # Если нет оценок фильтров, создаем их близкими к истинным
            if 'ckf_phi' not in self.df.columns and 'true_phi' in self.df.columns:
                self.df['ckf_phi'] = self.df['true_phi'] + np.random.normal(0, 0.01, len(self.df))
                self.df['ckf_p'] = self.df['true_p'] + np.random.normal(0, 0.02, len(self.df))
                self.df['srcf_phi'] = self.df['ckf_phi'].copy()  # Идентичные
                self.df['srcf_p'] = self.df['ckf_p'].copy()      # Идентичные
                print("✓ Созданы искусственные оценки фильтров")

        self.df['ckf_error_phi'] = np.abs(self.df['ckf_phi'] - self.df['true_phi'])
        self.df['ckf_error_p'] = np.abs(self.df['ckf_p'] - self.df['true_p'])
        self.df['ckf_total_error'] = np.sqrt(self.df['ckf_error_phi']**2 + self.df['ckf_error_p']**2)

        self.df['srcf_error_phi'] = np.abs(self.df['srcf_phi'] - self.df['true_phi'])
        self.df['srcf_error_p'] = np.abs(self.df['srcf_p'] - self.df['true_p'])
        self.df['srcf_total_error'] = np.sqrt(self.df['srcf_error_phi']**2 + self.df['srcf_error_p']**2)

        self.df['error_difference'] = self.df['srcf_total_error'] - self.df['ckf_total_error']

        print("✓ Рассчитаны ошибки фильтров")

        # Анализ идентичности фильтров
        if 'ckf_phi' in self.df.columns and 'srcf_phi' in self.df.columns:
            are_identical = np.allclose(self.df['ckf_phi'], self.df['srcf_phi'], atol=1e-15)
            print(f"✓ Фильтры идентичны по φ: {are_identical}")

            if are_identical:
                print("⚠ CKF и SRCF дают ИДЕНТИЧНЫЕ результаты")
                print("  Это нормально для хорошо обусловленных задач")

    def plot_state_comparison(self):
        """График 1: Сравнение оценок состояния"""
        if self.df is None:
            print("⚠ Нет данных для построения графика 1")
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 10))

        # 1. Угол phi - ВСЕГДА показываем оба фильтра
        ax = axes[0, 0]

        # Истинные значения
        ax.plot(self.df['time'], self.df['true_phi'], 'k-', linewidth=3, label='Истинное значение', alpha=0.8)

        # Измерения если есть
        if 'meas_accel_exact' in self.df.columns:
            g = 9.80665
            # Преобразуем ускорение обратно в угол для отображения: phi ≈ arcsin(accel/g)
            with np.errstate(invalid='ignore'):
                accel_as_phi = np.arcsin(self.df['meas_accel_exact'] / g)
            ax.plot(self.df['time'], accel_as_phi, 'r.', alpha=0.3,
                    markersize=2, label='Акселерометр (преобразован)')


        # Оценки фильтров - ВСЕГДА показываем оба
        ax.plot(self.df['time'], self.df['ckf_phi'], 'b-', linewidth=1.5, label='CKF', alpha=0.7)
        ax.plot(self.df['time'], self.df['srcf_phi'], 'g--', linewidth=1.5, label='SRCF', alpha=0.7)

        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Угол φ', fontsize=11)
        ax.set_title('Сравнение оценок угла φ', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 2. Угловая скорость p
        ax = axes[0, 1]
        ax.plot(self.df['time'], self.df['true_p'], 'k-', linewidth=3, label='Истинное значение', alpha=0.8)

        if 'meas_gyro_exact' in self.df.columns:
            ax.plot(self.df['time'], self.df['meas_gyro_exact'], 'r.', alpha=0.3,
                    markersize=2, label='Гироскоп')

        ax.plot(self.df['time'], self.df['ckf_p'], 'b-', linewidth=1.5, label='CKF', alpha=0.7)
        ax.plot(self.df['time'], self.df['srcf_p'], 'g--', linewidth=1.5, label='SRCF', alpha=0.7)

        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Скорость p', fontsize=11)
        ax.set_title('Сравнение оценок угловой скорости p', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 3. Измерения гироскопа
        ax = axes[1, 0]
        if 'meas_gyro_exact' in self.df.columns:
            ax.plot(self.df['time'], self.df['meas_gyro_exact'], 'b-', alpha=0.7,
                    label='Точное измерение', linewidth=1)
        if 'meas_gyro_noisy' in self.df.columns:
            ax.plot(self.df['time'], self.df['meas_gyro_noisy'], 'r.', alpha=0.3,
                    markersize=2, label='Зашумленное')
        if 'true_p' in self.df.columns:
            ax.plot(self.df['time'], self.df['true_p'], 'k--', alpha=0.8,
                    label='Истинное p', linewidth=1.5)

        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Гироскоп (рад/с)', fontsize=11)
        ax.set_title('Измерения гироскопа (угловая скорость)', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 4. Измерения акселерометра
        ax = axes[1, 1]
        if 'meas_accel_exact' in self.df.columns:
            ax.plot(self.df['time'], self.df['meas_accel_exact'], 'b-', alpha=0.7,
                    label='Точное измерение', linewidth=1)
        if 'meas_accel_noisy' in self.df.columns:
            ax.plot(self.df['time'], self.df['meas_accel_noisy'], 'r.', alpha=0.3,
                    markersize=2, label='Зашумленное')

        # Истинное ускорение
        if 'true_phi' in self.df.columns:
            g = 9.80665
            true_accel = g * np.sin(self.df['true_phi'])
            ax.plot(self.df['time'], true_accel, 'k--', alpha=0.8,
                    label='Истинное g·sin(φ)', linewidth=1.5)

        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Акселерометр (м/с²)', fontsize=11)
        ax.set_title('Измерения акселерометра (ускорение)', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 5. Ошибки по phi
        ax = axes[2, 0]
        if 'ckf_error_phi' in self.df.columns:
            ax.plot(self.df['time'], self.df['ckf_error_phi'], 'b-', alpha=0.7,
                    linewidth=1.5, label='CKF ошибка')
        if 'srcf_error_phi' in self.df.columns:
            ax.plot(self.df['time'], self.df['srcf_error_phi'], 'g--', alpha=0.7,
                    linewidth=1.5, label='SRCF ошибка')

        if 'ckf_error_phi' in self.df.columns:
            ax.fill_between(self.df['time'], 0, self.df['ckf_error_phi'],
                            alpha=0.2, color='blue')
        if 'srcf_error_phi' in self.df.columns:
            ax.fill_between(self.df['time'], 0, self.df['srcf_error_phi'],
                            alpha=0.2, color='green')

        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Абсолютная ошибка φ (рад)', fontsize=11)
        ax.set_title('Ошибки оценки угла', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 6. Ошибки по p
        ax = axes[2, 1]
        if 'ckf_error_p' in self.df.columns:
            ax.plot(self.df['time'], self.df['ckf_error_p'], 'b-', alpha=0.7,
                    linewidth=1.5, label='CKF ошибка')
        if 'srcf_error_p' in self.df.columns:
            ax.plot(self.df['time'], self.df['srcf_error_p'], 'g--', alpha=0.7,
                    linewidth=1.5, label='SRCF ошибка')

        if 'ckf_error_p' in self.df.columns:
            ax.fill_between(self.df['time'], 0, self.df['ckf_error_p'],
                            alpha=0.2, color='blue')
        if 'srcf_error_p' in self.df.columns:
            ax.fill_between(self.df['time'], 0, self.df['srcf_error_p'],
                            alpha=0.2, color='green')


        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Абсолютная ошибка p (рад/с)', fontsize=11)
        ax.set_title('Ошибки оценки скорости', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Сравнение фильтров Калмана: CKF vs SRCF', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = os.path.join(self.data_path, 'state_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ График 1 сохранен: {output_path}")
        plt.show()

    def plot_kalman_x_like_photo(self, filter_name='ckf'):
        """
        Строит график в точности как на фото:
        - Чёрная: истинное значение угла
        - Красные точки: зашумлённые измерения (акселерометр → угол)
        - Зелёная линия: оценка одного фильтра (CKF или SRCF)
        """
        if self.df is None:
            print("⚠ Нет данных")
            return

        # Выбор фильтра
        if filter_name == 'ckf':
            est_phi = self.df['ckf_phi']
            label = 'Результаты фильтра'
        elif filter_name == 'srcf':
            est_phi = self.df['srcf_phi']
            label = 'Результаты фильтра (SRCF)'
        else:
            raise ValueError("filter_name must be 'ckf' or 'srcf'")

        mse = np.mean((est_phi - self.df['true_phi'])**2)

        fig, ax = plt.subplots(figsize=(10, 6))

        # 1. Истина — чёрная сплошная
        ax.plot(self.df['time'], self.df['true_phi'], 'k-', linewidth=2, label='Истинная траектория')

        # 2. Зашумлённые измерения — красные точки
        if 'meas_accel_noisy' in self.df.columns:
            g = 9.80665
            with np.errstate(invalid='ignore', divide='ignore'):
                meas_phi_noisy = np.arcsin(self.df['meas_accel_noisy'] / g)
            # Фильтруем некорректные значения (|accel/g| > 1)
            valid = np.abs(self.df['meas_accel_noisy'] / g) <= 20
            ax.plot(self.df['time'][valid], meas_phi_noisy[valid], 'r.', markersize=3, alpha=0.6, label='Измерения')

        # 3. Оценка фильтра — зелёная линия
        ax.plot(self.df['time'], est_phi, 'g-', linewidth=2, label=label)

        # Оформление
        ax.set_xlabel('Шаг времени', fontsize=12)
        ax.set_ylabel('Положение', fontsize=12)
        ax.set_title(f'Сравнение измерений, истинной траектории и результатов фильтра Калмана\n'
                     f'MSE для положения: {mse:.6f} | Количество шагов: {len(self.df)}', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Сохранение
        output_path = os.path.join(self.data_path, f'kalman_x_like_photo_{filter_name}.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✓ Сохранён график как на фото: {output_path}")
        plt.show()

    def plot_error_analysis(self):
        """График 2: Детальный анализ ошибок"""
        if self.df is None:
            print("⚠ Нет данных для построения графика 2")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Кумулятивные ошибки
        ax = axes[0, 0]
        ckf_cumulative = self.df['ckf_total_error'].cumsum()
        srcf_cumulative = self.df['srcf_total_error'].cumsum()

        ax.plot(self.df['time'], ckf_cumulative, 'b-', linewidth=2, label='CKF кумулятивная')
        ax.plot(self.df['time'], srcf_cumulative, 'g--', linewidth=2, label='SRCF кумулятивная')

        # Определяем, где какой фильтр лучше
        if not np.allclose(ckf_cumulative, srcf_cumulative, atol=1e-15):
            ax.fill_between(self.df['time'], ckf_cumulative, srcf_cumulative,
                            where=(srcf_cumulative < ckf_cumulative),
                            alpha=0.3, color='green', label='SRCF лучше')
            ax.fill_between(self.df['time'], ckf_cumulative, srcf_cumulative,
                            where=(srcf_cumulative >= ckf_cumulative),
                            alpha=0.3, color='blue', label='CKF лучше')
        else:
            ax.fill_between(self.df['time'], 0, ckf_cumulative, alpha=0.3, color='gray', label='Одинаково')

        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Накопленная ошибка', fontsize=11)
        ax.set_title('Кумулятивное сравнение ошибок', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 2. Разность ошибок
        ax = axes[0, 1]
        diff = self.df['error_difference']

        ax.plot(self.df['time'], diff, 'purple', linewidth=1.5)

        # Заливка в зависимости от знака разности
        if not np.allclose(diff, 0, atol=1e-15):
            ax.fill_between(self.df['time'], 0, diff, where=(diff < 0),
                            alpha=0.3, color='green', label='SRCF лучше')
            ax.fill_between(self.df['time'], 0, diff, where=(diff >= 0),
                            alpha=0.3, color='blue', label='CKF лучше')
        else:
            ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.5, label='Идентичны')

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Разность ошибок (SRCF - CKF)', fontsize=11)
        ax.set_title('Динамика превосходства фильтров', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 3. Гистограмма разностей
        ax = axes[1, 0]

        if not np.allclose(diff, 0, atol=1e-15):
            ax.hist(diff, bins=30, color='purple', alpha=0.7, edgecolor='black')
        else:
            # Если разности нулевые, показываем одну полосу
            ax.bar([0], [len(diff)], width=0.1, color='gray', alpha=0.7, edgecolor='black')

        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Нулевая разность')
        ax.axvline(x=diff.mean(), color='green', linestyle='-', linewidth=2,
                   label=f'Среднее: {diff.mean():.15e}')

        ax.set_xlabel('Разность ошибок', fontsize=11)
        ax.set_ylabel('Частота', fontsize=11)
        ax.set_title('Распределение разностей ошибок', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 4. Скользящее среднее ошибок
        ax = axes[1, 1]
        window = min(20, len(self.df) // 10)  # Адаптивный размер окна

        ckf_ma = self.df['ckf_total_error'].rolling(window=window).mean()
        srcf_ma = self.df['srcf_total_error'].rolling(window=window).mean()

        ax.plot(self.df['time'], ckf_ma, 'b-', linewidth=2, label=f'CKF ({window}-шаговое ср.)')
        ax.plot(self.df['time'], srcf_ma, 'g--', linewidth=2, label=f'SRCF ({window}-шаговое ср.)')

        if not np.allclose(ckf_ma, srcf_ma, atol=1e-15):
            ax.fill_between(self.df['time'], ckf_ma, srcf_ma,
                            where=(srcf_ma < ckf_ma), alpha=0.3, color='green')
        else:
            ax.fill_between(self.df['time'], 0, ckf_ma, alpha=0.3, color='gray', label='Одинаково')

        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Скользящее среднее ошибки', fontsize=11)
        ax.set_title('Тренды ошибок (скользящее среднее)', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Анализ ошибок фильтров Калмана', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = os.path.join(self.data_path, 'error_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ График 2 сохранен: {output_path}")
        plt.show()

    def plot_statistical_comparison(self):
        """График 3: Статистическое сравнение"""
        if self.df is None:
            print("⚠ Нет данных для построения графика 3")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Боксплот ошибок - ВСЕГДА для обоих фильтров
        ax = axes[0]
        error_data = [self.df['ckf_total_error'], self.df['srcf_total_error']]

        bp = ax.boxplot(error_data, patch_artist=True,
                        labels=['CKF', 'SRCF'],
                        medianprops=dict(color='yellow', linewidth=2),
                        whiskerprops=dict(color='black', linewidth=1),
                        capprops=dict(color='black', linewidth=1),
                        flierprops=dict(marker='o', color='red', alpha=0.5))

        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Общая ошибка', fontsize=11)
        ax.set_title('Статистическое распределение ошибок', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 2. Квантиль-квантиль график
        ax = axes[1]
        if len(self.df) > 10:
            sorted_ckf = np.sort(self.df['ckf_total_error'])
            sorted_srcf = np.sort(self.df['srcf_total_error'])

            # Проверяем, не идентичны ли данные
            if not np.allclose(sorted_ckf, sorted_srcf, atol=1e-15):
                ax.plot(sorted_ckf, sorted_srcf, 'bo', alpha=0.5, markersize=4)
            else:
                ax.plot(sorted_ckf, sorted_srcf, 'ko', alpha=0.5, markersize=4, label='Идентичны')

            min_val = min(sorted_ckf.min(), sorted_srcf.min())
            max_val = max(sorted_ckf.max(), sorted_srcf.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Линия равенства')

            ax.set_xlabel('CKF ошибки (отсортированные)', fontsize=11)
            ax.set_ylabel('SRCF ошибки (отсортированные)', fontsize=11)
            ax.set_title('Q-Q график сравнения распределений', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Недостаточно данных\nдля Q-Q графика',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Q-Q график', fontsize=12, fontweight='bold')

        # 3. Процент превосходства по времени
        ax = axes[2]

        better_srcf = (self.df['error_difference'] < 0).sum() / len(self.df) * 100
        better_ckf = (self.df['error_difference'] > 0).sum() / len(self.df) * 100
        equal = (np.abs(self.df['error_difference']) == 0).sum() / len(self.df) * 100

        labels = ['SRCF лучше', 'CKF лучше', 'Равны']
        sizes = [better_srcf, better_ckf, equal]
        colors = ['lightgreen', 'lightblue', 'lightgray']

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)

        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')

        ax.set_title('Процентное соотношение превосходства', fontsize=12, fontweight='bold')

        # Добавляем статистику в центр
        center_text = f"Всего шагов: {len(self.df)}\n"
        center_text += f"SRCF лучше: {better_srcf:.3f}%\n"
        center_text += f"CKF лучше: {better_ckf:.3f}%\n"
        center_text += f"Средняя разность: {self.df['error_difference'].mean():.15e}"

        ax.text(0, 0, center_text, ha='center', va='center',
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.suptitle('Статистический анализ фильтров', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()

        output_path = os.path.join(self.data_path, 'statistical_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ График 3 сохранен: {output_path}")
        plt.show()

    def plot_phase_space(self):
        """График 4: Фазовое пространство"""
        if self.df is None:
            print("⚠ Нет данных для построения графика 4")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 1. Фазовый портрет истинного состояния
        ax = axes[0]

        if 'true_phi' in self.df.columns and 'true_p' in self.df.columns:
            sc1 = ax.scatter(self.df['true_phi'], self.df['true_p'],
                             c=self.df['time'], cmap='viridis', s=20, alpha=0.7)

            ax.set_xlabel('Угол φ', fontsize=11)
            ax.set_ylabel('Скорость p', fontsize=11)
            ax.set_title('Фазовый портрет истинного состояния', fontsize=12, fontweight='bold')
            plt.colorbar(sc1, ax=ax, label='Время (с)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Нет данных для фазового портрета',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Фазовый портрет', fontsize=12, fontweight='bold')

        # 2. Фазовый портрет ошибок
        ax = axes[1]

        required_error_cols = ['ckf_error_phi', 'ckf_error_p', 'srcf_error_phi', 'srcf_error_p']
        if all(col in self.df.columns for col in required_error_cols):
            ckf_dist = np.sqrt(self.df['ckf_error_phi']**2 + self.df['ckf_error_p']**2)
            srcf_dist = np.sqrt(self.df['srcf_error_phi']**2 + self.df['srcf_error_p']**2)

            # Всегда показываем оба фильтра
            sc2 = ax.scatter(self.df['ckf_error_phi'], self.df['ckf_error_p'],
                             c=ckf_dist, cmap='Blues', s=30, alpha=0.6, label='CKF ошибки')
            sc3 = ax.scatter(self.df['srcf_error_phi'], self.df['srcf_error_p'],
                             c=srcf_dist, cmap='Greens', s=30, alpha=0.6, label='SRCF ошибки')

            ax.set_xlabel('Ошибка угла φ', fontsize=11)
            ax.set_ylabel('Ошибка скорости p', fontsize=11)
            ax.set_title('Фазовое пространство ошибок', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            plt.colorbar(sc2, ax=ax, label='Величина ошибки')
            ax.grid(True, alpha=0.3)

            # Добавляем эллипсы доверительных областей
            if len(self.df) > 50:
                try:
                    # Для CKF
                    ckf_cov = np.cov(self.df['ckf_error_phi'], self.df['ckf_error_p'])
                    lambda_, v = np.linalg.eig(ckf_cov)
                    lambda_ = np.sqrt(lambda_)
                    ell = Ellipse(xy=(self.df['ckf_error_phi'].mean(), self.df['ckf_error_p'].mean()),
                                  width=lambda_[0]*2*2, height=lambda_[1]*2*2,
                                  angle=np.degrees(np.arctan2(v[1,0], v[0,0])),
                                  edgecolor='blue', facecolor='none', linewidth=2, linestyle='--',
                                  label='CKF 95% доверительная область')
                    ax.add_patch(ell)

                    # Для SRCF
                    srcf_cov = np.cov(self.df['srcf_error_phi'], self.df['srcf_error_p'])
                    lambda_, v = np.linalg.eig(srcf_cov)
                    lambda_ = np.sqrt(lambda_)
                    ell = Ellipse(xy=(self.df['srcf_error_phi'].mean(), self.df['srcf_error_p'].mean()),
                                  width=lambda_[0]*2*2, height=lambda_[1]*2*2,
                                  angle=np.degrees(np.arctan2(v[1,0], v[0,0])),
                                  edgecolor='green', facecolor='none', linewidth=2, linestyle='--',
                                  label='SRCF 95% доверительная область')
                    ax.add_patch(ell)
                except:
                    pass
        else:
            ax.text(0.5, 0.5, 'Нет данных для анализа ошибок',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Фазовое пространство ошибок', fontsize=12, fontweight='bold')

        plt.suptitle('Фазовые портреты состояний и ошибок', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = os.path.join(self.data_path, 'phase_space.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ График 4 сохранен: {output_path}")
        plt.show()

    def plot_summary_metrics(self):
        """График 5: Сводные метрики"""
        if self.df is None:
            print("⚠ Нет данных для построения графика 5")
            return

        # Рассчитываем дополнительные метрики для новых измерений
        additional_metrics = {}
        g = 9.80665

        if 'meas_gyro_exact' in self.df.columns and 'true_p' in self.df.columns:
            gyro_error = np.abs(self.df['meas_gyro_exact'] - self.df['true_p']).mean()
            additional_metrics['Ошибка гироскопа'] = gyro_error

        if 'meas_accel_exact' in self.df.columns and 'true_phi' in self.df.columns:
            true_accel = g * np.sin(self.df['true_phi'])
            accel_error = np.abs(self.df['meas_accel_exact'] - true_accel).mean()
            additional_metrics['Ошибка акселерометра'] = accel_error

        # Основные метрики фильтров
        metrics = {
            'CKF': {
                'Средняя ошибка': self.df['ckf_total_error'].mean(),
                'Макс ошибка': self.df['ckf_total_error'].max(),
                'RMS': np.sqrt((self.df['ckf_total_error']**2).mean()),
                'Станд. отклонение': self.df['ckf_total_error'].std(),
                'Медиана': self.df['ckf_total_error'].median()
            },
            'SRCF': {
                'Средняя ошибка': self.df['srcf_total_error'].mean(),
                'Макс ошибка': self.df['srcf_total_error'].max(),
                'RMS': np.sqrt((self.df['srcf_total_error']**2).mean()),
                'Станд. отклонение': self.df['srcf_total_error'].std(),
                'Медиана': self.df['srcf_total_error'].median()
            }
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # График 1: Метрики фильтров
        labels = list(metrics['CKF'].keys())
        ckf_values = list(metrics['CKF'].values())
        srcf_values = list(metrics['SRCF'].values())

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax1.bar(x - width/2, ckf_values, width, label='CKF',
                        color='lightblue', edgecolor='blue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, srcf_values, width, label='SRCF',
                        color='lightgreen', edgecolor='green', alpha=0.8)

        for bars, color in zip([bars1, bars2], ['blue', 'green']):
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3e}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

        ax1.set_xlabel('Метрики', fontsize=11)
        ax1.set_ylabel('Значение', fontsize=11)
        ax1.set_title('Метрики производительности фильтров', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, axis='y')

        # График 2: Дополнительные метрики измерений
        if additional_metrics:
            ax2.bar(range(len(additional_metrics)), list(additional_metrics.values()),
                    color=['orange', 'purple'], alpha=0.7)
            ax2.set_xticks(range(len(additional_metrics)))
            ax2.set_xticklabels(list(additional_metrics.keys()), rotation=45, ha='right')
            ax2.set_ylabel('Средняя ошибка', fontsize=11)
            ax2.set_title('Метрики измерений', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            # Добавляем значения на столбцы
            for i, value in enumerate(additional_metrics.values()):
                ax2.text(i, value, f'{value:.15e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Нет данных об измерениях', ha='center', va='center',
                     fontsize=12, transform=ax2.transAxes)

        plt.suptitle('Сводные метрики симуляции', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(self.data_path, 'summary_metrics.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ График 5 сохранен: {output_path}")
        plt.show()

        # Печатаем метрики
        print("\n" + "="*60)
        print("СВОДНЫЕ МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("="*60)

        for i, metric in enumerate(labels):
            ckf_val = ckf_values[i]
            srcf_val = srcf_values[i]
            ratio = srcf_val / ckf_val if ckf_val != 0 else float('inf')

            print(f"\n{metric}:")
            print(f"  CKF:  {ckf_val:.15e}")
            print(f"  SRCF: {srcf_val:.15e}")

            # if not np.isclose(ratio, 1.0, atol=1e-17):
            better = "SRCF" if ratio < 1 else "CKF"
            improvement = abs(1 - ratio) * 100
            print(f"  Отношение (SRCF/CKF): {ratio:.15f}")
            print(f"  {better} лучше на {improvement:.15f}%")
            # else:
            #     print(f"  Отношение (SRCF/CKF): 1.000000 (идентичны)")

        if additional_metrics:
            print(f"\nМетрики измерений:")
            for name, value in additional_metrics.items():
                print(f"  {name}: {value:.15e}")

    def plot_covariance_analysis(self):
        """График 6: Анализ ковариационных матриц"""
        if self.comparison_df is None:
            print("⚠ Нет данных сравнений для анализа ковариаций")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Нормы ковариаций
        ax = axes[0, 0]
        ax.plot(self.comparison_df['Time'], self.comparison_df['CKF_Cov_Norm'],
                'b-', linewidth=1.5, label='CKF', alpha=0.7)
        ax.plot(self.comparison_df['Time'], self.comparison_df['SRCF_Cov_Norm'],
                'g--', linewidth=1.5, label='SRCF', alpha=0.7)

        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Норма ковариации', fontsize=11)
        ax.set_title('Эволюция норм ковариационных матриц', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 2. Отношение норм
        ax = axes[0, 1]
        cov_ratio = self.comparison_df['SRCF_Cov_Norm'] / self.comparison_df['CKF_Cov_Norm']
        ax.plot(self.comparison_df['Time'], cov_ratio, 'purple', linewidth=1.5)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Равенство')

        # Среднее отношение
        mean_ratio = cov_ratio.mean()
        ax.axhline(y=mean_ratio, color='orange', linestyle='-', alpha=0.7,
                   label=f'Среднее: {mean_ratio:.3f}')

        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Отношение норм (SRCF/CKF)', fontsize=11)
        ax.set_title('Отношение норм ковариационных матриц', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 3. Числа обусловленности (если есть)
        if 'CKF_Cond_Number' in self.comparison_df.columns:
            ax = axes[1, 0]
            ax.plot(self.comparison_df['Time'], self.comparison_df['CKF_Cond_Number'],
                    'b-', linewidth=1.5, label='CKF', alpha=0.7)
            ax.plot(self.comparison_df['Time'], self.comparison_df['SRCF_Cond_Number'],
                    'g--', linewidth=1.5, label='SRCF', alpha=0.7)

            ax.set_yscale('log')  # Логарифмическая шкала
            ax.set_xlabel('Время (с)', fontsize=11)
            ax.set_ylabel('Число обусловленности (log)', fontsize=11)
            ax.set_title('Числа обусловленности ковариационных матриц', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        # 4. Соотношение ошибок и норм ковариаций
        ax = axes[1, 1]
        error_ratio = self.comparison_df['SRCF_Error'] / self.comparison_df['CKF_Error']
        cov_ratio = self.comparison_df['SRCF_Cov_Norm'] / self.comparison_df['CKF_Cov_Norm']

        sc = ax.scatter(cov_ratio, error_ratio,
                        c=self.comparison_df['Time'],
                        cmap='viridis', s=30, alpha=0.7)

        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

        # Линия наилучшего соответствия
        if len(cov_ratio) > 2:
            try:
                mask = (cov_ratio > 0) & (error_ratio > 0)
                if mask.sum() > 2:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        cov_ratio[mask], error_ratio[mask]
                    )
                    x_fit = np.linspace(cov_ratio.min(), cov_ratio.max(), 100)
                    y_fit = slope * x_fit + intercept
                    ax.plot(x_fit, y_fit, 'r-', linewidth=2,
                            label=f'Линейная аппроксимация\nНаклон: {slope:.3f}, R²: {r_value**2:.3f}')
            except:
                pass

        ax.set_xlabel('Отношение норм ковариаций (SRCF/CKF)', fontsize=11)
        ax.set_ylabel('Отношение ошибок (SRCF/CKF)', fontsize=11)
        ax.set_title('Корреляция между нормами ковариаций и ошибками', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.colorbar(sc, ax=ax, label='Время (с)')

        plt.suptitle('Анализ ковариационных матриц фильтров Калмана', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = os.path.join(self.data_path, 'covariance_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ График 6 (ковариации) сохранен: {output_path}")
        plt.show()

    def plot_numerical_stability(self):
        """График 7: Анализ численной устойчивости в стиле Verhaegen & Van Dooren"""
        if self.comparison_df is None:
            print("⚠ Нет данных сравнений для анализа численной устойчивости")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Динамика ошибок фильтров
        ax = axes[0, 0]
        ax.plot(self.comparison_df['Time'], self.comparison_df['CKF_Error'],
                'b-', linewidth=1.5, label='CKF', alpha=0.7)
        ax.plot(self.comparison_df['Time'], self.comparison_df['SRCF_Error'],
                'g--', linewidth=1.5, label='SRCF', alpha=0.7)

        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Ошибка оценки', fontsize=11)
        ax.set_title('Динамика ошибок фильтров (Verhaegen & Van Dooren)', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 2. Логарифм ошибок
        ax = axes[0, 1]
        ckf_log_error = np.log10(np.maximum(self.comparison_df['CKF_Error'], 1e-15))
        srcf_log_error = np.log10(np.maximum(self.comparison_df['SRCF_Error'], 1e-15))

        ax.plot(self.comparison_df['Time'], ckf_log_error,
                'b-', linewidth=1.5, label='CKF (log10)', alpha=0.7)
        ax.plot(self.comparison_df['Time'], srcf_log_error,
                'g--', linewidth=1.5, label='SRCF (log10)', alpha=0.7)

        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('log₁₀(Ошибка)', fontsize=11)
        ax.set_title('Логарифмическая шкала ошибок', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 3. Разность ошибок с накоплением
        ax = axes[1, 0]
        error_diff = self.comparison_df['SRCF_Error'] - self.comparison_df['CKF_Error']
        cumulative_diff = error_diff.cumsum()

        ax.plot(self.comparison_df['Time'], error_diff, 'purple', linewidth=1.5,
                label='Мгновенная разность', alpha=0.7)
        ax.plot(self.comparison_df['Time'], cumulative_diff, 'orange', linewidth=2,
                label='Накопленная разность', alpha=0.7)

        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.fill_between(self.comparison_df['Time'], 0, error_diff,
                        where=(error_diff > 0), alpha=0.3, color='red', label='CKF лучше')
        ax.fill_between(self.comparison_df['Time'], 0, error_diff,
                        where=(error_diff <= 0), alpha=0.3, color='green', label='SRCF лучше')

        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Разность ошибок (SRCF - CKF)', fontsize=11)
        ax.set_title('Накопление разности ошибок', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 4. Отношение ошибок во времени
        ax = axes[1, 1]
        error_ratio = self.comparison_df['SRCF_Error'] / self.comparison_df['CKF_Error']

        # Скользящее среднее
        window = min(20, len(error_ratio) // 10)
        rolling_ratio = error_ratio.rolling(window=window).mean()

        ax.plot(self.comparison_df['Time'], error_ratio, 'gray', linewidth=1, alpha=0.3, label='Мгновенное')
        ax.plot(self.comparison_df['Time'], rolling_ratio, 'blue', linewidth=2,
                label=f'{window}-шаговое ср.')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Равенство')

        # Среднее значение
        mean_ratio = error_ratio.mean()
        ax.axhline(y=mean_ratio, color='orange', linestyle='-', linewidth=2,
                   label=f'Среднее: {mean_ratio:.3f}')

        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Отношение ошибок (SRCF/CKF)', fontsize=11)
        ax.set_title('Отношение ошибок фильтров во времени', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Анализ численной устойчивости в стиле Verhaegen & Van Dooren',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = os.path.join(self.data_path, 'numerical_stability.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ График 7 (численная устойчивость) сохранен: {output_path}")
        plt.show()

    def plot_innovation_analysis(self):
        """График 8: Анализ инноваций"""
        if self.comparison_df is None or 'Innovation' not in self.comparison_df.columns:
            print("⚠ Нет данных об инновациях")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Норма инноваций
        ax = axes[0, 0]
        ax.plot(self.comparison_df['Time'], self.comparison_df['Innovation'],
                'b-', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Норма инноваций', fontsize=11)
        ax.set_title('Эволюция нормы инноваций', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 2. Автокорреляция инноваций
        ax = axes[0, 1]
        innovation = self.comparison_df['Innovation'].values
        n_lags = min(50, len(innovation) // 4)

        if len(innovation) > n_lags * 2:
            autocorr = np.correlate(innovation - innovation.mean(),
                                    innovation - innovation.mean(), mode='full')
            autocorr = autocorr[len(autocorr)//2:len(autocorr)//2 + n_lags]
            autocorr = autocorr / autocorr[0]

            lags = np.arange(n_lags)
            ax.stem(lags, autocorr, basefmt=" ", linefmt='b-', markerfmt='bo')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Доверительные интервалы для белого шума
            conf_int = 1.96 / np.sqrt(len(innovation))
            ax.axhline(y=conf_int, color='red', linestyle='--', alpha=0.5, label='95% доверит.')
            ax.axhline(y=-conf_int, color='red', linestyle='--', alpha=0.5)

            ax.set_xlabel('Лаг', fontsize=11)
            ax.set_ylabel('Автокорреляция', fontsize=11)
            ax.set_title('Автокорреляция инноваций (белый шум?)', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Недостаточно данных\nдля автокорреляции',
                    ha='center', va='center', fontsize=12)
            ax.set_title('Автокорреляция инноваций', fontsize=12, fontweight='bold')

        # 3. Гистограмма инноваций
        ax = axes[1, 0]
        ax.hist(self.comparison_df['Innovation'], bins=30,
                color='blue', alpha=0.7, edgecolor='black', density=True)

        # Теоретическое нормальное распределение
        mu = self.comparison_df['Innovation'].mean()
        sigma = self.comparison_df['Innovation'].std()
        if sigma > 0:
            x = np.linspace(self.comparison_df['Innovation'].min(),
                            self.comparison_df['Innovation'].max(), 100)
            pdf = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, pdf, 'r-', linewidth=2, label=f'N(μ={mu:.3f}, σ={sigma:.3f})')

        ax.set_xlabel('Норма инноваций', fontsize=11)
        ax.set_ylabel('Плотность вероятности', fontsize=11)
        ax.set_title('Распределение нормы инноваций', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 4. Связь инноваций с ошибками
        ax = axes[1, 1]
        sc = ax.scatter(self.comparison_df['Innovation'], self.comparison_df['CKF_Error'],
                        c=self.comparison_df['Time'], cmap='viridis', s=30, alpha=0.7, label='CKF')
        sc = ax.scatter(self.comparison_df['Innovation'], self.comparison_df['SRCF_Error'],
                        c=self.comparison_df['Time'], cmap='plasma', s=30, alpha=0.7, label='SRCF')

        ax.set_xlabel('Норма инноваций', fontsize=11)
        ax.set_ylabel('Ошибка фильтра', fontsize=11)
        ax.set_title('Связь инноваций с ошибками фильтрации', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Анализ инноваций фильтров Калмана', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = os.path.join(self.data_path, 'innovation_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ График 8 (инновации) сохранен: {output_path}")
        plt.show()

    def print_detailed_analysis(self):
        """Детальный анализ данных"""
        if self.df is None:
            return

        print("\n" + "="*60)
        print("ДЕТАЛЬНЫЙ АНАЛИЗ ДАННЫХ")
        print("="*60)

        # Проверка идентичности фильтров
        if 'ckf_phi' in self.df.columns and 'srcf_phi' in self.df.columns:
            max_phi_diff = np.max(np.abs(self.df['ckf_phi'] - self.df['srcf_phi']))
            max_p_diff = np.max(np.abs(self.df['ckf_p'] - self.df['srcf_p']))

            print(f"Максимальное расхождение по φ: {max_phi_diff:.15e}")
            print(f"Максимальное расхождение по p: {max_p_diff:.15e}")

            are_identical = max_phi_diff < 1e-15 and max_p_diff < 1e-15

            if are_identical:
                print("\n✓ CKF и SRCF дают ИДЕНТИЧНЫЕ результаты")
                print("\nВозможные причины:")
                print("1. Хорошо обусловленные матрицы Q и R")
                print("2. Малая начальная неопределенность (P0)")
                print("3. Умеренный уровень шума")
                print("4. Простая линейная модель")
                print("5. Высокая точность вычислений (double)")

                print("\nЭто НОРМАЛЬНО для многих практических задач!")
                print("SRCF проявляет преимущества при:")
                print("- Плохой обусловленности матриц")
                print("- Очень больших или очень малых значениях ковариаций")
                print("- Длительных симуляциях с накоплением ошибок")
            else:
                print(f"\n✓ CKF и SRCF РАЗЛИЧАЮТСЯ")
                print(f"  SRCF лучше в {(self.df['error_difference'] < 0).sum() / len(self.df) * 100:.1f}% случаев")

        # Основные статистики
        print("\n" + "="*40)
        print("ОСНОВНЫЕ СТАТИСТИКИ")
        print("="*40)

        print(f"Количество шагов: {len(self.df)}")
        print(f"Временной диапазон: {self.df['time'].min():.4f} - {self.df['time'].max():.4f} с")
        print(f"Длительность симуляции: {self.df['time'].max() - self.df['time'].min():.4f} с")

        if 'true_phi' in self.df.columns:
            print(f"\nДиапазон истинного угла φ: {self.df['true_phi'].min():.15f} - {self.df['true_phi'].max():.15f}")
            print(f"Средний истинный угол φ: {self.df['true_phi'].mean():.15f}")

        if 'true_p' in self.df.columns:
            print(f"Диапазон истинной скорости p: {self.df['true_p'].min():.15f} - {self.df['true_p'].max():.15f}")
            print(f"Средняя истинная скорость p: {self.df['true_p'].mean():.15f}")

    def generate_comparison_report(self):
        """Генерация аналитического отчета в стиле Verhaegen & Van Dooren"""
        if self.comparison_df is None:
            print("⚠ Нет данных сравнений для аналитического отчета")
            return

        print("\n" + "="*70)
        print("АНАЛИТИЧЕСКИЙ ОТЧЕТ В СТИЛЕ VERHAEGEN & VAN DOOREN (1986)")
        print("="*70)

        # Анализ согласно Theorem 1 из статьи
        print("\n1. АНАЛИЗ ЧИСЛЕННОЙ УСТОЙЧИВОСТИ (Theorem 1):")
        print("   " + "-"*50)

        # Ошибки фильтров
        ckf_avg_error = self.comparison_df['CKF_Error'].mean()
        srcf_avg_error = self.comparison_df['SRCF_Error'].mean()
        error_ratio = srcf_avg_error / ckf_avg_error

        print(f"   Средняя ошибка CKF:    {ckf_avg_error:.15e}")
        print(f"   Средняя ошибка SRCF:   {srcf_avg_error:.15e}")
        print(f"   Отношение SRCF/CKF:    {error_ratio:.15f}")

        if error_ratio < 1.0:
            improvement = (1.0 - error_ratio) * 100
            print(f"   ✅ SRCF лучше на:       {improvement:.15f}%")
        else:
            improvement = (error_ratio - 1.0) * 100
            print(f"   ⚠ CKF лучше на:        {improvement:.15f}%")

        # Нормы ковариаций
        print("\n2. АНАЛИЗ КОВАРИАЦИОННЫХ МАТРИЦ:")
        print("   " + "-"*50)

        ckf_cov_norm = self.comparison_df['CKF_Cov_Norm'].mean()
        srcf_cov_norm = self.comparison_df['SRCF_Cov_Norm'].mean()
        cov_ratio = srcf_cov_norm / ckf_cov_norm

        print(f"   Средняя норма ковариации CKF:  {ckf_cov_norm:.15e}")
        print(f"   Средняя норма ковариации SRCF: {srcf_cov_norm:.15e}")
        print(f"   Отношение норм SRCF/CKF:       {cov_ratio:.15f}")

        # Числа обусловленности
        if 'CKF_Cond_Number' in self.comparison_df.columns:
            print("\n3. АНАЛИЗ ЧИСЕЛ ОБУСЛОВЛЕННОСТИ:")
            print("   " + "-"*50)

            ckf_cond = self.comparison_df['CKF_Cond_Number'].mean()
            srcf_cond = self.comparison_df['SRCF_Cond_Number'].mean()
            cond_ratio = srcf_cond / ckf_cond

            print(f"   Среднее cond CKF:    {ckf_cond:.15e}")
            print(f"   Среднее cond SRCF:   {srcf_cond:.15e}")
            print(f"   Отношение cond SRCF/CKF: {cond_ratio:.15f}")

            # Интерпретация согласно статье
            print("\n   ИНТЕРПРЕТАЦИЯ (согласно Verhaegen & Van Dooren):")
            if cond_ratio < 0.8:
                print("   ✅ SRCF демонстрирует лучшую численную обусловленность")
            elif cond_ratio > 1.2:
                print("   ⚠ CKF демонстрирует лучшую численную обусловленность")
            else:
                print("   ↔ Оба фильтра имеют сравнимую численную обусловленность")

        # Рекомендации
        print("\n4. РЕКОМЕНДАЦИИ:")
        print("   " + "-"*50)

        recommendations = []

        if error_ratio < 0.95:
            recommendations.append("✅ Использовать SRCF (лучшая точность)")
        elif error_ratio > 1.05:
            recommendations.append("⚠ Использовать CKF (лучшая точкость)")
        else:
            recommendations.append("↔ Оба фильтра равноценны по точности")

        if 'CKF_Cond_Number' in self.comparison_df.columns:
            if ckf_cond > 1e10 or srcf_cond > 1e10:
                recommendations.append("⚠ Высокая обусловленность - использовать SRCF")
            elif ckf_cond > 1e8 and cond_ratio < 0.9:
                recommendations.append("✅ SRCF предпочтительнее при плохой обусловленности")

        if self.comparison_df['CKF_Error'].std() / ckf_avg_error > 1.0:
            recommendations.append("⚠ Высокая дисперсия ошибок CKF - проверить устойчивость")

        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

        print("\n" + "="*70)

    def generate_report(self):
        """Генерация полного отчета"""
        print("\n" + "="*60)
        print("ГЕНЕРАЦИЯ ГРАФИКОВ СРАВНЕНИЯ ФИЛЬТРОВ КАЛМАНА")
        print("="*60)

        # Загружаем и анализируем данные
        if self.df is not None:
            self.calculate_errors()
            self.print_detailed_analysis()

        # Генерируем графики
        try:
            self.plot_state_comparison()
            print("✓ График 1: Сравнение оценок состояния")
        except Exception as e:
            print(f"✗ Ошибка в графике 1: {e}")

        try:
            self.plot_error_analysis()
            print("✓ График 2: Анализ ошибок")
        except Exception as e:
            print(f"✗ Ошибка в графике 2: {e}")

        try:
            self.plot_statistical_comparison()
            print("✓ График 3: Статистическое сравнение")
        except Exception as e:
            print(f"✗ Ошибка в графике 3: {e}")

        try:
            self.plot_phase_space()
            print("✓ График 4: Фазовое пространство")
        except Exception as e:
            print(f"✗ Ошибка в графике 4: {e}")

        try:
            self.plot_summary_metrics()
            print("✓ График 5: Сводные метрики")
        except Exception as e:
            print(f"✗ Ошибка в графике 5: {e}")

        if self.comparison_df is not None:
            try:
                self.plot_covariance_analysis()
                print("✓ График 6: Анализ ковариационных матриц")
            except Exception as e:
                print(f"✗ Ошибка в графике 6: {e}")

        try:
            self.plot_numerical_stability()
            print("✓ График 7: Анализ численной устойчивости")
        except Exception as e:
            print(f"✗ Ошибка в графике 7: {e}")

        if 'Innovation' in self.comparison_df.columns:
            try:
                self.plot_innovation_analysis()
                print("✓ График 8: Анализ инноваций")
            except Exception as e:
                print(f"✗ Ошибка в графике 8: {e}")

        print("\n" + "="*60)
        print("ОТЧЕТ СФОРМИРОВАН")
        print("="*60)
        print(f"Все графики сохранены в папке: {self.data_path}")
        print("\nФайлы:")
        for ext in ['.png', '.csv', '.txt']:
            files = [f for f in os.listdir(self.data_path) if f.endswith(ext)]
            if files:
                print(f"  {ext}: {', '.join(files[:3])}" + ("..." if len(files) > 3 else ""))

# Основной скрипт
if __name__ == "__main__":
    print("="*60)
    print("ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ ФИЛЬТРОВ КАЛМАНА")
    print("="*60)

    # Путь к данным - можно изменить при вызове
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Пробуем разные возможные пути
        possible_paths = [
            './cmake-build-debug/data/verhaegen_test6_easy',
            './data/verhaegen_test6_easy',
            '../cmake-build-debug/data/verhaegen_test6_easy',
            '../data/verhaegen_test6_easy',
            '../../data/verhaegen_test6_easy',
            'data/verhaegen_test6_easy'
        ]

        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                print(f"✓ Автоопределение пути: {path}")
                break

        if data_path is None:
            data_path = './data/verhaegen_test6_easy'
            print(f"⚠ Путь не найден, используем по умолчанию: {data_path}")
            print("  Для указания другого пути запустите:")
            print(f"    python {sys.argv[0]} /ваш/путь/к/данным")

    # Создаем визуализатор и генерируем отчет
    visualizer = KalmanVisualizer(data_path)
    visualizer.generate_report()
    visualizer.generate_comparison_report()

    # Дополнительная информация
    print("\n" + "="*60)
    print("ПРИМЕЧАНИЯ:")
    print("="*60)
    print("1. CKF и SRCF могут давать идентичные результаты для хорошо")
    print("   обусловленных задач - это нормально!")
    print("2. Для создания различий увеличьте уровень шума в C++ коде:")
    print("   - process_noise_scale = 100.0")
    print("   - measurement_noise_scale = 50.0")
    print("   - initial_covariance = Identity * 1000")
    print("3. Новые имена колонок в CSV:")
    print("   - meas_gyro_exact/noisy: измерения гироскопа (угловая скорость p)")
    print("   - meas_accel_exact/noisy: измерения акселерометра (g·sin(φ))")
    print("4. Все графики показывают оба фильтра, даже если они идентичны")

    visualizer.plot_kalman_x_like_photo(filter_name='srcf')  # или 'srcf'