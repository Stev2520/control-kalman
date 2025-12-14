import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import os
from matplotlib.patches import Ellipse

class KalmanVisualizer:
    def __init__(self, data_path='./cmake-build-debug/data/test_process_noise'):
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
                './data/test_process_noise',
                '../data/test_process_noise',
                '../../data/test_process_noise',
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

        # Загружаем метрики если есть
        self.load_metrics_files()

        # Если данные всё ещё не загружены, создаём тестовые
        if self.df is None:
            print("\n⚠ Данные не найдены, создаю тестовые данные...")
            self.create_test_data()
        else:
            # НОВОЕ: Анализ структуры данных
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
            print(f"  Отличается от true_p на: {gyro_diff.mean():.3e} ± {gyro_diff.std():.3e}")
            print(f"  Диапазон: {self.df['meas_gyro_exact'].min():.3f} ... {self.df['meas_gyro_exact'].max():.3f}")

        if 'meas_accel_exact' in self.df.columns and 'true_phi' in self.df.columns:
            true_accel = g * np.sin(self.df['true_phi'])
            accel_diff = (self.df['meas_accel_exact'] - true_accel).abs()
            print(f"\nАкселерометр (meas_accel_exact):")
            print(f"  Отличается от g·sin(φ) на: {accel_diff.mean():.3e} ± {accel_diff.std():.3e}")
            print(f"  Диапазон: {self.df['meas_accel_exact'].min():.3f} ... {self.df['meas_accel_exact'].max():.3f}")

        # Проверяем шум
        if 'meas_gyro_exact' in self.df.columns and 'meas_gyro_noisy' in self.df.columns:
            gyro_noise = (self.df['meas_gyro_noisy'] - self.df['meas_gyro_exact']).abs()
            print(f"\nШум гироскопа (meas_gyro_noisy - meas_gyro_exact):")
            print(f"  Средний: {gyro_noise.mean():.3e}, Макс: {gyro_noise.max():.3e}")

        if 'meas_accel_exact' in self.df.columns and 'meas_accel_noisy' in self.df.columns:
            accel_noise = (self.df['meas_accel_noisy'] - self.df['meas_accel_exact']).abs()
            print(f"Шум акселерометра (meas_accel_noisy - meas_accel_exact):")
            print(f"  Средний: {accel_noise.mean():.3e}, Макс: {accel_noise.max():.3e}")

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
        """Создание тестовых данных только если нет реальных"""
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

        # Теперь рассчитываем ошибки
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
            are_identical = np.allclose(self.df['ckf_phi'], self.df['srcf_phi'], atol=1e-10)
            print(f"✓ Фильтры идентичны по φ: {are_identical}")

            if are_identical:
                print("⚠ CKF и SRCF дают ИДЕНТИЧНЫЕ результаты")
                print("  Это нормально для хорошо обусловленных задач")

    def plot_state_comparison(self):
        """График 1: Сравнение оценок состояния"""
        if self.df is None:
            print("⚠ Нет данных для построения графика 1")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

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
        if not np.allclose(ckf_cumulative, srcf_cumulative, atol=1e-10):
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
        if not np.allclose(diff, 0, atol=1e-10):
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

        if not np.allclose(diff, 0, atol=1e-10):
            ax.hist(diff, bins=30, color='purple', alpha=0.7, edgecolor='black')
        else:
            # Если разности нулевые, показываем одну полосу
            ax.bar([0], [len(diff)], width=0.1, color='gray', alpha=0.7, edgecolor='black')

        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Нулевая разность')
        ax.axvline(x=diff.mean(), color='green', linestyle='-', linewidth=2,
                   label=f'Среднее: {diff.mean():.2e}')

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

        if not np.allclose(ckf_ma, srcf_ma, atol=1e-10):
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
            if not np.allclose(sorted_ckf, sorted_srcf, atol=1e-10):
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

        better_srcf = (self.df['error_difference'] < -1e-10).sum() / len(self.df) * 100
        better_ckf = (self.df['error_difference'] > 1e-10).sum() / len(self.df) * 100
        equal = (np.abs(self.df['error_difference']) <= 1e-10).sum() / len(self.df) * 100

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
        center_text += f"SRCF лучше: {better_srcf:.1f}%\n"
        center_text += f"CKF лучше: {better_ckf:.1f}%\n"
        center_text += f"Средняя разность: {self.df['error_difference'].mean():.2e}"

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
                    pass  # Пропускаем если не удалось вычислить эллипсы
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
                ax2.text(i, value, f'{value:.3e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
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
            print(f"  CKF:  {ckf_val:.6e}")
            print(f"  SRCF: {srcf_val:.6e}")

            if not np.isclose(ratio, 1.0, atol=1e-10):
                better = "SRCF" if ratio < 1 else "CKF"
                improvement = abs(1 - ratio) * 100
                print(f"  Отношение (SRCF/CKF): {ratio:.6f}")
                print(f"  {better} лучше на {improvement:.2f}%")
            else:
                print(f"  Отношение (SRCF/CKF): 1.000000 (идентичны)")

        if additional_metrics:
            print(f"\nМетрики измерений:")
            for name, value in additional_metrics.items():
                print(f"  {name}: {value:.6e}")

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

            print(f"Максимальное расхождение по φ: {max_phi_diff:.2e}")
            print(f"Максимальное расхождение по p: {max_p_diff:.2e}")

            are_identical = max_phi_diff < 1e-10 and max_p_diff < 1e-10

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
        print(f"Временной диапазон: {self.df['time'].min():.2f} - {self.df['time'].max():.2f} с")
        print(f"Длительность симуляции: {self.df['time'].max() - self.df['time'].min():.2f} с")

        if 'true_phi' in self.df.columns:
            print(f"\nДиапазон истинного угла φ: {self.df['true_phi'].min():.3f} - {self.df['true_phi'].max():.3f}")
            print(f"Средний истинный угол φ: {self.df['true_phi'].mean():.3f}")

        if 'true_p' in self.df.columns:
            print(f"Диапазон истинной скорости p: {self.df['true_p'].min():.3f} - {self.df['true_p'].max():.3f}")
            print(f"Средняя истинная скорость p: {self.df['true_p'].mean():.3f}")

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
            './cmake-build-debug/data/test_process_noise',
            './data/test_process_noise',
            '../cmake-build-debug/data/test_process_noise',
            '../data/test_process_noise',
            '../../data/test_process_noise',
            'data/test_process_noise'
        ]

        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                print(f"✓ Автоопределение пути: {path}")
                break

        if data_path is None:
            data_path = './data/test_process_noise'
            print(f"⚠ Путь не найден, используем по умолчанию: {data_path}")
            print("  Для указания другого пути запустите:")
            print(f"    python {sys.argv[0]} /ваш/путь/к/данным")

    # Создаем визуализатор и генерируем отчет
    visualizer = KalmanVisualizer(data_path)
    visualizer.generate_report()

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