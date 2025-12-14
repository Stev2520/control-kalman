import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import os

class KalmanVisualizer:
    def __init__(self, data_path='./cmake-build-debug/data/test_with_noise'):
        self.data_path = data_path
        self.df = None
        self.times = None
        self.load_data()

    def load_data(self):
        """Загрузка данных из CSV файлов"""
        try:
            # Загрузка основных данных
            self.df = pd.read_csv(f'{self.data_path}/simulation_data.csv')

            # Загрузка метрик
            with open(f'{self.data_path}/filter_metrics.txt', 'r') as f:
                self.metrics_text = f.read()

            with open(f'{self.data_path}/comparison_details.txt', 'r') as f:
                self.comparison_text = f.read()

            print(f"Данные загружены: {len(self.df)} строк")
            print(f"Колонки: {list(self.df.columns)}")

        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            # Создаем тестовые данные если файлов нет
            self.create_test_data()

    def create_test_data(self):
        """Создание тестовых данных если файлы отсутствуют"""
        print("Создание тестовых данных...")
        n = 100
        t = np.linspace(0, 10, n)

        # Истинные состояния (синусоида с трендом)
        true_phi = 2 * np.sin(2 * np.pi * 0.5 * t) + 0.1 * t
        true_p = np.cos(2 * np.pi * 0.3 * t) + 0.05 * t

        # Зашумленные измерения
        noise_level = 0.5
        meas_phi = true_phi + np.random.normal(0, noise_level, n)
        meas_p = true_p + np.random.normal(0, noise_level, n)

        # Оценки фильтров (с преднамеренными различиями)
        ckf_phi = true_phi + np.random.normal(0, 0.2, n) * (1 + 0.1 * np.sin(t))
        ckf_p = true_p + np.random.normal(0, 0.15, n)

        srcf_phi = true_phi + np.random.normal(0, 0.15, n) * (1 + 0.05 * np.sin(t))
        srcf_p = true_p + np.random.normal(0, 0.1, n)

        self.df = pd.DataFrame({
            'time': t,
            'true_phi': true_phi,
            'true_p': true_p,
            'meas_phi_noisy': meas_phi,
            'meas_p_noisy': meas_p,
            'ckf_phi': ckf_phi,
            'ckf_p': ckf_p,
            'srcf_phi': srcf_phi,
            'srcf_p': srcf_p
        })

    def calculate_errors(self):
        """Расчет ошибок фильтров"""
        self.df['ckf_error_phi'] = np.abs(self.df['ckf_phi'] - self.df['true_phi'])
        self.df['ckf_error_p'] = np.abs(self.df['ckf_p'] - self.df['true_p'])
        self.df['ckf_total_error'] = np.sqrt(self.df['ckf_error_phi']**2 + self.df['ckf_error_p']**2)

        self.df['srcf_error_phi'] = np.abs(self.df['srcf_phi'] - self.df['true_phi'])
        self.df['srcf_error_p'] = np.abs(self.df['srcf_p'] - self.df['true_p'])
        self.df['srcf_total_error'] = np.sqrt(self.df['srcf_error_phi']**2 + self.df['srcf_error_p']**2)

        self.df['error_difference'] = self.df['srcf_total_error'] - self.df['ckf_total_error']

    def plot_state_comparison(self):
        """График 1: Сравнение оценок состояния"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Угол phi
        ax = axes[0, 0]
        ax.plot(self.df['time'], self.df['true_phi'], 'k-', linewidth=2, label='Истинное')
        ax.plot(self.df['time'], self.df['meas_phi_noisy'], 'r.', alpha=0.3, markersize=3, label='Измерения')
        ax.plot(self.df['time'], self.df['ckf_phi'], 'b-', linewidth=1.5, label='CKF')
        ax.plot(self.df['time'], self.df['srcf_phi'], 'g-', linewidth=1.5, label='SRCF')
        ax.set_xlabel('Время (с)')
        ax.set_ylabel('Угол φ')
        ax.set_title('Сравнение оценок угла φ')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Угловая скорость p
        ax = axes[0, 1]
        ax.plot(self.df['time'], self.df['true_p'], 'k-', linewidth=2, label='Истинное')
        ax.plot(self.df['time'], self.df['meas_p_noisy'], 'r.', alpha=0.3, markersize=3, label='Измерения')
        ax.plot(self.df['time'], self.df['ckf_p'], 'b-', linewidth=1.5, label='CKF')
        ax.plot(self.df['time'], self.df['srcf_p'], 'g-', linewidth=1.5, label='SRCF')
        ax.set_xlabel('Время (с)')
        ax.set_ylabel('Скорость p')
        ax.set_title('Сравнение оценок угловой скорости p')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Ошибки по phi
        ax = axes[1, 0]
        ax.plot(self.df['time'], self.df['ckf_error_phi'], 'b-', alpha=0.7, label='CKF ошибка')
        ax.plot(self.df['time'], self.df['srcf_error_phi'], 'g-', alpha=0.7, label='SRCF ошибка')
        ax.fill_between(self.df['time'], 0, self.df['ckf_error_phi'], alpha=0.2, color='blue')
        ax.fill_between(self.df['time'], 0, self.df['srcf_error_phi'], alpha=0.2, color='green')
        ax.set_xlabel('Время (с)')
        ax.set_ylabel('Абсолютная ошибка φ')
        ax.set_title('Ошибки оценки угла')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Ошибки по p
        ax = axes[1, 1]
        ax.plot(self.df['time'], self.df['ckf_error_p'], 'b-', alpha=0.7, label='CKF ошибка')
        ax.plot(self.df['time'], self.df['srcf_error_p'], 'g-', alpha=0.7, label='SRCF ошибка')
        ax.fill_between(self.df['time'], 0, self.df['ckf_error_p'], alpha=0.2, color='blue')
        ax.fill_between(self.df['time'], 0, self.df['srcf_error_p'], alpha=0.2, color='green')
        ax.set_xlabel('Время (с)')
        ax.set_ylabel('Абсолютная ошибка p')
        ax.set_title('Ошибки оценки скорости')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.data_path}/state_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_error_analysis(self):
        """График 2: Детальный анализ ошибок"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Кумулятивные ошибки
        ax = axes[0, 0]
        ckf_cumulative = self.df['ckf_total_error'].cumsum()
        srcf_cumulative = self.df['srcf_total_error'].cumsum()
        ax.plot(self.df['time'], ckf_cumulative, 'b-', linewidth=2, label='CKF кумулятивная')
        ax.plot(self.df['time'], srcf_cumulative, 'g-', linewidth=2, label='SRCF кумулятивная')
        ax.fill_between(self.df['time'], ckf_cumulative, srcf_cumulative,
                        where=(srcf_cumulative < ckf_cumulative),
                        alpha=0.3, color='green', label='SRCF лучше')
        ax.fill_between(self.df['time'], ckf_cumulative, srcf_cumulative,
                        where=(srcf_cumulative >= ckf_cumulative),
                        alpha=0.3, color='blue', label='CKF лучше')
        ax.set_xlabel('Время (с)')
        ax.set_ylabel('Накопленная ошибка')
        ax.set_title('Кумулятивное сравнение ошибок')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Разность ошибок
        ax = axes[0, 1]
        diff = self.df['error_difference']
        ax.plot(self.df['time'], diff, 'purple', linewidth=1.5)
        ax.fill_between(self.df['time'], 0, diff, where=(diff < 0),
                        alpha=0.3, color='green', label='SRCF лучше')
        ax.fill_between(self.df['time'], 0, diff, where=(diff >= 0),
                        alpha=0.3, color='blue', label='CKF лучше')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Время (с)')
        ax.set_ylabel('Разность ошибок (SRCF - CKF)')
        ax.set_title('Динамика превосходства фильтров')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Гистограмма разностей
        ax = axes[1, 0]
        ax.hist(diff, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Нулевая разность')
        ax.axvline(x=diff.mean(), color='green', linestyle='-', linewidth=2,
                   label=f'Среднее: {diff.mean():.3e}')
        ax.set_xlabel('Разность ошибок')
        ax.set_ylabel('Частота')
        ax.set_title('Распределение разностей ошибок')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Скользящее среднее ошибок
        ax = axes[1, 1]
        window = 20
        ckf_ma = self.df['ckf_total_error'].rolling(window=window).mean()
        srcf_ma = self.df['srcf_total_error'].rolling(window=window).mean()
        ax.plot(self.df['time'], ckf_ma, 'b-', linewidth=2, label=f'CKF ({window}-шаговое ср.)')
        ax.plot(self.df['time'], srcf_ma, 'g-', linewidth=2, label=f'SRCF ({window}-шаговое ср.)')
        ax.fill_between(self.df['time'], ckf_ma, srcf_ma,
                        where=(srcf_ma < ckf_ma), alpha=0.3, color='green')
        ax.set_xlabel('Время (с)')
        ax.set_ylabel('Скользящее среднее ошибки')
        ax.set_title('Тренды ошибок (скользящее среднее)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.data_path}/error_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_statistical_comparison(self):
        """График 3: Статистическое сравнение"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Боксплот ошибок
        ax = axes[0]
        error_data = [self.df['ckf_total_error'], self.df['srcf_total_error']]
        bp = ax.boxplot(error_data, patch_artist=True,
                        labels=['CKF', 'SRCF'],
                        medianprops=dict(color='yellow', linewidth=2))
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_ylabel('Общая ошибка')
        ax.set_title('Статистическое распределение ошибок')
        ax.grid(True, alpha=0.3)

        # Квантиль-квантиль график
        ax = axes[1]
        if len(self.df) > 10:
            sorted_ckf = np.sort(self.df['ckf_total_error'])
            sorted_srcf = np.sort(self.df['srcf_total_error'])
            ax.plot(sorted_ckf, sorted_srcf, 'bo', alpha=0.5, markersize=4)
            min_val = min(sorted_ckf.min(), sorted_srcf.min())
            max_val = max(sorted_ckf.max(), sorted_srcf.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Линия равенства')
            ax.set_xlabel('CKF ошибки (отсортированные)')
            ax.set_ylabel('SRCF ошибки (отсортированные)')
            ax.set_title('Q-Q график сравнения распределений')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Процент превосходства по времени
        ax = axes[2]
        better_srcf = (self.df['error_difference'] < 0).sum() / len(self.df) * 100
        better_ckf = (self.df['error_difference'] > 0).sum() / len(self.df) * 100
        equal = (self.df['error_difference'] == 0).sum() / len(self.df) * 100

        labels = ['SRCF лучше', 'CKF лучше', 'Равны']
        sizes = [better_srcf, better_ckf, equal]
        colors = ['lightgreen', 'lightblue', 'lightgray']

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        ax.set_title('Процентное соотношение превосходства')

        # Добавляем статистику в центр
        center_text = f"Всего шагов: {len(self.df)}\n"
        center_text += f"SRCF лучше: {better_srcf:.1f}%\n"
        center_text += f"CKF лучше: {better_ckf:.1f}%\n"
        center_text += f"Средняя разность: {self.df['error_difference'].mean():.3e}"

        ax.text(0, 0, center_text, ha='center', va='center',
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        plt.savefig(f'{self.data_path}/statistical_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_phase_space(self):
        """График 4: Фазовое пространство"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Фазовый портрет истинного состояния
        ax = axes[0]
        sc1 = ax.scatter(self.df['true_phi'], self.df['true_p'],
                         c=self.df['time'], cmap='viridis', s=20, alpha=0.7)
        ax.set_xlabel('Угол φ')
        ax.set_ylabel('Скорость p')
        ax.set_title('Фазовый портрет истинного состояния')
        plt.colorbar(sc1, ax=ax, label='Время (с)')
        ax.grid(True, alpha=0.3)

        # Фазовый портрет ошибок
        ax = axes[1]
        ckf_dist = np.sqrt(self.df['ckf_error_phi']**2 + self.df['ckf_error_p']**2)
        srcf_dist = np.sqrt(self.df['srcf_error_phi']**2 + self.df['srcf_error_p']**2)

        sc2 = ax.scatter(self.df['ckf_error_phi'], self.df['ckf_error_p'],
                         c=ckf_dist, cmap='Blues', s=30, alpha=0.6, label='CKF ошибки')
        sc3 = ax.scatter(self.df['srcf_error_phi'], self.df['srcf_error_p'],
                         c=srcf_dist, cmap='Greens', s=30, alpha=0.6, label='SRCF ошибки')

        ax.set_xlabel('Ошибка угла φ')
        ax.set_ylabel('Ошибка скорости p')
        ax.set_title('Фазовое пространство ошибок')
        ax.legend()
        plt.colorbar(sc2, ax=ax, label='Величина ошибки')
        ax.grid(True, alpha=0.3)

        # Добавляем эллипсы доверительных областей
        if len(self.df) > 50:
            from matplotlib.patches import Ellipse
            # Для CKF
            ckf_cov = np.cov(self.df['ckf_error_phi'], self.df['ckf_error_p'])
            lambda_, v = np.linalg.eig(ckf_cov)
            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(xy=(self.df['ckf_error_phi'].mean(), self.df['ckf_error_p'].mean()),
                          width=lambda_[0]*2*2, height=lambda_[1]*2*2,
                          angle=np.degrees(np.arctan2(v[1,0], v[0,0])),
                          edgecolor='blue', facecolor='none', linewidth=2, linestyle='--')
            ax.add_patch(ell)

            # Для SRCF
            srcf_cov = np.cov(self.df['srcf_error_phi'], self.df['srcf_error_p'])
            lambda_, v = np.linalg.eig(srcf_cov)
            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(xy=(self.df['srcf_error_phi'].mean(), self.df['srcf_error_p'].mean()),
                          width=lambda_[0]*2*2, height=lambda_[1]*2*2,
                          angle=np.degrees(np.arctan2(v[1,0], v[0,0])),
                          edgecolor='green', facecolor='none', linewidth=2, linestyle='--')
            ax.add_patch(ell)

        plt.tight_layout()
        plt.savefig(f'{self.data_path}/phase_space.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_summary_metrics(self):
        """График 5: Сводные метрики"""
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

        fig, ax = plt.subplots(figsize=(12, 6))

        labels = list(metrics['CKF'].keys())
        ckf_values = list(metrics['CKF'].values())
        srcf_values = list(metrics['SRCF'].values())

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, ckf_values, width, label='CKF', color='lightblue', edgecolor='blue')
        bars2 = ax.bar(x + width/2, srcf_values, width, label='SRCF', color='lightgreen', edgecolor='green')

        # Добавляем числовые значения
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3e}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Метрики')
        ax.set_ylabel('Значение')
        ax.set_title('Сравнение метрик производительности фильтров')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{self.data_path}/summary_metrics.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Печатаем метрики
        print("\n" + "="*60)
        print("СВОДНЫЕ МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("="*60)
        for metric in labels:
            ckf_val = metrics['CKF'][metric]
            srcf_val = metrics['SRCF'][metric]
            ratio = srcf_val / ckf_val if ckf_val != 0 else float('inf')
            better = "SRCF" if ratio < 1 else "CKF" if ratio > 1 else "Одинаково"
            improvement = abs(1 - ratio) * 100

            print(f"{metric}:")
            print(f"  CKF:  {ckf_val:.6f}")
            print(f"  SRCF: {srcf_val:.6f}")
            print(f"  Отношение (SRCF/CKF): {ratio:.4f}")
            if ratio != 1:
                print(f"  {better} лучше на {improvement:.2f}%")
            print()

    def generate_report(self):
        """Генерация полного отчета"""
        self.calculate_errors()

        print("Генерация графиков сравнения фильтров Калмана...")
        print("="*60)

        self.plot_state_comparison()
        print("✓ График 1: Сравнение оценок состояния сохранен")

        self.plot_error_analysis()
        print("✓ График 2: Анализ ошибок сохранен")

        self.plot_statistical_comparison()
        print("✓ График 3: Статистическое сравнение сохранен")

        self.plot_phase_space()
        print("✓ График 4: Фазовое пространство сохранен")

        self.plot_summary_metrics()
        print("✓ График 5: Сводные метрики сохранен")

        print("\n" + "="*60)
        print("ВСЕ ГРАФИКИ СОХРАНЕНЫ В ПАПКУ:", self.data_path)
        print("="*60)

# Основной скрипт
if __name__ == "__main__":
    # Инициализация визуализатора
    visualizer = KalmanVisualizer('./cmake-build-debug/data/test_with_noise')  # Укажите вашу папку с данными

    # Генерация полного отчета
    visualizer.generate_report()

    # Дополнительно: если хотите протестировать на идеальных данных
    print("\n" + "="*60)
    print("ТЕСТ НА ИДЕАЛЬНЫХ ДАННЫХ (для сравнения)")
    print("="*60)

    test_visualizer = KalmanVisualizer('./cmake-build-debug/data/test_with_noise')
    test_visualizer.calculate_errors()

    # Проверяем, идентичны ли фильтры
    ckf_errors = test_visualizer.df['ckf_total_error']
    srcf_errors = test_visualizer.df['srcf_total_error']

    if np.allclose(ckf_errors, srcf_errors, atol=1e-10):
        print("ВНИМАНИЕ: CKF и SRCF дают идентичные результаты!")
        print("Это означает:")
        print("1. Реализации математически эквивалентны для данного случая")
        print("2. Уровень шума слишком низкий для проявления различий")
        print("3. Численная устойчивость SRCF не проявляется")
        print("\nРекомендации:")
        print("• Увеличьте уровень шума (process_noise_scale, measurement_noise_scale)")
        print("• Используйте неравномерную временную сетку (RANDOM_JITTER)")
        print("• Задайте экстремальные начальные условия")
    else:
        print(f"Фильтры различаются. Средняя разность: {(srcf_errors - ckf_errors).mean():.3e}")