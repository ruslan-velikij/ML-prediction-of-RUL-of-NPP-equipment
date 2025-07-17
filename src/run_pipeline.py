import sys
import os
from data_prep import prep_stat, prep_deg
from models import fit_weibull, fit_degradation

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    # Блок запуска модели 1
    print("Подготавливаем данные для статистического анализа отказов...")
    stat_df = prep_stat()

    print("Обучаем Weibull-модель и строим survival-кривую...")
    fit_weibull(stat_df)

    print(
          "Готово! Модель сохранена в models/model_stat.pkl, "
          "график — в results/stat_performance.png"
        )

    # Блок запуска модели 2
    print("Подготавливаем данные для экстраполяции деградационного параметра...")
    deg_df = prep_deg()
    print("Строим тренд и вычисляем RUL...")
    result = fit_degradation(deg_df)

    RUL_days = result['RUL_days']
    if RUL_days >= 1:
        RUL_str = f"{RUL_days:.1f} дней"
    else:
        RUL_hours = RUL_days * 24
        RUL_str = f"{RUL_hours:.1f} часов"
    print(
          "Готово! Модель сохранена в models/model_deg.pkl, "
          f"график — в results/deg_performance.png, прогноз RUL = {RUL_str}"
    )


if __name__ == "__main__":
    main()
