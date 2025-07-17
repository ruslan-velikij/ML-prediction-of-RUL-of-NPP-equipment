import sys
import os
from data_prep import prep_stat
from models import fit_weibull

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("Подготавливаем данные для статистического анализа отказов...")
    stat_df = prep_stat()

    print("Обучаем Weibull-модель и строим survival-кривую...")
    fit_weibull(stat_df)

    print(
          "Готово! Модель сохранена в models/model_stat.pkl, "
          "график — в results/stat_performance.png"
        )


if __name__ == "__main__":
    main()
