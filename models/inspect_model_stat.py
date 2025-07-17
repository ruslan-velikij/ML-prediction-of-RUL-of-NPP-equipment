import joblib
import numpy as np
import matplotlib.pyplot as plt

model = joblib.load('models/model_stat.pkl')

print("\n1) Вывод основных свойств:")
print("Описание модели:\n", model)
print("\nПараметры Weibull:")
print(f"  • rho_    = {model.rho_:.4f}")
print(f"  • lambda_ = {model.lambda_:.4f}")

print("\n2) Таблица summary (коэффициенты и CI):")
print("Summary (коэффициенты и доверительные интервалы):")
print(model.summary)

print("\n3) Мини-пример: survival-функция для ряда точек времени:")
t = np.linspace(0, 100, 11)
sf = model.survival_function_at_times(t)
print("Survival function S(t) для t=0..100:")
print(sf)

print("\n4) Визуализация:")
ax = model.plot_survival_function()
ax.set_title("Survival Function из inspect_model.py")
plt.savefig('results/inspect_survival_inspect_stat.png')
print("График сохранён в results/inspect_survival_inspect.png")
