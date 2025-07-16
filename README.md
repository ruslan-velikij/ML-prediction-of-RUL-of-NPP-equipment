# ML-prediction-of-RUL-of-NPP-equipment
Различные модели машинного обучения для прогнозирования остаточного ресурса промышленного оборудования, в том чисте оборудования АЭС

# Подготовка датасетов
Скачать датасеты по ссылке: https://disk.yandex.ru/d/gC3CcPDnitIo2w
Распаковать в корень проекта

# Подготовка библиотек
## Для пользователей Linux/macOS
- перейти в корень проекта
- активировать виртуальное окружение: source venv/bin/activate
- скачать библиотеки: pip install -r requirements.txt

## Для пользователей Windows
- перейти в корень проекта
- если политика выполнения скриптов запрещает запуск, выполнить сначала (PowerShell): Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
- активировать виртуальное окружение: .\venv\Scripts\Activate.ps1
- скачать библиотеки: pip install -r requirements.txt