import sys
import os
import json # Добавлен импорт json для команды "показать состояние"

# Добавляем директорию src в PYTHONPATH, чтобы можно было импортировать модули из нее
# Это необходимо, так как main.py находится в корне, а модули в src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from embedding_generator import EmbeddingGenerator
    from state_manager import StateManager
    from controller import Controller
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что скрипт main.py находится в корневой директории проекта,")
    print("а модули embedding_generator, state_manager, controller находятся в директории src/.")
    print("Также проверьте, что все зависимости установлены (pip install -r requirements.txt).")
    sys.exit(1)

def print_welcome_message():
    """Печатает приветственное сообщение и инструкции."""
    print("\nДобро пожаловать в прототип интеллектуального ассистента!")
    print("Система готова к обработке ваших предложений.")
    print("Примеры команд, которые система может попытаться понять:")
    print("  - 'В комнате 5 яблок.'")
    print("  - 'Добавили 3 яблока.'")
    print("  - 'Убрали 2 яблока.'")
    print("  - 'Сколько яблок?'")
    print("  - 'Где яблоки?'")
    print("  - 'сохранить состояние <имя_файла>' (например, 'сохранить состояние my_state') - .json добавится автоматически")
    print("  - 'загрузить состояние <имя_файла>' (например, 'загрузить состояние my_state') - .json добавится автоматически")
    print("  - 'показать состояние'")
    print("  - 'сбросить состояние'")
    print("Для выхода введите 'выход', 'exit' или 'quit'.\n")

def main_loop():
    """Основной цикл программы для взаимодействия с пользователем."""
    try:
        print("Инициализация компонентов системы...")
        embedding_gen = EmbeddingGenerator()
        if not hasattr(embedding_gen, 'model') or embedding_gen.model is None:
            print("Критическая ошибка: Не удалось загрузить модель эмбеддингов.")
            print("Проверьте интернет-соединение (для первой загрузки модели) и имя модели.")
            print("Работа программы будет прекращена.")
            return

        state_mng = StateManager()
        contr = Controller(embedding_gen, state_mng)
        print("Компоненты успешно инициализированы.")
    except Exception as e:
        print(f"Произошла ошибка при инициализации компонентов: {e}")
        import traceback
        print(traceback.format_exc())
        print("Работа программы будет прекращена.")
        return

    print_welcome_message()

    # Создаем директорию data, если ее нет, для сохранения состояний
    data_dir = "data"
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir, exist_ok=True)
        except OSError as e:
            print(f"Не удалось создать директорию {data_dir}: {e}. Сохранение/загрузка состояния может не работать.")


    while True:
        try:
            user_input = input("Вы: ").strip()
        except KeyboardInterrupt:
            print("\nПолучен сигнал прерывания. Завершение работы...")
            break
        except EOFError: # Обработка Ctrl+D
            print("\nДостигнут конец ввода. Завершение работы...")
            break

        if not user_input:
            continue

        if user_input.lower() in ["выход", "exit", "quit"]:
            print("Завершение работы. До свидания!")
            break

        elif user_input.lower().startswith("сохранить состояние "):
            parts = user_input.split(maxsplit=2)
            if len(parts) == 3:
                filename = parts[2].strip()
                if not filename:
                    print("Система: Имя файла для сохранения не может быть пустым.")
                    continue
                if not filename.endswith(".json"):
                    filename += ".json"
                filepath = os.path.join(data_dir, filename)

                if state_mng.save_state_to_file(filepath):
                    print(f"Система: Состояние сохранено в {filepath}")
                else:
                    print(f"Система: Не удалось сохранить состояние в {filepath}")
            else:
                print("Система: Неверный формат команды. Используйте: 'сохранить состояние <имя_файла>'")
            continue

        elif user_input.lower().startswith("загрузить состояние "):
            parts = user_input.split(maxsplit=2)
            if len(parts) == 3:
                filename = parts[2].strip()
                if not filename:
                    print("Система: Имя файла для загрузки не может быть пустым.")
                    continue
                if not filename.endswith(".json"):
                    filename += ".json"
                filepath = os.path.join(data_dir, filename)
                if state_mng.load_state_from_file(filepath):
                    print(f"Система: Состояние загружено из {filepath}")
                else:
                    print(f"Система: Не удалось загрузить состояние из {filepath}.")
            else:
                print("Система: Неверный формат команды. Используйте: 'загрузить состояние <имя_файла>'")
            continue

        elif user_input.lower() == "показать состояние":
            current_state = state_mng.get_current_state()
            print("Система: Текущее состояние:")
            print(json.dumps(current_state, ensure_ascii=False, indent=4))
            continue

        elif user_input.lower() == "сбросить состояние":
            state_mng.reset_state()
            print("Система: Состояние было сброшено.")
            continue

        response = contr.process_sentence(user_input)
        print(f"Система: {response}")

if __name__ == "__main__":
    main_loop()
