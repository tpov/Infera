import sys
import os
import json

# Добавляем директорию src в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from state_manager import StateManager
    from controller import NLUController
    from logical_controller import LogicalController # <--- Добавляем импорт
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure main.py is in the project root directory,")
    print("and modules (state_manager, controller) are in the src/ directory.")
    print("Also, check if all dependencies are installed (pip install -r requirements.txt).")
    sys.exit(1)

def print_welcome_message_en():
    """Prints the welcome message and instructions in English."""
    print("\nWelcome to the Intelligent Assistant Prototype!")
    print("The system is ready to process your sentences in English.")
    print("Examples of commands the system might understand (after NLU models are trained):")
    print("  - 'in the kitchen there are 5 red apples'")
    print("  - 'add 3 books'")
    print("  - 'remove 2 apples'")
    print("  - 'how many apples?'")
    print("  - 'where are the books?'")
    print("  - 'what color are the apples?'")
    print("  - 'save state <filename>' (e.g., 'save state my_session_state') - .json will be added")
    print("  - 'load state <filename>' (e.g., 'load state my_session_state') - .json will be added")
    print("  - 'show state'")
    print("  - 'reset state'")
    print("To exit, type 'exit', 'quit', or 'bye'.\n")

def main_loop():
    """Main loop for user interaction."""
    # Determine device
    device_param = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main.py will suggest NLUController to use device: {device_param}")

    try:
        print("Initializing system components...")
        state_mng = StateManager()
        nlu_ctrl = NLUController(state_manager=state_mng, device=device_param) # NLUController теперь может не требовать state_manager при инициализации
                                                                            # или он может быть None, если его логика работы с состоянием полностью ушла.
                                                                            # В текущей реализации NLUController он не использует state_manager.

        # Проверка, что модели загрузились в NLUController
        if not nlu_ctrl.tokenizer or not nlu_ctrl.intent_model or not nlu_ctrl.slot_model:
            print("Critical Error: NLU models or tokenizer could not be loaded in NLUController.")
            # Используем пути, которые NLUController использует для загрузки
            # (они должны быть доступны как атрибуты класса или через какой-то config)
            # Для простоты пока оставим INTENT_MODEL_PATH, SLOT_MODEL_PATH как глобальные или импортируемые
            # из controller.py, как было сделано ниже в if __name__ == "__main__"
            intent_model_path_check = "models/intent_classifier_bert_en" # Захардкодим для сообщения
            slot_model_path_check = "models/slot_filler_bert_en"
            print(f"Please ensure trained models exist at '{intent_model_path_check}' and '{slot_model_path_check}'.")
            print("Run training scripts (train_intent_classifier.py, train_slot_filler.py) first.")
            print("Exiting program.")
            return

        # Инициализируем LogicalController
        logic_ctrl = LogicalController(nlu_controller=nlu_ctrl, state_manager=state_mng)

        print("Components initialized successfully.")
    except Exception as e:
        print(f"An error occurred during component initialization: {e}")
        import traceback
        print(traceback.format_exc())
        print("Exiting program.")
        return

    print_welcome_message_en()

    data_dir = "data"
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir, exist_ok=True)
        except OSError as e:
            print(f"Could not create data directory {data_dir}: {e}. Save/load state might not work.")

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nInterrupt signal received. Exiting...")
            break
        except EOFError:
            print("\nEnd of input reached. Exiting...")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]: # Added more exit commands
            print("Exiting. Goodbye!")
            break

        elif user_input.lower().startswith("save state "):
            parts = user_input.split(maxsplit=2)
            if len(parts) == 3:
                filename = parts[2].strip()
                if not filename:
                    print("System: Filename for saving cannot be empty.")
                    continue
                if not filename.endswith(".json"):
                    filename += ".json"
                # Сохраняем состояния в data/ (как и раньше)
                filepath = os.path.join(data_dir, filename)

                if state_mng.save_state_to_file(filepath):
                    print(f"System: State saved to {filepath}")
                else:
                    print(f"System: Failed to save state to {filepath}")
            else:
                print("System: Invalid command format. Use: 'save state <filename>'")
            continue

        elif user_input.lower().startswith("load state "):
            parts = user_input.split(maxsplit=2)
            if len(parts) == 3:
                filename = parts[2].strip()
                if not filename:
                    print("System: Filename for loading cannot be empty.")
                    continue
                if not filename.endswith(".json"):
                    filename += ".json"
                filepath = os.path.join(data_dir, filename)
                if state_mng.load_state_from_file(filepath):
                    print(f"System: State loaded from {filepath}")
                else:
                    print(f"System: Failed to load state from {filepath}.")
            else:
                print("System: Invalid command format. Use: 'load state <filename>'")
            continue

        elif user_input.lower() == "show state":
            current_state = state_mng.get_current_state()
            print("System: Current state:")
            print(json.dumps(current_state, ensure_ascii=False, indent=4))
            continue

        elif user_input.lower() == "reset state":
            state_mng.reset_state()
            print("System: State has been reset.")
            continue

        # Обработка предложения через LogicalController
        response = logic_ctrl.process_user_input(user_input)
        print(f"System: {response}")

if __name__ == "__main__":
    # Пути к моделям для проверки (можно импортировать или определить здесь)
    # NLUController сам знает свои пути, но для предварительной проверки в main.py:
    intent_model_path_main_check = "models/intent_classifier_bert_en"
    slot_model_path_main_check = "models/slot_filler_bert_en"

    if not os.path.exists(intent_model_path_main_check) or not os.path.exists(slot_model_path_main_check):
        print("--- IMPORTANT ---")
        print("NLU models for English not found!")
        print(f"Please ensure trained models exist at '{intent_model_path_main_check}' and '{slot_model_path_main_check}'.")
        print("You need to run the training scripts first:")
        print("1. `python src/training_data_generator.py` (to generate data/synthetic_nlu_data_en.jsonl)")
        print("2. `python src/train_intent_classifier.py`")
        print("3. `python src/train_slot_filler.py`")
        print("The program will attempt to initialize, but NLU functionality will be missing or will fail.")

    main_loop()
