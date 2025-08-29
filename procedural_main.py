from src.logical_controller.procedural_controller import ProceduralController

def get_commands_from_neural_network_stub(text_input: str) -> list:
    """
    Это заглушка, имитирующая работу нейросети (НС2),
    которая преобразует текст в процедурные команды.
    """
    print(f"--- Нейросеть-заглушка получила текст: '{text_input}' ---")
    print("--- Генерирую процедурные команды... ---\n")

    # Логика заглушки: в зависимости от входного текста,
    # возвращаем разные наборы команд.

    text_lower = text_input.lower()

    if "у меня 2 яблока" in text_lower and "у алины 3" in text_lower:
        return [
            "CREATE person user_me WITH name='Я'",
            "CREATE item apple_my_1 WITH quantity=2",
            "LINK user_me owns apple_my_1",
            "CREATE person user_alice WITH name='Алина'",
            "CREATE item apple_alice_1 WITH quantity=3",
            "LINK user_alice owns apple_alice_1",
            "QUERY item COMPUTE SUM(quantity)" # "Сколько всего яблок?"
        ]
    elif "отдал одно яблоко" in text_lower:
        # Предполагаем, что яблоки уже есть в графе из предыдущего шага
        return [
            "UPDATE apple_my_1 SET quantity=PREVIOUS_VALUE-1",
            "QUERY item WHERE owner='user_me' COMPUTE SUM(quantity)" # "Сколько у меня осталось?"
        ]
    else:
        return ["CREATE log entry_1 WITH status='Неизвестный запрос'"]


def main():
    # --- Сценарий 1: Первоначальное создание объектов ---
    print("--- Сценарий 1: Создание объектов ---")

    # 1. Инициализируем контроллер. Он будет хранить состояние.
    controller = ProceduralController()

    # 2. Входное предложение от пользователя
    input_1 = "У меня 2 яблока, а у Алины 3. Сколько всего яблок?"

    # 3. Нейросеть генерирует команды
    commands_1 = get_commands_from_neural_network_stub(input_1)

    # 4. Контроллер выполняет команды
    for cmd in commands_1:
        print(f"Executing: {cmd}")
        controller.execute(cmd)

    controller.print_summary()

    # --- Сценарий 2: Обновление состояния ---
    print("\n\n--- Сценарий 2: Обновление состояния ---")
    print("--- (Используем тот же контроллер, который помнит предыдущее состояние) ---\n")

    # 1. Новое предложение от пользователя
    input_2 = "Я отдал одно яблоко. Сколько у меня осталось?"

    # 2. Нейросеть генерирует новые команды
    commands_2 = get_commands_from_neural_network_stub(input_2)

    # 3. Контроллер выполняет новые команды, изменяя существующий граф
    for cmd in commands_2:
        print(f"Executing: {cmd}")
        controller.execute(cmd)

    controller.print_summary()


if __name__ == "__main__":
    main()
