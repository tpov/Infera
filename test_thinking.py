from thinking_pipeline import ThinkingPipeline

def test_thinking_mode():
    """
    Тестирует режим "думания" с простыми данными
    """
    print("Тестирование режима 'думания'...")
    
    # Создаем пайплайн с уменьшенным количеством циклов для теста
    pipeline = ThinkingPipeline(max_thinking_cycles=3)
    
    # Тестовый запрос
    test_query = "Создай робота с возрастом -5"
    
    print(f"Запрос: {test_query}")
    print("Запускаем режим 'думания'...")
    
    try:
        result = pipeline.process_with_thinking(test_query)
        
        print(f"\nРезультаты:")
        print(f"Успех: {result['success']}")
        print(f"Количество циклов: {len(result['thinking_cycles'])}")
        print(f"Финальный ответ: {result['final_response']}")
        
        if result['error']:
            print(f"Ошибка: {result['error']}")
        
        # Показываем детали каждого цикла
        for i, cycle in enumerate(result['thinking_cycles'], 1):
            print(f"\nЦикл {i}:")
            print(f"  Вход: {cycle['input'][:50]}...")
            print(f"  Команды: {cycle['commands']}")
            print(f"  Ответ: {cycle['response']}")
        
        # Показываем статус системы
        status = pipeline.get_system_status()
        print(f"\nОбъектов в системе: {status['total_objects']}")
        
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")

if __name__ == "__main__":
    test_thinking_mode() 