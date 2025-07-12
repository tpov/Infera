from command_controller import CommandController

def test_question_objects():
    controller = CommandController()
    
    # Создаем объекты с проблемами
    test_commands = """
    create robot age -5 cost 10, 
    create it1 name it1 age 25, 
    create sensor position "distance 150", 
    create device quantity 0, 
    create component exists false
    """
    
    print("Тестируем создание объектов question и it...")
    result = controller.execute_commands(test_commands)
    
    print(f"\nРезультат выполнения:")
    print(f"Успех: {result.success}")
    print(f"Сообщение: {result.message}")
    print(f"Создано объектов: {len(result.created_objects)}")
    
    print("\nСозданные объекты:")
    for obj in result.created_objects:
        if 'problem' in obj.properties:
            print(f"  {obj.name}: {obj.properties['problem'].value}")
        else:
            print(f"  {obj.name}: Нет описания проблемы")
    
    print("\nВсе объекты в системе:")
    for obj_id, obj in controller.system_state.objects.items():
        print(f"  {obj.name} (ID: {obj_id})")
        if 'problem' in obj.properties:
            print(f"    Проблема: {obj.properties['problem'].value}")

if __name__ == "__main__":
    test_question_objects() 