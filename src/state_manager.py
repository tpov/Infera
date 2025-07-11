import json
from typing import Any, Dict, List, Optional

class StateManager:
    """
    Класс для управления состоянием (сущностями, их атрибутами и фактами).
    Представляет "память" системы или "автоматы", хранящие контекст.
    """
    def __init__(self):
        """
        Инициализирует StateManager с пустым состоянием.
        Состояние хранится в словаре `self.state`, который может включать:
        - 'entities': словарь сущностей, где каждая сущность имеет свои атрибуты.
        - 'facts': список фактов или отношений.
        """
        self.state: Dict[str, Any] = {
            "entities": {},  # name: {attr1: value1, attr2: value2}
            "facts": []      # list of strings or structured facts
        }
        self._entity_id_counter = 0 # Для генерации уникальных ID, если имена не уникальны

    def _generate_entity_id(self, entity_name: str) -> str:
        """Генерирует уникальный ID для сущности, если это необходимо."""
        # Пока используем имя как ID, но можно расширить для уникальности
        # self._entity_id_counter += 1
        # return f"{entity_name}_{self._entity_id_counter}"
        return entity_name # Для простоты пока имя = ID

    # --- Методы для работы с сущностями ---
    def add_or_update_entity(self, entity_name: str, attributes: Dict[str, Any]) -> None:
        """
        Добавляет новую сущность или обновляет атрибуты существующей.
        Если сущность с таким именем не существует, она создается.
        Если существует, ее атрибуты обновляются/дополняются.

        Args:
            entity_name (str): Имя (идентификатор) сущности.
            attributes (Dict[str, Any]): Словарь атрибутов для сущности.
        """
        entity_id = self._generate_entity_id(entity_name) # Пока entity_name и есть ID
        if entity_id not in self.state["entities"]:
            self.state["entities"][entity_id] = {}

        for attr_name, attr_value in attributes.items():
            # Если атрибут - это число и он уже существует, можно реализовать логику сложения
            # Например, для 'count'
            if isinstance(attr_value, (int, float)) and \
               attr_name in self.state["entities"][entity_id] and \
               isinstance(self.state["entities"][entity_id][attr_name], (int, float)):
                # Пример: если добавляем количество к существующему количеству
                # Пока просто перезаписываем, логику сложения/изменения можно добавить в контроллере
                self.state["entities"][entity_id][attr_name] = attr_value
            else:
                self.state["entities"][entity_id][attr_name] = attr_value
        print(f"State: Entity '{entity_id}' updated/added with attributes: {attributes}")

    def get_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает атрибуты сущности по ее имени.

        Args:
            entity_name (str): Имя сущности.

        Returns:
            Optional[Dict[str, Any]]: Словарь атрибутов сущности или None, если не найдена.
        """
        return self.state["entities"].get(entity_name)

    def get_all_entities(self) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает все сущности и их атрибуты.
        """
        return self.state["entities"]

    def remove_entity(self, entity_name: str) -> bool:
        """
        Удаляет сущность из состояния.

        Args:
            entity_name (str): Имя удаляемой сущности.

        Returns:
            bool: True, если сущность была удалена, False в противном случае.
        """
        if entity_name in self.state["entities"]:
            del self.state["entities"][entity_name]
            print(f"State: Entity '{entity_name}' removed.")
            return True
        print(f"State: Entity '{entity_name}' not found for removal.")
        return False

    # --- Методы для работы с фактами ---
    def add_fact(self, fact: str | Dict[str, Any]) -> None:
        """
        Добавляет факт в список фактов.

        Args:
            fact (str | Dict[str, Any]): Описание факта (строка или структурированный словарь).
        """
        self.state["facts"].append(fact)
        print(f"State: Fact added: {fact}")

    def get_facts(self) -> List[str | Dict[str, Any]]:
        """
        Возвращает список всех фактов.
        """
        return self.state["facts"]

    def clear_facts(self) -> None:
        """
        Очищает список фактов.
        """
        self.state["facts"] = []
        print("State: All facts cleared.")

    # --- Общие методы ---
    def get_current_state(self) -> Dict[str, Any]:
        """
        Возвращает полное текущее состояние.
        """
        return self.state

    def reset_state(self) -> None:
        """
        Сбрасывает состояние к начальному (пустому).
        """
        self.state = {
            "entities": {},
            "facts": []
        }
        self._entity_id_counter = 0
        print("State: Manager reset to initial state.")

    # --- Сохранение и загрузка состояния ---
    def save_state_to_file(self, filepath: str) -> bool:
        """
        Сохраняет текущее состояние в JSON-файл.

        Args:
            filepath (str): Путь к файлу для сохранения.

        Returns:
            bool: True в случае успеха, False в противном случае.
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=4)
            print(f"State: Successfully saved to {filepath}")
            return True
        except IOError as e:
            print(f"State: Error saving state to {filepath}: {e}")
            return False

    def load_state_from_file(self, filepath: str) -> bool:
        """
        Загружает состояние из JSON-файла.

        Args:
            filepath (str): Путь к файлу для загрузки.

        Returns:
            bool: True в случае успеха, False в противном случае.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_state = json.load(f)

            # Простая валидация структуры загруженного состояния
            if "entities" in loaded_state and "facts" in loaded_state:
                self.state = loaded_state
                # Можно добавить обновление счетчика ID, если он используется более сложно
                print(f"State: Successfully loaded from {filepath}")
                return True
            else:
                print(f"State: Invalid state format in {filepath}")
                return False
        except FileNotFoundError:
            print(f"State: File not found {filepath}. Initializing with empty state.")
            self.reset_state() # Или можно просто вернуть False и не сбрасывать текущее
            return False
        except json.JSONDecodeError as e:
            print(f"State: Error decoding JSON from {filepath}: {e}")
            return False
        except IOError as e:
            print(f"State: Error loading state from {filepath}: {e}")
            return False

if __name__ == '__main__':
    print("Initializing StateManager...")
    manager = StateManager()

    print("\n--- Testing Entity Management ---")
    manager.add_or_update_entity("яблоко", {"count": 5, "color": "красный"})
    manager.add_or_update_entity("банан", {"count": 3, "color": "желтый"})
    print("Текущие сущности:", manager.get_all_entities())

    manager.add_or_update_entity("яблоко", {"count": 10, "location": "стол"}) # Обновление существующей
    print("Атрибуты яблока:", manager.get_entity("яблоко"))

    manager.remove_entity("банан")
    print("Сущности после удаления банана:", manager.get_all_entities())
    manager.remove_entity("груша") # Попытка удалить несуществующую

    print("\n--- Testing Fact Management ---")
    manager.add_fact("Солнце светит.")
    manager.add_fact({"event": "дождь", "location": "город", "intensity": "слабый"})
    print("Текущие факты:", manager.get_facts())
    manager.clear_facts()
    print("Факты после очистки:", manager.get_facts())

    print("\n--- Testing State Reset ---")
    manager.add_or_update_entity("машина", {"color": "синий"})
    print("Состояние до сброса:", manager.get_current_state())
    manager.reset_state()
    print("Состояние после сброса:", manager.get_current_state())

    print("\n--- Testing Save/Load State ---")
    manager.add_or_update_entity("книга", {"title": "AI Adventures", "pages": 300})
    manager.add_fact("Книга интересная.")
    save_path = "test_state.json"

    print(f"Сохранение состояния в {save_path}...")
    manager.save_state_to_file(save_path)

    print("Сброс текущего состояния менеджера...")
    manager.reset_state()
    print("Состояние после сброса (перед загрузкой):", manager.get_current_state())

    print(f"Загрузка состояния из {save_path}...")
    manager.load_state_from_file(save_path)
    print("Состояние после загрузки:", manager.get_current_state())

    print("\nПроверка загрузки несуществующего файла:")
    manager.load_state_from_file("non_existent_state.json")
    print("Состояние после попытки загрузки несуществующего файла:", manager.get_current_state())


    print("\nStateManager script finished.")
