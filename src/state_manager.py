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
        self.state: Dict[str, Any] = self._get_initial_state()
        self._entity_id_counter = 0
        self._snapshots: Dict[str, Dict[str, Any]] = {} # Для хранения снимков состояния

    def _get_initial_state(self) -> Dict[str, Any]:
        """Возвращает начальную структуру состояния."""
        return {
            "entities": {},  # name: {attr1: value1, attr2: value2}
            "facts": [],     # list of strings or structured facts
            "active_goals": [], # Список активных целей или недостающей информации
                                # Например: [{"goal_type": "QUERY_BUSINESS_PLAN", "status": "pending_clarification", "missing_slots": ["budget"]}]
            "dialog_history": [] # Можно добавить историю для контекста
        }

    def _generate_entity_id(self, entity_name: str) -> str:
        return entity_name

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
        self.state = self._get_initial_state()
        self._entity_id_counter = 0
        # Снимки НЕ сбрасываются при reset_state, если это не требуется явно.
        # Если нужно очищать и снимки, добавить self._snapshots = {}
        print("State: Manager reset to initial (active) state.")

    # --- Управление целями/недостающей информацией ---
    def add_goal(self, goal_description: Dict[str, Any]) -> None:
        """Добавляет новую цель или элемент недостающей информации."""
        if "active_goals" not in self.state: # на всякий случай
            self.state["active_goals"] = []
        self.state["active_goals"].append(goal_description)
        print(f"State: Goal added: {goal_description}")

    def get_active_goals(self) -> List[Dict[str, Any]]:
        """Возвращает список активных целей."""
        return self.state.get("active_goals", [])

    def update_goal_status(self, goal_id_or_type: str, new_status: str, criteria: str = "goal_type") -> bool:
        """Обновляет статус цели (находит по goal_type или другому критерию)."""
        updated = False
        for goal in self.state.get("active_goals", []):
            if goal.get(criteria) == goal_id_or_type:
                goal["status"] = new_status
                print(f"State: Goal '{goal_id_or_type}' status updated to '{new_status}'")
                updated = True
                # break # если предполагается уникальность по критерию
        return updated

    def remove_goal(self, goal_id_or_type: str, criteria: str = "goal_type") -> bool:
        """Удаляет цель из списка."""
        initial_len = len(self.state.get("active_goals", []))
        self.state["active_goals"] = [
            goal for goal in self.state.get("active_goals", []) if goal.get(criteria) != goal_id_or_type
        ]
        removed = len(self.state.get("active_goals", [])) < initial_len
        if removed:
            print(f"State: Goal '{goal_id_or_type}' (by {criteria}) removed.")
        return removed

    # --- Снимки состояния (Snapshots) для "дерева думания" ---
    def create_snapshot(self, snapshot_id: str) -> bool:
        """Создает снимок (копию) текущего активного состояния."""
        if snapshot_id in self._snapshots:
            print(f"State: Snapshot ID '{snapshot_id}' already exists. Overwriting.")
        # Глубокое копирование, чтобы изменения в активном состоянии не влияли на снимок
        # и наоборот, после восстановления. json.loads(json.dumps()) - простой способ.
        try:
            self._snapshots[snapshot_id] = json.loads(json.dumps(self.state))
            print(f"State: Snapshot '{snapshot_id}' created.")
            return True
        except Exception as e:
            print(f"State: Error creating snapshot '{snapshot_id}': {e}")
            return False

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Восстанавливает активное состояние из снимка."""
        if snapshot_id in self._snapshots:
            try:
                self.state = json.loads(json.dumps(self._snapshots[snapshot_id]))
                print(f"State: Snapshot '{snapshot_id}' restored to active state.")
                return True
            except Exception as e:
                print(f"State: Error restoring snapshot '{snapshot_id}': {e}")
                return False
        else:
            print(f"State: Snapshot ID '{snapshot_id}' not found.")
            return False

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Удаляет снимок."""
        if snapshot_id in self._snapshots:
            del self._snapshots[snapshot_id]
            print(f"State: Snapshot '{snapshot_id}' deleted.")
            return True
        print(f"State: Snapshot ID '{snapshot_id}' not found for deletion.")
        return False

    def list_snapshots(self) -> List[str]:
        """Возвращает список ID существующих снимков."""
        return list(self._snapshots.keys())

    # --- Сохранение и загрузка состояния (включая снимки) ---
    def save_state_to_file(self, filepath: str) -> bool:
        """
        Сохраняет текущее состояние в JSON-файл.

        Args:
            filepath (str): Путь к файлу для сохранения.

        Returns:
            bool: True в случае успеха, False в противном случае.
        """
        try:
            # Сохраняем и активное состояние, и все снимки
            full_dump = {
                "active_state": self.state,
                "snapshots": self._snapshots
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(full_dump, f, ensure_ascii=False, indent=4)
            print(f"State: Successfully saved (active state and snapshots) to {filepath}")
            return True
        except Exception as e: # Более общее исключение для json.dumps ошибок
            print(f"State: Error saving full state to {filepath}: {e}")
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
                full_load = json.load(f)

            if "active_state" in full_load and "snapshots" in full_load:
                # Проверка структуры активного состояния (минимальная)
                if "entities" in full_load["active_state"] and \
                   "facts" in full_load["active_state"]:
                    self.state = full_load["active_state"]
                    self._snapshots = full_load["snapshots"]
                    # Сбрасываем счетчик, если он не является частью сохраняемого состояния
                    # self._entity_id_counter = 0
                    print(f"State: Successfully loaded (active state and snapshots) from {filepath}")
                    return True
                else:
                    print(f"State: Invalid 'active_state' format in {filepath}")
                    return False
            elif "entities" in full_load and "facts" in full_load: # Обратная совместимость со старым форматом
                print("State: Loading old format (only active state, no snapshots).")
                self.state = full_load
                self._snapshots = {} # Очищаем снимки, если грузим старый формат
                print(f"State: Successfully loaded old format from {filepath}")
                return True
            else:
                print(f"State: Invalid full state format in {filepath}")
                return False
        except FileNotFoundError:
            print(f"State: File not found {filepath}. Initializing with empty state.")
            self.reset_state()
            self._snapshots = {} # Также очищаем снимки
            return False
        except json.JSONDecodeError as e:
            print(f"State: Error decoding JSON from {filepath}: {e}")
            return False
        except Exception as e: # Более общее исключение
            print(f"State: Error loading full state from {filepath}: {e}")
            return False

if __name__ == '__main__':
    print("Initializing StateManager...")
    manager = StateManager()

    print("\n--- Testing Entity Management (English example) ---")
    manager.add_or_update_entity("apple", {"count": 5, "color": "red"})
    manager.add_or_update_entity("banana", {"count": 3, "color": "yellow"})
    print("Current entities:", manager.get_all_entities())

    manager.add_or_update_entity("apple", {"count": 10, "location": "table"}) # Update existing
    print("Apple attributes:", manager.get_entity("apple"))

    manager.remove_entity("banana")
    print("Entities after removing banana:", manager.get_all_entities())
    manager.remove_entity("pear") # Try to remove non-existent

    print("\n--- Testing Fact Management ---")
    manager.add_fact("The sun is shining.")
    manager.add_fact({"event": "rain", "location": "city", "intensity": "light"})
    print("Current facts:", manager.get_facts())
    manager.clear_facts()
    print("Facts after clearing:", manager.get_facts())

    print("\n--- Testing Goal Management ---")
    manager.add_goal({"goal_type": "FIND_ITEM", "item_name": "keys", "status": "pending"})
    manager.add_goal({"goal_type": "PLAN_TRIP", "destination": "Paris", "status": "needs_dates"})
    print("Active goals:", manager.get_active_goals())
    manager.update_goal_status("FIND_ITEM", "in_progress")
    manager.remove_goal("PLAN_TRIP")
    print("Active goals after updates:", manager.get_active_goals())


    print("\n--- Testing State Reset ---")
    manager.add_or_update_entity("car", {"color": "blue"})
    print("State before reset:", manager.get_current_state())
    manager.reset_state() # Resets active_state, not _snapshots by default
    print("State after reset:", manager.get_current_state())

    print("\n--- Testing Snapshots ---")
    manager.add_or_update_entity("book", {"title": "AI Adventures", "pages": 300})
    manager.add_fact("The book is interesting.")
    manager.create_snapshot("snapshot1")
    print("Snapshots list:", manager.list_snapshots())

    manager.add_or_update_entity("pen", {"color": "black"})
    manager.add_fact("The pen is new.")
    print("Active state before restore:", manager.get_current_state())
    manager.restore_snapshot("snapshot1")
    print("Active state after restoring snapshot1:", manager.get_current_state())
    assert "pen" not in manager.get_all_entities() # Pen should be gone

    manager.create_snapshot("snapshot2_after_restore") # snapshot of the restored state
    manager.delete_snapshot("snapshot1")
    print("Snapshots list after delete:", manager.list_snapshots())


    print("\n--- Testing Save/Load State (with snapshots) ---")
    manager.reset_state() # Start fresh for save/load test
    manager._snapshots = {} # Clear snapshots for this specific test

    manager.add_or_update_entity("computer", {"type": "laptop"})
    manager.create_snapshot("snap_A")
    manager.add_or_update_entity("mouse", {"type": "wireless"})

    # Use a path in the data_dir, consistent with main.py
    data_dir_test = "data"
    if not os.path.exists(data_dir_test):
        os.makedirs(data_dir_test, exist_ok=True)
    save_path = os.path.join(data_dir_test, "test_full_state.json")

    print(f"Saving full state to {save_path}...")
    manager.save_state_to_file(save_path)

    print("Resetting current manager state and snapshots...")
    manager.reset_state()
    manager._snapshots = {} # Explicitly clear snapshots for testing load
    print("State after reset (before load):", manager.get_current_state())
    print("Snapshots after reset (before load):", manager.list_snapshots())

    print(f"Loading full state from {save_path}...")
    manager.load_state_from_file(save_path)
    print("State after load:", manager.get_current_state())
    print("Snapshots after load:", manager.list_snapshots())
    assert "computer" in manager.get_all_entities()
    assert "mouse" in manager.get_all_entities()
    assert "snap_A" in manager.list_snapshots()

    # Test loading old format
    old_format_save_path = os.path.join(data_dir_test, "test_old_format_state.json")
    with open(old_format_save_path, 'w', encoding='utf-8') as f:
        json.dump({"entities": {"test_entity": {"val":1}}, "facts":["test_fact"]}, f)
    manager.load_state_from_file(old_format_save_path)
    print("State after loading old format:", manager.get_current_state())
    assert "test_entity" in manager.get_all_entities()
    assert not manager.list_snapshots() # Snapshots should be cleared when loading old format

    print("\nStateManager script finished.")
