import re
from typing import Dict, Any, Optional

# Предполагается, что эти классы находятся в тех же src директориях
# и могут быть импортированы, если controller.py запускается из корневой директории проекта
# или если src добавлена в PYTHONPATH.
# Для прямого запуска controller.py (if __name__ == '__main__') может потребоваться
# добавить src в sys.path, как показано в блоке if __name__ == '__main__'
try:
    from embedding_generator import EmbeddingGenerator
    from state_manager import StateManager
except ImportError:
    # Это нужно, если мы запускаем controller.py напрямую для тестов,
    # и он не может найти модули в src/.
    # При запуске из main.py, который в корне, импорты должны работать нормально.
    if __name__ == '__main__':
        import sys
        import os
        # Добавляем родительскую директорию (корень проекта), чтобы найти src
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.embedding_generator import EmbeddingGenerator
        from src.state_manager import StateManager
    else:
        raise


class Controller:
    """
    Класс "Контроллер" для обработки предложений, извлечения информации
    и обновления состояния через StateManager.
    """
    def __init__(self, embedding_generator: EmbeddingGenerator, state_manager: StateManager):
        """
        Инициализирует Контроллер.

        Args:
            embedding_generator (EmbeddingGenerator): Экземпляр генератора эмбеддингов.
            state_manager (StateManager): Экземпляр менеджера состояния.
        """
        self.embedding_generator = embedding_generator
        self.state_manager = state_manager
        print("Controller initialized.")

    def _extract_info_simple(self, sentence: str) -> Optional[Dict[str, Any]]:
        """
        Очень упрощенное извлечение информации на основе ключевых слов/регулярных выражений.
        Это временная заглушка.
        Примеры паттернов:
        - "В [место] [число] [объект]" -> add_entity, name, count, location
        - "Добавили [число] [объект]" -> update_entity_count (add)
        - "Убрали [число] [объект]" -> update_entity_count (subtract)
        - "Где [объект]?" -> query_location
        - "Сколько [объект]?" -> query_count
        """
        sentence_lower = sentence.lower()

        # Паттерн: "В X Y Z" (В комнате 5 яблок)
        match_place_count_object = re.search(r"в (.+) (\d+) (.+)", sentence_lower)
        if match_place_count_object:
            location, count, entity_name = match_place_count_object.groups()
            # Убираем возможные знаки препинания в конце имени сущности
            entity_name = re.sub(r"[.,!?]$", "", entity_name).strip()
            # Удаляем возможные предлоги или артикли перед названием места, если они попали
            location = location.replace("на ", "").replace("под ", "").strip()
            return {
                "action": "add_entity_at_location",
                "entity_name": entity_name,
                "count": int(count),
                "location": location
            }

        # Паттерн: "Добавили X Y" (Добавили 3 яблока)
        match_add_count_object = re.search(r"(добавь|добавили|положили) (\d+) (.+)", sentence_lower)
        if match_add_count_object:
            _, count, entity_name = match_add_count_object.groups()
            entity_name = re.sub(r"[.,!?]$", "", entity_name).strip()
            return {
                "action": "update_entity_count",
                "entity_name": entity_name,
                "change_count": int(count) # Положительное число для добавления
            }

        # Паттерн: "Убрали X Y" (Убрали 2 яблока)
        match_remove_count_object = re.search(r"(убери|убрали|взяли) (\d+) (.+)", sentence_lower)
        if match_remove_count_object:
            _, count, entity_name = match_remove_count_object.groups()
            entity_name = re.sub(r"[.,!?]$", "", entity_name).strip()
            return {
                "action": "update_entity_count",
                "entity_name": entity_name,
                "change_count": -int(count) # Отрицательное число для вычитания
            }

        # Паттерн: "Где X?"
        match_where_is_object = re.search(r"где (.+)\?", sentence_lower)
        if match_where_is_object:
            entity_name = match_where_is_object.group(1).strip()
            entity_name = re.sub(r"[.,!?]$", "", entity_name).strip()
            return {"action": "query_location", "entity_name": entity_name}

        # Паттерн: "Сколько X?"
        match_how_many_object = re.search(r"сколько (.+)\?", sentence_lower)
        if match_how_many_object:
            entity_name = match_how_many_object.group(1).strip()
            # Убираем возможные знаки препинания в конце имени сущности
            entity_name = re.sub(r"[.,!?]$", "", entity_name).strip()
            # Для простоты не будем агрессивно менять окончания, положимся на точное совпадение
            # или на то, что пользователь использует ту же форму, что и при создании
            return {"action": "query_count", "entity_name": entity_name}

        # --- Первые шаги к "думанию" и обработке неполных команд ---

        # Паттерн: "Положи X" (но не сказано куда)
        # Должен идти после более специфичных паттернов типа "положили X Y" (где Y - количество)
        match_put_object_no_location = re.search(r"^(положи|клади)\s+(.+)$", sentence_lower)
        if match_put_object_no_location:
            entity_name = match_put_object_no_location.group(2).strip()
            entity_name = re.sub(r"[.,!?]$", "", entity_name).strip()
            return {
                "action": "clarification_needed",
                "entity_name": entity_name,
                "missing_info": "location_for_put",
                "original_intent": "put_entity"
            }

        # Паттерн: "Возьми X" / "Убери X" (без указания количества, подразумевается 1)
        match_take_object_no_count = re.search(r"^(возьми|забери|убери)\s+(.+)$", sentence_lower)
        if match_take_object_no_count:
            entity_name = match_take_object_no_count.group(2).strip()
            entity_name = re.sub(r"[.,!?]$", "", entity_name).strip()

            entity_data = self.state_manager.get_entity(entity_name)
            if entity_data:
                # Если такой объект есть, предполагаем, что нужно убрать/взять 1
                return {
                    "action": "update_entity_count",
                    "entity_name": entity_name,
                    "change_count": -1
                }
            else: # Объекта нет
                return {
                    "action": "clarification_needed",
                    "entity_name": entity_name,
                    "missing_info": "entity_not_found_for_take",
                    "original_intent": "take_entity"
                }

        return None # Не удалось распознать команду

    def process_sentence(self, sentence: str) -> str:
        """
        Обрабатывает одно предложение.

        Args:
            sentence (str): Входное предложение.

        Returns:
            str: Ответ системы или статус обработки.
        """
        print(f"\nController processing: '{sentence}'")
        # Шаг 1: Получение эмбеддинга (пока не используется в простой логике, но задел на будущее)
        # embedding = self.embedding_generator.get_embeddings(sentence)
        # if embedding is None:
        #     return "Ошибка: не удалось получить эмбеддинг для предложения."
        # print(f"Embedding shape for sentence: {embedding.shape}")

        # Шаг 2: Извлечение информации (упрощенная версия)
        extracted_info = self._extract_info_simple(sentence)

        if not extracted_info:
            return f"Не удалось распознать команду в предложении: '{sentence}'"

        action = extracted_info.get("action")
        entity_name = extracted_info.get("entity_name")
        response = f"Действие: {action}, Сущность: {entity_name}."

        # Шаг 3: Обновление состояния или выполнение запроса
        if action == "add_entity_at_location":
            attrs = {"count": extracted_info["count"], "location": extracted_info["location"]}
            self.state_manager.add_or_update_entity(entity_name, attrs)
            response = f"Добавлено/обновлено: {entity_name} (количество: {attrs['count']}) в локации '{attrs['location']}'."

        elif action == "update_entity_count":
            current_entity = self.state_manager.get_entity(entity_name)
            change = extracted_info["change_count"]
            if current_entity:
                current_count = current_entity.get("count", 0)
                new_count = current_count + change
                if new_count < 0:
                    response = f"Невозможно выполнить: у {entity_name} всего {current_count}, нельзя убрать {abs(change)}."
                else:
                    self.state_manager.add_or_update_entity(entity_name, {"count": new_count})
                    if change > 0:
                        response = f"К {entity_name} добавлено {change}. Теперь их {new_count}."
                    else:
                        response = f"У {entity_name} убрано {abs(change)}. Теперь их {new_count}."
            else:
                if change > 0: # Если добавляем несуществующую сущность
                    self.state_manager.add_or_update_entity(entity_name, {"count": change})
                    response = f"Добавлена новая сущность: {entity_name} (количество: {change})."
                else: # Пытаемся убрать из несуществующей
                    response = f"Сущность '{entity_name}' не найдена, невозможно изменить количество."

        elif action == "query_location":
            entity_data = self.state_manager.get_entity(entity_name)
            if entity_data and "location" in entity_data:
                response = f"{entity_name.capitalize()} находится в локации '{entity_data['location']}'."
            elif entity_data:
                response = f"Местоположение для '{entity_name}' неизвестно."
            else:
                response = f"Сущность '{entity_name}' не найдена."

        elif action == "query_count":
            entity_data = self.state_manager.get_entity(entity_name)
            if entity_data and "count" in entity_data:
                response = f"Количество {entity_name}: {entity_data['count']}."
            elif entity_data:
                response = f"Количество для '{entity_name}' неизвестно."
            else:
                # Перед тем как сказать "не найдена", попробуем нормализовать запрос как в _extract_info_simple
                # Это очень грубая попытка, в идеале нужна лемматизация при сохранении и запросе
                normalized_entity_name = entity_name
                if normalized_entity_name.endswith("ов") and len(normalized_entity_name) > 2 : normalized_entity_name = normalized_entity_name[:-2]
                elif normalized_entity_name.endswith("ев") and len(normalized_entity_name) > 2 : normalized_entity_name = normalized_entity_name[:-2]
                elif normalized_entity_name.endswith("ей") and len(normalized_entity_name) > 2 : normalized_entity_name = normalized_entity_name[:-2]
                # ... (можно добавить больше правил или использовать лемматизатор)
                # Попробуем еще раз с нормализованным именем, если оно изменилось
                if normalized_entity_name != entity_name:
                    entity_data_normalized = self.state_manager.get_entity(normalized_entity_name)
                    if entity_data_normalized and "count" in entity_data_normalized:
                        response = f"Количество {normalized_entity_name}: {entity_data_normalized['count']}."
                        entity_name = normalized_entity_name # Обновляем для лога
                    elif entity_data_normalized:
                        response = f"Количество для '{normalized_entity_name}' неизвестно."
                        entity_name = normalized_entity_name # Обновляем для лога
                    else:
                        response = f"Сущность '{entity_name}' (или '{normalized_entity_name}') не найдена."
                else:
                    response = f"Сущность '{entity_name}' не найдена."

        elif action == "clarification_needed":
            missing = extracted_info.get("missing_info")
            entity = extracted_info.get("entity_name")
            # original_intent = extracted_info.get("original_intent") # Пока не используется в ответе

            if missing == "location_for_put":
                response = f"Куда вы хотите положить '{entity}'?"
            elif missing == "entity_not_found_for_take":
                response = f"Я не могу взять '{entity}', так как такой сущности нет."
            # Добавить другие типы уточнений если будут
            else:
                response = f"Мне нужно больше информации для обработки запроса по '{entity}'."

        else:
            response = f"Неизвестное действие: {action}"

        print(f"Controller response: {response}")
        return response

if __name__ == '__main__':
    print("Initializing components for Controller test...")
    try:
        # Убедимся, что src в sys.path для импортов, если запускаем этот файл напрямую
        if 'src.embedding_generator' not in sys.modules:
             import os
             # Добавляем родительскую директорию (корень проекта), чтобы найти src
             sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
             from src.embedding_generator import EmbeddingGenerator
             from src.state_manager import StateManager

        emb_gen = EmbeddingGenerator() # Использует модель по умолчанию
        state_mng = StateManager()
        controller = Controller(emb_gen, state_mng)

        print("\n--- Testing Controller Processing ---")

        test_sentences = [
            "В комнате 5 яблок.",
            "Добавили 3 яблока.",
            "Сколько яблок?",
            "Где яблоки?",
            "Убрали 10 яблок.", # Попытка убрать больше, чем есть
            "Сколько яблок?",
            "Убрали 2 яблока.",
            "Сколько яблок?",
            "Где груши?", # Запрос о несуществующей сущности
            "Положили 2 банана.",
            "Сколько бананов?",
            "Это непонятное предложение."
        ]

        for sentence in test_sentences:
            controller.process_sentence(sentence)
            # print("Current state:", state_mng.get_current_state()["entities"])
            print("-" * 30)

        print("\n--- Testing Clarification Logic ---")
        clarification_sentences = [
            "Положи яблоко.",
            "Возьми грушу.", # Груши нет в состоянии
            "Убери стол." # Стол не является сущностью с количеством
        ]
        for sentence in clarification_sentences:
            controller.process_sentence(sentence)
            print("-" * 30)

        # Проверим, что после "Возьми яблоко" (если яблоки есть), одно яблоко уберется
        state_mng.reset_state()
        controller.process_sentence("В комнате 3 яблока.")
        print("Состояние до 'Возьми яблоко':", state_mng.get_entity("яблоко"))
        controller.process_sentence("Возьми яблоко.")
        print("Состояние после 'Возьми яблоко':", state_mng.get_entity("яблоко"))
        controller.process_sentence("Сколько яблок?")


        print("\nController test finished.")

    except Exception as e:
        import traceback
        print(f"An error occurred during the Controller test: {e}")
        print(traceback.format_exc())
        print("Ensure EmbeddingGenerator and StateManager can be initialized.")
        print("If SentenceTransformer model download fails, check internet connection.")
