import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import json
import re
from vectorizer import get_vector
from enhanced_controller import EnhancedCommandController

@dataclass
class LifeStateObject:
    """Объект состояния с временной меткой и связями"""
    id: str
    entity_type: str  # person, resource, state, action, transaction, etc.
    properties: Dict[str, Any]
    timestamp: datetime
    duration: Optional[timedelta] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    time_chain: List[str] = field(default_factory=list)

@dataclass
class LifeActionTemplate:
    """Шаблон для создания связанных объектов жизненных действий"""
    triggers: List[str]  # Ключевые слова, которые активируют шаблон
    creates: List[Dict[str, Any]]  # Что создается
    updates: List[Dict[str, Any]]  # Что обновляется
    time_flow: Dict[str, Any]  # Как течет время
    uncertainty_patterns: List[str] = field(default_factory=list)  # Паттерны неопределенностей

class UnifiedLifeSystem:
    """
    Единая система жизненных команд, интегрированная с Infera
    """
    
    def __init__(self):
        self.objects = {}  # Все объекты системы
        self.templates = self._init_life_templates()
        self.current_time = datetime.now()
        self.context = {
            "current_user": None,
            "active_entities": [],
            "location": None,
            "situation": None
        }
        self.controller = EnhancedCommandController()
        
    def _init_life_templates(self) -> Dict[str, LifeActionTemplate]:
        """Инициализация шаблонов для разных типов жизненных действий"""
        return {
            # === БАЗОВЫЕ ФИЗИОЛОГИЧЕСКИЕ ДЕЙСТВИЯ ===
            "eat": LifeActionTemplate(
                triggers=["съел", "поел", "покушал", "ем", "еда", "кушаю"],
                creates=[
                    {"type": "state", "name": "голодный", "value": False, "time_offset": 0},
                    {"type": "state", "name": "сытый", "value": True, "time_offset": 0},
                    {"type": "resource", "name": "еда", "change": -1, "time_offset": 0},
                    {"type": "action", "name": "прием_пищи", "duration": "30min", "time_offset": 0}
                ],
                updates=[
                    {"target": "person", "property": "hunger", "value": 0},
                    {"target": "person", "property": "energy", "change": "+20"}
                ],
                time_flow={"next_hunger": "4hours", "digestion": "2hours"},
                uncertainty_patterns=["что съел?", "сколько съел?", "когда съел?"]
            ),
            
            "sleep": LifeActionTemplate(
                triggers=["спал", "сплю", "лег спать", "сон", "отдых"],
                creates=[
                    {"type": "state", "name": "уставший", "value": False, "time_offset": 0},
                    {"type": "state", "name": "отдохнувший", "value": True, "time_offset": "8hours"},
                    {"type": "action", "name": "сон", "duration": "8hours", "time_offset": 0}
                ],
                updates=[
                    {"target": "person", "property": "tiredness", "value": 0, "time_offset": "8hours"},
                    {"target": "person", "property": "energy", "value": 100, "time_offset": "8hours"}
                ],
                time_flow={"next_sleep": "16hours"},
                uncertainty_patterns=["сколько спал?", "когда лег?", "как спал?"]
            ),
            
            # === ФИНАНСОВЫЕ ДЕЙСТВИЯ ===
            "earn_money": LifeActionTemplate(
                triggers=["заработал", "получил зарплату", "доход", "прибыль", "зарплата"],
                creates=[
                    {"type": "transaction", "name": "доход", "amount": "$amount", "time_offset": 0},
                    {"type": "state", "name": "финансовое_состояние", "value": "improved", "time_offset": 0}
                ],
                updates=[
                    {"target": "person", "property": "money", "change": "+$amount"},
                    {"target": "person", "property": "financial_stress", "change": "-10"}
                ],
                time_flow={"next_salary": "1month"},
                uncertainty_patterns=["сколько заработал?", "откуда доход?", "когда получил?"]
            ),
            
            "spend_money": LifeActionTemplate(
                triggers=["потратил", "купил", "заплатил", "расход", "покупка"],
                creates=[
                    {"type": "transaction", "name": "расход", "amount": "$amount", "time_offset": 0},
                    {"type": "resource", "name": "$item", "change": "+1", "time_offset": 0},
                    {"type": "state", "name": "финансовое_состояние", "value": "decreased", "time_offset": 0}
                ],
                updates=[
                    {"target": "person", "property": "money", "change": "-$amount"},
                    {"target": "person", "property": "possessions", "add": "$item"}
                ],
                time_flow={},
                uncertainty_patterns=["сколько потратил?", "что купил?", "зачем купил?"]
            ),
            
            # === СОЦИАЛЬНЫЕ ДЕЙСТВИЯ ===
            "meet_person": LifeActionTemplate(
                triggers=["встретил", "увидел", "познакомился", "встреча", "общение"],
                creates=[
                    {"type": "person", "name": "$person", "time_offset": 0},
                    {"type": "relationship", "between": ["я", "$person"], "status": "acquaintance", "time_offset": 0},
                    {"type": "action", "name": "встреча", "duration": "$duration", "time_offset": 0}
                ],
                updates=[
                    {"target": "person", "property": "social_connections", "add": "$person"},
                    {"target": "person", "property": "social_energy", "change": "+5"}
                ],
                time_flow={"relationship_development": "ongoing"},
                uncertainty_patterns=["кто это?", "где встретил?", "как долго общались?"]
            ),
            
            # === РАБОЧИЕ ДЕЙСТВИЯ ===
            "work": LifeActionTemplate(
                triggers=["работаю", "делаю проект", "задача", "работа", "труд"],
                creates=[
                    {"type": "action", "name": "работа", "duration": "$duration", "time_offset": 0},
                    {"type": "state", "name": "занятый", "value": True, "time_offset": 0},
                    {"type": "progress", "name": "$task", "value": "$progress", "time_offset": 0}
                ],
                updates=[
                    {"target": "person", "property": "productivity", "change": "+$progress"},
                    {"target": "person", "property": "tiredness", "change": "+$duration/2"}
                ],
                time_flow={"work_day": "8hours", "break_needed": "2hours"},
                uncertainty_patterns=["над чем работаешь?", "сколько времени?", "какой прогресс?"]
            ),
            
            # === ОБУЧЕНИЕ ===
            "learn": LifeActionTemplate(
                triggers=["изучаю", "учусь", "читаю", "курс", "обучение", "знания"],
                creates=[
                    {"type": "action", "name": "обучение", "subject": "$subject", "duration": "$duration", "time_offset": 0},
                    {"type": "knowledge", "name": "$subject", "level": "$level", "time_offset": 0},
                    {"type": "progress", "name": "обучение_$subject", "value": "$progress", "time_offset": 0}
                ],
                updates=[
                    {"target": "person", "property": "skills", "add": "$subject"},
                    {"target": "person", "property": "knowledge_level", "change": "+$progress"}
                ],
                time_flow={"skill_retention": "30days", "practice_needed": "daily"},
                uncertainty_patterns=["что изучаешь?", "сколько времени?", "какой уровень?"]
            ),
            
            # === ЗДОРОВЬЕ И СПОРТ ===
            "exercise": LifeActionTemplate(
                triggers=["тренируюсь", "спорт", "бегаю", "зал", "фитнес", "упражнения"],
                creates=[
                    {"type": "action", "name": "тренировка", "type": "$exercise_type", "duration": "$duration", "time_offset": 0},
                    {"type": "state", "name": "активный", "value": True, "time_offset": 0},
                    {"type": "progress", "name": "физическая_форма", "change": "+$intensity", "time_offset": 0}
                ],
                updates=[
                    {"target": "person", "property": "fitness", "change": "+$intensity"},
                    {"target": "person", "property": "energy", "change": "-20", "time_offset": 0},
                    {"target": "person", "property": "energy", "change": "+30", "time_offset": "2hours"}
                ],
                time_flow={"recovery": "24hours", "optimal_frequency": "every_2days"},
                uncertainty_patterns=["какие упражнения?", "сколько времени?", "какая интенсивность?"]
            )
        }
    
    def process_life_command(self, command: str) -> Dict[str, Any]:
        """
        Обрабатывает жизненную команду и создает все необходимые объекты
        """
        # Парсинг команды
        parsed = self._parse_command(command)
        
        # Определение шаблона
        template = self._match_template(parsed)
        
        if not template:
            return {"error": "Не найден подходящий шаблон", "command": command}
        
        # Создание объектов по шаблону
        created_objects = []
        updated_objects = []
        
        # Создание новых объектов
        for create_spec in template.creates:
            obj = self._create_object(create_spec, parsed)
            created_objects.append(obj)
        
        # Обновление существующих объектов
        for update_spec in template.updates:
            updated = self._update_object(update_spec, parsed)
            updated_objects.append(updated)
        
        # Установка временных связей
        self._setup_time_flow(template.time_flow, created_objects)
        
        # Анализ неопределенностей
        uncertainties = self._analyze_uncertainties(command, template)
        
        return {
            "success": True,
            "command": command,
            "created_objects": created_objects,
            "updated_objects": updated_objects,
            "time_flow": template.time_flow,
            "uncertainties": uncertainties
        }
    
    def _parse_command(self, command: str) -> Dict[str, Any]:
        """Парсит команду и извлекает сущности"""
        words = command.lower().split()
        
        parsed = {
            "action": command.lower(),
            "entities": [],
            "amounts": [],
            "time_references": [],
            "locations": [],
            "people": []
        }
        
        # Поиск числовых значений
        for word in words:
            if word.isdigit():
                parsed["amounts"].append(int(word))
            elif "час" in word or "мин" in word or "день" in word:
                parsed["time_references"].append(word)
        
        # Поиск людей (имена с большой буквы)
        original_words = command.split()
        for word in original_words:
            if word[0].isupper() and len(word) > 1:
                parsed["people"].append(word)
        
        return parsed
    
    def _match_template(self, parsed: Dict[str, Any]) -> Optional[LifeActionTemplate]:
        """Находит подходящий шаблон для команды"""
        for template_name, template in self.templates.items():
            for trigger in template.triggers:
                if trigger in parsed.get("action", "").lower():
                    return template
        return None
    
    def _create_object(self, spec: Dict[str, Any], parsed: Dict[str, Any]) -> LifeStateObject:
        """Создает объект по спецификации"""
        obj_id = str(uuid.uuid4())
        
        properties = {}
        for key, value in spec.items():
            if isinstance(value, str) and value.startswith("$"):
                # Подстановка значений из парсера
                var_name = value[1:]  # убираем $
                if var_name in parsed:
                    properties[key] = parsed[var_name]
                else:
                    properties[key] = f"unknown_{var_name}"
            else:
                properties[key] = value
        
        timestamp = self.current_time
        if "time_offset" in spec:
            offset = self._parse_time_offset(spec["time_offset"])
            timestamp += offset
        
        obj = LifeStateObject(
            id=obj_id,
            entity_type=spec["type"],
            properties=properties,
            timestamp=timestamp
        )
        
        self.objects[obj_id] = obj
        return obj
    
    def _update_object(self, spec: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Обновляет существующий объект"""
        target_type = spec["target"]
        target_objects = [obj for obj in self.objects.values() if obj.entity_type == target_type]
        
        if not target_objects:
            # Создаем объект если его нет
            obj_id = str(uuid.uuid4())
            obj = LifeStateObject(
                id=obj_id,
                entity_type=target_type,
                properties={spec["property"]: spec["value"]},
                timestamp=self.current_time
            )
            self.objects[obj_id] = obj
            return {"created": obj}
        
        # Обновляем существующий объект
        target_obj = target_objects[0]  # Берем первый найденный
        
        if "change" in spec:
            change_value = spec["change"]
            if isinstance(change_value, str) and change_value.startswith("+"):
                # Увеличиваем значение
                current = target_obj.properties.get(spec["property"], 0)
                # Обрабатываем переменные в change_value
                if change_value.startswith("+$"):
                    var_name = change_value[2:]  # убираем +$
                    if var_name in parsed and parsed[var_name]:
                        change_amount = parsed[var_name][0] if isinstance(parsed[var_name], list) else parsed[var_name]
                        target_obj.properties[spec["property"]] = current + int(change_amount)
                    else:
                        target_obj.properties[spec["property"]] = current + 0
                else:
                    target_obj.properties[spec["property"]] = current + int(change_value[1:])
            elif isinstance(change_value, str) and change_value.startswith("-"):
                # Уменьшаем значение
                current = target_obj.properties.get(spec["property"], 0)
                # Обрабатываем переменные в change_value
                if change_value.startswith("-$"):
                    var_name = change_value[2:]  # убираем -$
                    if var_name in parsed and parsed[var_name]:
                        change_amount = parsed[var_name][0] if isinstance(parsed[var_name], list) else parsed[var_name]
                        target_obj.properties[spec["property"]] = current - int(change_amount)
                    else:
                        target_obj.properties[spec["property"]] = current - 0
                else:
                    target_obj.properties[spec["property"]] = current - int(change_value[1:])
        else:
            # Устанавливаем значение
            if "value" in spec:
                target_obj.properties[spec["property"]] = spec["value"]
            elif "add" in spec:
                # Добавляем в список
                current_list = target_obj.properties.get(spec["property"], [])
                if isinstance(current_list, list):
                    current_list.append(spec["add"])
                    target_obj.properties[spec["property"]] = current_list
                else:
                    target_obj.properties[spec["property"]] = [spec["add"]]
        
        return {"updated": target_obj}
    
    def _setup_time_flow(self, time_flow: Dict[str, Any], created_objects: List[LifeStateObject]):
        """Устанавливает временные связи между объектами"""
        for flow_name, flow_spec in time_flow.items():
            # Создание будущих событий
            future_time = self.current_time + self._parse_time_offset(flow_spec)
            
            # Создание объекта будущего события
            future_obj = LifeStateObject(
                id=str(uuid.uuid4()),
                entity_type="future_event",
                properties={
                    "event_type": flow_name,
                    "trigger_time": future_time,
                    "related_objects": [obj.id for obj in created_objects]
                },
                timestamp=future_time
            )
            
            self.objects[future_obj.id] = future_obj
    
    def _analyze_uncertainties(self, command: str, template: LifeActionTemplate) -> List[str]:
        """Анализирует команду на неопределенности"""
        uncertainties = []
        
        for pattern in template.uncertainty_patterns:
            if "?" in pattern:
                # Проверяем, есть ли в команде неопределенности
                if self._has_uncertainty(command, pattern):
                    uncertainties.append(pattern)
        
        return uncertainties
    
    def _has_uncertainty(self, command: str, pattern: str) -> bool:
        """Проверяет наличие неопределенности в команде"""
        # Простая проверка по ключевым словам
        uncertainty_keywords = ["что", "кто", "где", "когда", "как", "почему", "сколько", "какой"]
        
        for keyword in uncertainty_keywords:
            if keyword in command.lower():
                return True
        
        return False
    
    def _parse_time_offset(self, offset_str: str) -> timedelta:
        """Парсит строку времени в timedelta"""
        if isinstance(offset_str, str):
            try:
                if "min" in offset_str:
                    minutes = int(offset_str.replace("min", "").replace("utes", ""))
                    return timedelta(minutes=minutes)
                elif "hour" in offset_str:
                    hours = int(offset_str.replace("hours", "").replace("hour", ""))
                    return timedelta(hours=hours)
                elif "day" in offset_str:
                    days = int(offset_str.replace("days", "").replace("day", ""))
                    return timedelta(days=days)
                elif "month" in offset_str:
                    return timedelta(days=30)  # Упрощение
                elif "ongoing" in offset_str:
                    return timedelta(days=1)  # Для ongoing процессов
                elif "every_" in offset_str:
                    # Для периодических событий
                    return timedelta(days=1)
                else:
                    return timedelta()  # По умолчанию
            except ValueError:
                # Если не удается распарсить, возвращаем 0
                return timedelta()
        return timedelta()
    
    def get_current_state(self, entity_type: str = None) -> List[LifeStateObject]:
        """Возвращает текущее состояние объектов"""
        if entity_type:
            return [obj for obj in self.objects.values() if obj.entity_type == entity_type]
        return list(self.objects.values())
    
    def query_objects(self, conditions: Dict[str, Any]) -> List[LifeStateObject]:
        """Запрос объектов по условиям"""
        results = []
        for obj in self.objects.values():
            match = True
            for key, value in conditions.items():
                if key not in obj.properties or obj.properties[key] != value:
                    match = False
                    break
            if match:
                results.append(obj)
        return results

class LifeCommandGenerator(nn.Module):
    """
    Нейросеть для генерации жизненных команд
    """
    
    def __init__(self, input_dim=768, hidden_dim=512, max_commands=30):
        super(LifeCommandGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_commands = max_commands
        
        # Энкодер для входного вектора
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # LSTM для генерации последовательности команд
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.3
        )
        
        # Выходной слой для генерации токенов команд
        self.output_layer = nn.Linear(hidden_dim, 2000)  # Увеличенный размер словаря
        
        # Attention механизм
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
    def forward(self, input_vector, target_commands=None, max_length=None):
        batch_size = input_vector.size(0)
        
        # Кодируем входной вектор
        encoded = self.encoder(input_vector)  # [batch_size, hidden_dim]
        
        # Подготавливаем для LSTM
        encoded = encoded.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Инициализируем скрытое состояние
        h0 = torch.zeros(3, batch_size, self.hidden_dim).to(input_vector.device)
        c0 = torch.zeros(3, batch_size, self.hidden_dim).to(input_vector.device)
        
        # Генерируем последовательность
        outputs = []
        current_input = encoded
        
        max_len = max_length or self.max_commands
        
        for _ in range(max_len):
            # LSTM шаг
            lstm_out, (h0, c0) = self.lstm(current_input, (h0, c0))
            
            # Attention
            lstm_out = lstm_out.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = attn_out.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
            
            # Выходной слой
            output = self.output_layer(lstm_out[:, -1, :])  # Берем последний выход
            outputs.append(output)
            
            # Следующий вход: всегда подаем encoded
            current_input = encoded  # [batch_size, 1, hidden_dim]
        
        # После генерации outputs
        outputs = torch.stack(outputs, dim=1)  # [batch, max_len, vocab_size]
        return outputs

def main():
    """
    Демонстрация интегрированной системы
    """
    print("=== ИНТЕГРИРОВАННАЯ СИСТЕМА ЖИЗНЕННЫХ КОМАНД INFERA ===")
    
    # Создаем систему
    life_system = UnifiedLifeSystem()
    
    # Примеры команд
    test_commands = [
        "Я съел борщ",
        "Заработал 50000 рублей",
        "Встретил Алину в кафе",
        "Тренировался в зале 2 часа",
        "Лег спать в 23:00",
        "Изучаю Python уже 3 месяца"
    ]
    
    for command in test_commands:
        print(f"\n📝 Команда: {command}")
        result = life_system.process_life_command(command)
        
        if result.get("success"):
            print(f"✅ Создано объектов: {len(result['created_objects'])}")
            print(f"🔄 Обновлено объектов: {len(result['updated_objects'])}")
            
            for obj in result['created_objects'][:2]:  # Показываем первые 2
                print(f"   - {obj.entity_type}: {obj.properties}")
            
            if result.get('uncertainties'):
                print(f"❓ Неопределенности: {result['uncertainties']}")
        else:
            print(f"❌ Ошибка: {result.get('error', 'Unknown error')}")
    
    print(f"\n📊 Всего объектов в системе: {len(life_system.objects)}")
    print(f"🧠 Состояние персоны: {len(life_system.get_current_state('person'))} объектов")
    print(f"💰 Финансовые транзакции: {len(life_system.get_current_state('transaction'))} объектов")
    print(f"🎯 Действия: {len(life_system.get_current_state('action'))} объектов")

if __name__ == "__main__":
    main() 