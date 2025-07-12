import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
import ast
import operator

class ObjectType(Enum):
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    CONTROLLER = "controller"
    SYSTEM = "system"
    DEVICE = "device"
    COMPONENT = "component"
    INTERFACE = "interface"
    NETWORK = "network"
    DATABASE = "database"
    SERVICE = "service"

class ObjectState(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    ENABLED = "enabled"
    DISABLED = "disabled"
    RUNNING = "running"
    STOPPED = "stopped"
    PAUSED = "paused"
    IDLE = "idle"
    BUSY = "busy"
    READY = "ready"
    WAITING = "waiting"
    ERROR = "error"
    WARNING = "warning"
    NORMAL = "normal"
    ABNORMAL = "abnormal"
    CRITICAL = "critical"
    SAFE = "safe"
    UNSAFE = "unsafe"
    OPEN = "open"
    CLOSED = "closed"
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    FULL = "full"
    EMPTY = "empty"
    PARTIAL = "partial"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    VALID = "valid"
    INVALID = "invalid"
    TRUE = "true"
    FALSE = "false"
    YES = "yes"
    NO = "no"

@dataclass
class ObjectProperty:
    name: str
    value: Any
    unit: Optional[str] = None
    data_type: str = "string"
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    is_formula: bool = False
    formula: Optional[str] = None

@dataclass
class ObjectNode:
    id: str
    name: str
    object_type: ObjectType
    properties: Dict[str, ObjectProperty] = field(default_factory=dict)
    children: List['ObjectNode'] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)  # Список ID родителей
    state: ObjectState = ObjectState.INACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    position: Optional[Dict[str, float]] = None
    orientation: Optional[Dict[str, float]] = None
    quantity: int = 1
    conditions: List[str] = field(default_factory=list)
    logical_connections: List[str] = field(default_factory=list)
    time_chain: List[str] = field(default_factory=list)  # Цепочка объектов по времени



@dataclass
class CommandResult:
    success: bool
    message: str
    created_objects: List[ObjectNode] = field(default_factory=list)
    modified_objects: List[ObjectNode] = field(default_factory=list)
    deleted_objects: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    computed_values: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemState:
    objects: Dict[str, ObjectNode] = field(default_factory=dict)
    object_counter: int = 0
    execution_history: List[CommandResult] = field(default_factory=list)
    contradictions_log: List[str] = field(default_factory=list)
    warnings_log: List[str] = field(default_factory=list)

class FormulaEvaluator:
    """Вычисляет формулы и условия"""
    
    def __init__(self, system_state: SystemState):
        self.system_state = system_state
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne,
            '!': operator.not_
        }
    
    def evaluate_formula(self, formula: str, context_object: ObjectNode) -> Tuple[Any, bool]:
        """
        Вычисляет формулу и возвращает результат
        """
        try:
            # Заменяем ссылки на объекты на их значения
            processed_formula = self._replace_object_references(formula, context_object)
            
            # Если формула содержит неизвестные значения, возвращаем частично вычисленную
            if '[' in processed_formula and ']' in processed_formula:
                return processed_formula, False  # Не полностью вычислена
            
            # Вычисляем результат
            result = self._evaluate_expression(processed_formula)
            return result, True
            
        except Exception as e:
            return f"ERROR: {str(e)}", False
    
    def _replace_object_references(self, formula: str, context_object: ObjectNode) -> str:
        """
        Заменяет ссылки на объекты на их значения
        """
        # Ищем паттерны типа nameObject.property
        pattern = r'(\w+)\.(\w+)'
        
        def replace_match(match):
            obj_name = match.group(1)
            prop_name = match.group(2)
            
            # Ищем объект по имени
            target_object = self._find_object_by_name(obj_name)
            if target_object and prop_name in target_object.properties:
                value = target_object.properties[prop_name].value
                return str(value)
            else:
                # Возвращаем оригинальную ссылку, если значение не найдено
                return match.group(0)
        
        return re.sub(pattern, replace_match, formula)
    
    def _find_object_by_name(self, name: str) -> Optional[ObjectNode]:
        """
        Находит объект по имени (последний в цепочке времени)
        """
        matching_objects = [
            obj for obj in self.system_state.objects.values()
            if obj.name == name
        ]
        
        if not matching_objects:
            return None
        
        # Возвращаем самый новый объект
        return max(matching_objects, key=lambda x: x.created_at)
    
    def _evaluate_expression(self, expression: str) -> Any:
        """
        Вычисляет математическое выражение
        """
        # Убираем пробелы
        expression = expression.replace(' ', '')
        
        # Обрабатываем логическое НЕ
        if expression.startswith('!'):
            inner_result = self._evaluate_expression(expression[1:])
            return not inner_result
        
        # Обрабатываем сравнения
        for op in ['>=', '<=', '!=', '==', '>', '<']:
            if op in expression:
                left, right = expression.split(op, 1)
                left_val = self._evaluate_expression(left)
                right_val = self._evaluate_expression(right)
                return self.operators[op](left_val, right_val)
        
        # Обрабатываем арифметические операции
        for op in ['*', '/', '+', '-']:
            if op in expression:
                parts = expression.split(op)
                if len(parts) == 2:
                    left_val = self._evaluate_expression(parts[0])
                    right_val = self._evaluate_expression(parts[1])
                    return self.operators[op](left_val, right_val)
        
        # Если это число
        try:
            return float(expression)
        except ValueError:
            # Если это строка
            return expression

class CommandController:
    def __init__(self):
        self.system_state = SystemState()
        self.formula_evaluator = FormulaEvaluator(self.system_state)
        
    def execute_commands(self, commands_string: str) -> CommandResult:
        import time
        start_time = time.time()
        result = CommandResult(success=True, message="")
        
        # Сохраняем обычные созданные объекты отдельно
        normal_created_objects = []
        
        try:
            commands = self._parse_commands(commands_string)
            for command in commands:
                command_result = self._execute_single_command(command)
                # Обычные объекты не добавляем в result.created_objects
                normal_created_objects.extend([
                    obj for obj in command_result.created_objects
                    if not (obj.name.startswith("$question") or obj.name.startswith("$it"))
                ])
                result.modified_objects.extend(command_result.modified_objects)
                result.deleted_objects.extend(command_result.deleted_objects)
                result.contradictions.extend(command_result.contradictions)
                result.warnings.extend(command_result.warnings)
                result.computed_values.update(command_result.computed_values)
                if not command_result.success:
                    result.success = False
                    result.message = command_result.message
                    break
            # После выполнения команд анализируем объекты и создаём только объекты ошибок
            self._analyze_objects_and_create_questions(result)
            # Оставляем в created_objects только объекты с ошибками
            result.created_objects = [obj for obj in result.created_objects if obj.name.startswith("$question") or obj.name.startswith("$it")]
        except Exception as e:
            result.success = False
            result.message = f"Ошибка выполнения команды: {str(e)}"
        result.execution_time = time.time() - start_time
        self.system_state.execution_history.append(result)
        return result
    
    def _parse_commands(self, commands_string: str) -> List[Dict[str, Any]]:
        """
        Парсит строку команд
        """
        commands = []
        
        # Убираем квадратные скобки и разделяем по запятым
        clean_commands = commands_string.strip('[]').split(',')
        
        for command_str in clean_commands:
            command_str = command_str.strip()
            if not command_str:
                continue
                
            # Парсим команду
            parsed = self._parse_single_command(command_str)
            if parsed:
                commands.append(parsed)
        
        return commands
    
    def _parse_single_command(self, command_str: str) -> Optional[Dict[str, Any]]:
        """
        Парсит одну команду с поддержкой условий в полях
        """
        parts = command_str.split()
        
        if len(parts) < 2:
            return None
        
        command = {
            "operation": parts[0],
            "object_name": parts[1],
            "properties": {},
            "conditions": [],
            "dependencies": [],
            "parents": []
        }
        
        i = 2
        while i < len(parts):
            if parts[i] == "value":
                if i + 1 < len(parts):
                    command["properties"]["value"] = parts[i + 1]
                    i += 2
            elif parts[i] == "unit":
                if i + 1 < len(parts):
                    command["properties"]["unit"] = parts[i + 1]
                    i += 2
            elif parts[i] == "quantity":
                if i + 1 < len(parts):
                    command["properties"]["quantity"] = int(parts[i + 1])
                    i += 2
            elif parts[i] == "position":
                if i + 1 < len(parts):
                    # Собираем всю позицию (может содержать условия)
                    position_parts = []
                    i += 1
                    while i < len(parts) and parts[i] != "," and not self._is_property_name(parts[i]):
                        position_parts.append(parts[i])
                        i += 1
                    command["properties"]["position"] = " ".join(position_parts)
            elif parts[i] == "orientation":
                if i + 1 < len(parts):
                    command["properties"]["orientation"] = parts[i + 1]
                    i += 2
            elif parts[i] == "formula":
                # Собираем формулу
                formula_parts = []
                i += 1
                while i < len(parts) and parts[i] != ",":
                    formula_parts.append(parts[i])
                    i += 1
                command["properties"]["formula"] = " ".join(formula_parts)
            elif parts[i] == "time":
                if i + 1 < len(parts):
                    # Собираем время (может быть диапазоном)
                    time_parts = []
                    i += 1
                    while i < len(parts) and parts[i] != "," and not self._is_property_name(parts[i]):
                        time_parts.append(parts[i])
                        i += 1
                    command["properties"]["time"] = " ".join(time_parts)
            elif parts[i] == "parents":
                # Собираем родителей
                parent_parts = []
                i += 1
                while i < len(parts) and parts[i] != ",":
                    parent_parts.append(parts[i])
                    i += 1
                command["parents"] = [p.strip() for p in " ".join(parent_parts).split(",")]
            elif parts[i] == "condition":
                # Собираем условие
                condition_parts = []
                i += 1
                while i < len(parts) and parts[i] != ",":
                    condition_parts.append(parts[i])
                    i += 1
                command["conditions"].append(" ".join(condition_parts))
            elif parts[i] == "dependency":
                # Собираем зависимость
                dependency_parts = []
                i += 1
                while i < len(parts) and parts[i] != ",":
                    dependency_parts.append(parts[i])
                    i += 1
                command["dependencies"].append(" ".join(dependency_parts))
            elif self._is_property_name(parts[i]):
                # Любое другое свойство
                prop_name = parts[i]
                prop_parts = []
                i += 1
                while i < len(parts) and parts[i] != "," and not self._is_property_name(parts[i]):
                    prop_parts.append(parts[i])
                    i += 1
                command["properties"][prop_name] = " ".join(prop_parts)
            else:
                i += 1
        
        return command
    
    def _is_property_name(self, word: str) -> bool:
        """
        Проверяет, является ли слово именем свойства
        """
        property_names = {
            "value", "unit", "quantity", "position", "orientation", 
            "time", "formula", "parents", "condition", "dependency",
            "color", "size", "weight", "age", "cost", "exists", "name",
            "type", "status", "mode", "power", "speed", "length", "width",
            "height", "temperature", "pressure", "humidity", "brightness",
            "volume", "mass", "density", "energy", "force", "velocity"
        }
        return word in property_names
    
    def _execute_single_command(self, command: Dict[str, Any]) -> CommandResult:
        """
        Выполняет одну команду
        """
        operation = command["operation"]
        object_name = command["object_name"]
        
        if operation == "create":
            return self._create_object(command)
        elif operation == "set":
            return self._set_object_property(command)
        elif operation == "delete":
            return self._delete_object(command)
        elif operation == "configure":
            return self._configure_object(command)
        elif operation == "activate":
            return self._activate_object(command)
        elif operation == "deactivate":
            return self._deactivate_object(command)
        elif operation == "count":
            return self._count_formula(command)
        else:
            return CommandResult(
                success=False,
                message=f"Неизвестная операция: {operation}"
            )
    
    def _create_object(self, command: Dict[str, Any]) -> CommandResult:
        """
        Создает новый объект с наследованием по времени
        """
        object_name = command["object_name"]
        properties = command.get("properties", {})
        quantity = properties.get("quantity", 1)
        parents = command.get("parents", [])
        
        result = CommandResult(success=True, message=f"Создан объект: {object_name}")
        
        # Определяем тип объекта
        object_type = self._determine_object_type(object_name)
        
        # Создаем объекты
        for i in range(quantity):
            object_id = f"{object_name}_{self.system_state.object_counter}"
            self.system_state.object_counter += 1
            
            # Создаем объект
            new_object = ObjectNode(
                id=object_id,
                name=object_name,
                object_type=object_type,
                parents=parents
            )
            
            # Наследуем свойства от родителей
            self._inherit_properties(new_object, parents)
            
            # Добавляем пользовательские свойства
            for prop_name, prop_value in properties.items():
                if prop_name != "quantity":
                    new_object.properties[prop_name] = ObjectProperty(
                        name=prop_name,
                        value=prop_value,
                        timestamp=datetime.now()
                    )
            
            # Обновляем цепочку времени
            self._update_time_chain(new_object)
            
            # Сохраняем объект
            self.system_state.objects[object_id] = new_object
            result.created_objects.append(new_object)
        
        return result
    
    def _inherit_properties(self, new_object: ObjectNode, parent_names: List[str]):
        """
        Наследует свойства от родителей
        """
        for parent_name in parent_names:
            # Ищем родительские объекты
            parent_objects = [
                obj for obj in self.system_state.objects.values()
                if obj.name == parent_name and obj.created_at < new_object.created_at
            ]
            
            if parent_objects:
                # Берем ближайшего по времени родителя
                closest_parent = max(parent_objects, key=lambda x: x.created_at)
                
                # Наследуем свойства, которых нет у нового объекта
                for prop_name, prop_value in closest_parent.properties.items():
                    if prop_name not in new_object.properties:
                        new_object.properties[prop_name] = ObjectProperty(
                            name=prop_name,
                            value=prop_value.value,
                            unit=prop_value.unit,
                            data_type=prop_value.data_type,
                            timestamp=datetime.now(),
                            source=f"inherited_from_{closest_parent.id}"
                        )
    
    def _update_time_chain(self, new_object: ObjectNode):
        """
        Обновляет цепочку времени для объекта
        """
        # Находим все объекты с таким же именем
        same_name_objects = [
            obj for obj in self.system_state.objects.values()
            if obj.name == new_object.name
        ]
        
        # Сортируем по времени создания
        same_name_objects.sort(key=lambda x: x.created_at)
        
        # Обновляем цепочку времени для всех объектов
        for obj in same_name_objects:
            obj.time_chain = [o.id for o in same_name_objects if o.created_at <= obj.created_at]
    
    def _set_object_property(self, command: Dict[str, Any]) -> CommandResult:
        """
        Устанавливает свойство объекта
        """
        object_name = command["object_name"]
        properties = command.get("properties", {})
        
        result = CommandResult(success=True, message=f"Установлены свойства для: {object_name}")
        
        # Ищем объекты с таким именем
        matching_objects = [
            obj for obj in self.system_state.objects.values()
            if obj.name == object_name
        ]
        
        if not matching_objects:
            return CommandResult(
                success=False,
                message=f"Объект не найден: {object_name}"
            )
        
        # Обновляем свойства
        for obj in matching_objects:
            for prop_name, prop_value in properties.items():
                if prop_name in obj.properties:
                    obj.properties[prop_name].value = prop_value
                    obj.properties[prop_name].timestamp = datetime.now()
                else:
                    obj.properties[prop_name] = ObjectProperty(
                        name=prop_name,
                        value=prop_value,
                        timestamp=datetime.now()
                    )
            
            obj.modified_at = datetime.now()
            result.modified_objects.append(obj)
        
        return result
    
    def _delete_object(self, command: Dict[str, Any]) -> CommandResult:
        """
        Удаляет объект и всех его потомков
        """
        object_name = command["object_name"]
        
        result = CommandResult(success=True, message=f"Удален объект: {object_name}")
        
        # Ищем объекты для удаления
        objects_to_delete = []
        for obj_id, obj in self.system_state.objects.items():
            if obj.name == object_name:
                objects_to_delete.append(obj_id)
        
        # Удаляем объекты и их потомков
        for obj_id in objects_to_delete:
            self._delete_object_and_children(obj_id, result)
        
        if not objects_to_delete:
            return CommandResult(
                success=False,
                message=f"Объект не найден: {object_name}"
            )
        
        return result
    
    def _delete_object_and_children(self, obj_id: str, result: CommandResult):
        """
        Рекурсивно удаляет объект и всех его потомков
        """
        if obj_id in self.system_state.objects:
            obj = self.system_state.objects[obj_id]
            
            # Удаляем всех потомков
            for child_id in list(self.system_state.objects.keys()):
                child = self.system_state.objects[child_id]
                if obj_id in child.parents:
                    self._delete_object_and_children(child_id, result)
            
            # Удаляем сам объект
            del self.system_state.objects[obj_id]
            result.deleted_objects.append(obj_id)
    
    def _configure_object(self, command: Dict[str, Any]) -> CommandResult:
        """
        Настраивает объект
        """
        object_name = command["object_name"]
        properties = command.get("properties", {})
        
        result = CommandResult(success=True, message=f"Настроен объект: {object_name}")
        
        # Ищем объекты
        matching_objects = [
            obj for obj in self.system_state.objects.values()
            if obj.name == object_name
        ]
        
        if not matching_objects:
            return CommandResult(
                success=False,
                message=f"Объект не найден: {object_name}"
            )
        
        # Настраиваем объекты
        for obj in matching_objects:
            for prop_name, prop_value in properties.items():
                obj.properties[prop_name] = ObjectProperty(
                    name=prop_name,
                    value=prop_value,
                    timestamp=datetime.now()
                )
            
            obj.modified_at = datetime.now()
            result.modified_objects.append(obj)
        
        return result
    
    def _activate_object(self, command: Dict[str, Any]) -> CommandResult:
        """
        Активирует объект
        """
        object_name = command["object_name"]
        
        result = CommandResult(success=True, message=f"Активирован объект: {object_name}")
        
        # Ищем объекты
        matching_objects = [
            obj for obj in self.system_state.objects.values()
            if obj.name == object_name
        ]
        
        if not matching_objects:
            return CommandResult(
                success=False,
                message=f"Объект не найден: {object_name}"
            )
        
        # Активируем объекты
        for obj in matching_objects:
            obj.state = ObjectState.ACTIVE
            obj.modified_at = datetime.now()
            result.modified_objects.append(obj)
        
        return result
    
    def _deactivate_object(self, command: Dict[str, Any]) -> CommandResult:
        """
        Деактивирует объект
        """
        object_name = command["object_name"]
        
        result = CommandResult(success=True, message=f"Деактивирован объект: {object_name}")
        
        # Ищем объекты
        matching_objects = [
            obj for obj in self.system_state.objects.values()
            if obj.name == object_name
        ]
        
        if not matching_objects:
            return CommandResult(
                success=False,
                message=f"Объект не найден: {object_name}"
            )
        
        # Деактивируем объекты
        for obj in matching_objects:
            obj.state = ObjectState.INACTIVE
            obj.modified_at = datetime.now()
            result.modified_objects.append(obj)
        
        return result
    
    def _count_formula(self, command: Dict[str, Any]) -> CommandResult:
        """
        Вычисляет формулу и сохраняет результат
        """
        object_name = command["object_name"]
        properties = command.get("properties", {})
        
        result = CommandResult(success=True, message=f"Вычислена формула для: {object_name}")
        
        # Ищем объекты с таким именем
        matching_objects = [
            obj for obj in self.system_state.objects.values()
            if obj.name == object_name
        ]
        
        if not matching_objects:
            return CommandResult(
                success=False,
                message=f"Объект не найден: {object_name}"
            )
        
        # Вычисляем формулы для каждого объекта
        for obj in matching_objects:
            for prop_name, prop_value in properties.items():
                # Создаем формулу как свойство объекта
                formula_prop = ObjectProperty(
                    name=prop_name,
                    value=prop_value,
                    timestamp=datetime.now(),
                    is_formula=True,
                    formula=prop_value
                )
                
                obj.properties[prop_name] = formula_prop
                obj.modified_at = datetime.now()
                result.modified_objects.append(obj)
        
        return result
    
    def _determine_object_type(self, object_name: str) -> ObjectType:
        """
        Определяет тип объекта по имени
        """
        if "sensor" in object_name:
            return ObjectType.SENSOR
        elif "robot" in object_name or "motor" in object_name or "actuator" in object_name:
            return ObjectType.ACTUATOR
        elif "controller" in object_name or "plc" in object_name:
            return ObjectType.CONTROLLER
        elif "system" in object_name:
            return ObjectType.SYSTEM
        elif "network" in object_name or "ethernet" in object_name:
            return ObjectType.NETWORK
        elif "database" in object_name or "server" in object_name:
            return ObjectType.DATABASE
        elif "service" in object_name:
            return ObjectType.SERVICE
        elif "interface" in object_name or "hmi" in object_name:
            return ObjectType.INTERFACE
        else:
            return ObjectType.DEVICE
    
    def _evaluate_all_formulas_and_conditions(self, result: CommandResult):
        """
        Вычисляет все формулы и условия
        """
        for obj in self.system_state.objects.values():
            # Вычисляем формулы в свойствах
            for prop_name, prop in obj.properties.items():
                if prop.is_formula and prop.formula:
                    computed_value, success = self.formula_evaluator.evaluate_formula(
                        prop.formula, obj
                    )
                    if success:
                        prop.value = computed_value
                        result.computed_values[f"{obj.name}.{prop_name}"] = computed_value
                    else:
                        # Сохраняем частично вычисленную формулу
                        prop.value = computed_value
                        result.computed_values[f"{obj.name}.{prop_name}"] = computed_value
            
            # Вычисляем условия существования
            for condition in obj.conditions:
                computed_value, success = self.formula_evaluator.evaluate_formula(
                    condition, obj
                )
                if success:
                    if not computed_value:  # Если условие false
                        obj.state = ObjectState.NOT_EXISTS
                        result.computed_values[f"{obj.name}.{condition}"] = False
                    else:
                        result.computed_values[f"{obj.name}.{condition}"] = True
                else:
                    result.computed_values[f"{obj.name}.{condition}"] = computed_value
    
    def _create_question_object(self, name: str, problem_description: str) -> ObjectNode:
        """
        Создает простой объект question с проблемой
        """
        question_obj = ObjectNode(
            id=f"question_{self.system_state.object_counter}",
            name=name,
            object_type=ObjectType.SYSTEM,
            properties={
                "problem": ObjectProperty(
                    name="problem",
                    value=problem_description,
                    data_type="string"
                )
            }
        )
        self.system_state.object_counter += 1
        self.system_state.objects[question_obj.id] = question_obj
        return question_obj
    
    def _create_it_object(self, name: str, problem_description: str) -> ObjectNode:
        """
        Создает простой объект it с проблемой
        """
        it_obj = ObjectNode(
            id=f"it_{self.system_state.object_counter}",
            name=name,
            object_type=ObjectType.SYSTEM,
            properties={
                "problem": ObjectProperty(
                    name="problem",
                    value=problem_description,
                    data_type="string"
                )
            }
        )
        self.system_state.object_counter += 1
        self.system_state.objects[it_obj.id] = it_obj
        return it_obj

    def _analyze_objects_and_create_questions(self, result: CommandResult):
        """
        Анализирует объекты и создает простые объекты question и it для проблемных случаев
        """
        question_counter = 0
        it_counter = 0
        
        # Создаем копию списка объектов, чтобы избежать изменения во время итерации
        objects_to_analyze = list(self.system_state.objects.values())
        
        for obj in objects_to_analyze:
            # Анализируем каждое свойство объекта
            for prop_name, prop in obj.properties.items():
                question_obj = self._analyze_property_for_questions(obj, prop_name, prop)
                if question_obj:
                    result.created_objects.append(question_obj)
                    question_counter += 1
            
            # Анализируем условия существования
            for condition in obj.conditions:
                question_obj = self._analyze_condition_for_questions(obj, condition)
                if question_obj:
                    result.created_objects.append(question_obj)
                    question_counter += 1
    
    def _analyze_property_for_questions(self, obj: ObjectNode, prop_name: str, prop: ObjectProperty):
        """
        Анализирует свойство на предмет проблем и создает простой объект question или it
        """
        value = prop.value
        
        # Проверяем отрицательные значения
        if isinstance(value, (int, float)) and value < 0:
            question_obj = self._create_question_object(f"$question${len(self.system_state.objects)}", 
                                                      f"Отрицательное значение {prop_name}: {value}")
            return question_obj
        
        # Проверяем некорректные имена (it, it1, it2)
        if prop_name == "name" and isinstance(value, str) and value.startswith("it"):
            it_obj = self._create_it_object(f"$it${len(self.system_state.objects)}", 
                                          f"Временное имя: {value}")
            return it_obj
        
        # Проверяем расстояния
        if prop_name == "position" and isinstance(value, str):
            if "distance" in value.lower() or "расстояние" in value.lower():
                # Анализируем условие расстояния
                return self._analyze_distance_condition(obj, value)
        
        # Проверяем существование
        if prop_name == "exists" and value == False:
            question_obj = self._create_question_object(f"$question${len(self.system_state.objects)}", 
                                                      f"Объект помечен как несуществующий")
            return question_obj
        
        # Проверяем количество
        if prop_name == "quantity" and isinstance(value, (int, float)) and value <= 0:
            question_obj = self._create_question_object(f"$question${len(self.system_state.objects)}", 
                                                      f"Некорректное количество: {value}")
            return question_obj
        
        return None
    
    def _analyze_distance_condition(self, obj: ObjectNode, position_value: str):
        """
        Анализирует условие расстояния
        """
        # Простой анализ - если есть числа больше 100, считаем проблемой
        import re
        numbers = re.findall(r'\d+', position_value)
        for num in numbers:
            if int(num) > 100:
                question_obj = self._create_question_object(f"$question${len(self.system_state.objects)}", 
                                                          f"Объект находится на расстоянии больше 100: {position_value}")
                return question_obj
        return None
    
    def _analyze_condition_for_questions(self, obj: ObjectNode, condition: str):
        """
        Анализирует условие на предмет проблем
        """
        # Проверяем условия, которые всегда ложны
        if "false" in condition.lower() or "0" in condition:
            question_obj = self._create_question_object(f"$question${len(self.system_state.objects)}", 
                                                      f"Объект имеет ложное условие: {condition}")
            return question_obj
        return None
    
    def get_system_tree(self) -> Dict[str, Any]:
        """
        Возвращает дерево объектов системы
        """
        tree = {
            "total_objects": len(self.system_state.objects),
            "object_types": {},
            "objects": {}
        }
        
        # Группируем по типам
        for obj in self.system_state.objects.values():
            obj_type = obj.object_type.value
            if obj_type not in tree["object_types"]:
                tree["object_types"][obj_type] = 0
            tree["object_types"][obj_type] += 1
            
            # Добавляем объект в дерево
            tree["objects"][obj.id] = {
                "name": obj.name,
                "type": obj_type,
                "state": obj.state.value,
                "parents": obj.parents,
                "time_chain": obj.time_chain,
                "properties": {
                    name: {
                        "value": prop.value,
                        "unit": prop.unit,
                        "data_type": prop.data_type,
                        "is_formula": prop.is_formula,
                        "formula": prop.formula,
                        "timestamp": prop.timestamp.isoformat()
                    }
                    for name, prop in obj.properties.items()
                },
                "conditions": obj.conditions,
                "quantity": obj.quantity,
                "created_at": obj.created_at.isoformat(),
                "modified_at": obj.modified_at.isoformat()
            }
        
        return tree
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку выполнения команд
        """
        if not self.system_state.execution_history:
            return {"message": "Нет истории выполнения"}
        
        latest_result = self.system_state.execution_history[-1]
        
        return {
            "success": latest_result.success,
            "message": latest_result.message,
            "created_objects": len(latest_result.created_objects),
            "modified_objects": len(latest_result.modified_objects),
            "deleted_objects": len(latest_result.deleted_objects),
            "contradictions": len(latest_result.contradictions),
            "warnings": len(latest_result.warnings),
            "computed_values": latest_result.computed_values,
            "execution_time": latest_result.execution_time,
            "total_objects_in_system": len(self.system_state.objects)
        }

if __name__ == "__main__":
    # Тестируем контроллер
    controller = CommandController()
    
    # Пример команд с наследованием и формулами
    test_commands = "[create robot age 25 cost 10 time 10:00, create robot age 30 time 11:00 parents robot_0, set robot age 35, count robot.age - robot.cost * 2]"
    
    print("Выполняем команды...")
    result = controller.execute_commands(test_commands)
    
    print(f"\nРезультат выполнения:")
    print(f"Успех: {result.success}")
    print(f"Сообщение: {result.message}")
    print(f"Создано объектов: {len(result.created_objects)}")
    print(f"Изменено объектов: {len(result.modified_objects)}")
    print(f"Удалено объектов: {len(result.deleted_objects)}")
    print(f"Противоречия: {len(result.contradictions)}")
    print(f"Предупреждения: {len(result.warnings)}")
    print(f"Вычисленные значения: {result.computed_values}")
    print(f"Время выполнения: {result.execution_time:.3f} сек")
    
    # Показываем дерево объектов
    print("\nДерево объектов:")
    tree = controller.get_system_tree()
    print(json.dumps(tree, indent=2, ensure_ascii=False))
    
    # Показываем сводку
    print("\nСводка выполнения:")
    summary = controller.get_execution_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False)) 