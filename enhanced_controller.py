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
    PERSON = "person"
    OBJECT = "object"
    TRANSFER = "transfer"
    QUESTION = "question"
    IT = "it"

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
    parents: List[str] = field(default_factory=list)
    state: ObjectState = ObjectState.INACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    position: Optional[Dict[str, float]] = None
    orientation: Optional[Dict[str, float]] = None
    quantity: int = 1
    conditions: List[str] = field(default_factory=list)
    logical_connections: List[str] = field(default_factory=list)
    time_chain: List[str] = field(default_factory=list)

@dataclass
class ConversationContext:
    current_user: Optional[str] = None
    last_mentioned_objects: List[str] = field(default_factory=list)
    topic: Optional[str] = None
    unresolved_questions: List[str] = field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

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
    questions_generated: List[str] = field(default_factory=list)

@dataclass
class SystemState:
    objects: Dict[str, ObjectNode] = field(default_factory=dict)
    object_counter: int = 0
    execution_history: List[CommandResult] = field(default_factory=list)
    contradictions_log: List[str] = field(default_factory=list)
    warnings_log: List[str] = field(default_factory=list)
    context: ConversationContext = field(default_factory=ConversationContext)

class EnhancedCommandController:
    """
    Улучшенный контроллер команд с поддержкой контекста и неопределенностей
    """
    
    def __init__(self):
        self.system_state = SystemState()
        self.formula_evaluator = FormulaEvaluator(self.system_state)
        
    def execute_commands(self, commands_string: str) -> CommandResult:
        """
        Выполняет команды с поддержкой контекста
        """
        start_time = datetime.now()
        
        try:
            # Парсим команды
            commands = self._parse_commands(commands_string)
            
            # Выполняем команды
            result = CommandResult(success=True, message="Команды выполнены успешно")
            
            for command in commands:
                command_result = self._execute_single_command(command)
                
                # Объединяем результаты
                result.created_objects.extend(command_result.created_objects)
                result.modified_objects.extend(command_result.modified_objects)
                result.deleted_objects.extend(command_result.deleted_objects)
                result.contradictions.extend(command_result.contradictions)
                result.warnings.extend(command_result.warnings)
                result.questions_generated.extend(command_result.questions_generated)
                
                if not command_result.success:
                    result.success = False
                    result.message = command_result.message
            
            # Обновляем контекст
            self._update_context(result)
            
            # Анализируем неопределенности
            self._analyze_uncertainties(result)
            
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка выполнения команд: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _parse_commands(self, commands_string: str) -> List[Dict[str, Any]]:
        """
        Парсит строку команд в список команд
        """
        commands = []
        
        # Убираем внешние скобки
        commands_string = commands_string.strip()
        if commands_string.startswith('[') and commands_string.endswith(']'):
            commands_string = commands_string[1:-1]
        
        # Разделяем на отдельные команды
        command_parts = self._split_commands(commands_string)
        
        for part in command_parts:
            command = self._parse_single_command(part.strip())
            if command:
                commands.append(command)
        
        return commands
    
    def _split_commands(self, commands_string: str) -> List[str]:
        """
        Разделяет строку команд на отдельные команды
        """
        commands = []
        current_command = ""
        bracket_count = 0
        
        for char in commands_string:
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
            
            if char == ',' and bracket_count == 0:
                if current_command.strip():
                    commands.append(current_command.strip())
                current_command = ""
            else:
                current_command += char
        
        if current_command.strip():
            commands.append(current_command.strip())
        
        return commands
    
    def _parse_single_command(self, command_str: str) -> Optional[Dict[str, Any]]:
        """
        Парсит одну команду
        """
        parts = command_str.split()
        if not parts:
            return None
        
        command_type = parts[0].lower()
        command = {"type": command_type}
        
        if command_type == "create":
            return self._parse_create_command(parts[1:])
        elif command_type == "set":
            return self._parse_set_command(parts[1:])
        elif command_type == "query":
            return self._parse_query_command(parts[1:])
        elif command_type == "count":
            return self._parse_count_command(parts[1:])
        elif command_type == "resolve":
            return self._parse_resolve_command(parts[1:])
        elif command_type == "if":
            return self._parse_if_command(parts[1:])
        elif command_type == "when":
            return self._parse_when_command(parts[1:])
        elif command_type == "define":
            return self._parse_define_command(parts[1:])
        
        return None
    
    def _parse_create_command(self, parts: List[str]) -> Dict[str, Any]:
        """
        Парсит команду create
        """
        command = {"type": "create"}
        
        if not parts:
            return command
        
        # Первое слово - тип объекта
        command["object_type"] = parts[0]
        
        # Парсим свойства
        properties = {}
        i = 1
        while i < len(parts):
            if parts[i] in ["name", "quantity", "owner", "place", "time", "value"]:
                prop_name = parts[i]
                i += 1
                if i < len(parts):
                    prop_value = parts[i]
                    # Обрабатываем кавычки
                    if prop_value.startswith('"') and prop_value.endswith('"'):
                        prop_value = prop_value[1:-1]
                    properties[prop_name] = prop_value
            i += 1
        
        command["properties"] = properties
        return command
    
    def _parse_set_command(self, parts: List[str]) -> Dict[str, Any]:
        """
        Парсит команду set
        """
        command = {"type": "set"}
        
        if len(parts) >= 3:
            command["object"] = parts[0]
            command["property"] = parts[1]
            command["value"] = " ".join(parts[2:])
        
        return command
    
    def _parse_query_command(self, parts: List[str]) -> Dict[str, Any]:
        """
        Парсит команду query
        """
        command = {"type": "query"}
        
        if len(parts) >= 2:
            command["object"] = parts[0]
            command["property"] = parts[1]
            if len(parts) > 2:
                command["conditions"] = parts[2:]
        
        return command
    
    def _execute_single_command(self, command: Dict[str, Any]) -> CommandResult:
        """
        Выполняет одну команду
        """
        command_type = command.get("type")
        
        if command_type == "create":
            return self._create_object(command)
        elif command_type == "set":
            return self._set_object_property(command)
        elif command_type == "query":
            return self._query_objects(command)
        elif command_type == "count":
            return self._count_formula(command)
        elif command_type == "resolve":
            return self._resolve_reference(command)
        
        return CommandResult(success=False, message=f"Неизвестная команда: {command_type}")
    
    def _create_object(self, command: Dict[str, Any]) -> CommandResult:
        """
        Создает объект
        """
        object_type = command.get("object_type", "object")
        properties = command.get("properties", {})
        
        # Генерируем ID
        obj_id = f"{object_type}_{self.system_state.object_counter}"
        self.system_state.object_counter += 1
        
        # Определяем тип объекта
        if object_type.startswith("$"):
            obj_type = ObjectType.IT if object_type.startswith("$it") else ObjectType.QUESTION
        else:
            obj_type = self._determine_object_type(object_type)
        
        # Создаем объект
        obj = ObjectNode(
            id=obj_id,
            name=properties.get("name", object_type),
            object_type=obj_type
        )
        
        # Добавляем свойства
        for prop_name, prop_value in properties.items():
            if prop_name != "name":
                obj.properties[prop_name] = ObjectProperty(
                    name=prop_name,
                    value=prop_value
                )
        
        # Добавляем в систему
        self.system_state.objects[obj_id] = obj
        
        return CommandResult(
            success=True,
            message=f"Создан объект {obj.name}",
            created_objects=[obj]
        )
    
    def _set_object_property(self, command: Dict[str, Any]) -> CommandResult:
        """
        Устанавливает свойство объекта
        """
        obj_name = command.get("object")
        prop_name = command.get("property")
        prop_value = command.get("value")
        
        # Находим объект
        target_obj = self._find_object_by_name(obj_name)
        if not target_obj:
            return CommandResult(
                success=False,
                message=f"Объект {obj_name} не найден"
            )
        
        # Устанавливаем свойство
        target_obj.properties[prop_name] = ObjectProperty(
            name=prop_name,
            value=prop_value
        )
        target_obj.modified_at = datetime.now()
        
        return CommandResult(
            success=True,
            message=f"Свойство {prop_name} объекта {obj_name} установлено",
            modified_objects=[target_obj]
        )
    
    def _query_objects(self, command: Dict[str, Any]) -> CommandResult:
        """
        Выполняет запрос к объектам
        """
        obj_name = command.get("object")
        prop_name = command.get("property")
        
        # Находим объект
        target_obj = self._find_object_by_name(obj_name)
        if not target_obj:
            return CommandResult(
                success=False,
                message=f"Объект {obj_name} не найден"
            )
        
        # Получаем значение свойства
        if prop_name in target_obj.properties:
            value = target_obj.properties[prop_name].value
            return CommandResult(
                success=True,
                message=f"Значение {prop_name} объекта {obj_name}: {value}",
                computed_values={"query_result": value}
            )
        else:
            return CommandResult(
                success=False,
                message=f"Свойство {prop_name} не найдено у объекта {obj_name}"
            )
    
    def _count_formula(self, command: Dict[str, Any]) -> CommandResult:
        """
        Вычисляет формулу
        """
        formula = command.get("formula", "")
        
        try:
            result, success = self.formula_evaluator.evaluate_formula(formula, None)
            if success:
                return CommandResult(
                    success=True,
                    message=f"Результат вычисления: {result}",
                    computed_values={"formula_result": result}
                )
            else:
                return CommandResult(
                    success=False,
                    message=f"Не удалось вычислить формулу: {result}"
                )
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка вычисления формулы: {str(e)}"
            )
    
    def _resolve_reference(self, command: Dict[str, Any]) -> CommandResult:
        """
        Разрешает ссылку на объект
        """
        reference = command.get("reference", "")
        resolution = command.get("resolution", "")
        
        # Находим объект по ссылке
        target_obj = self._resolve_reference_internal(reference)
        if not target_obj:
            return CommandResult(
                success=False,
                message=f"Ссылка {reference} не найдена"
            )
        
        # Применяем разрешение
        if "name" in resolution:
            target_obj.name = resolution["name"]
            target_obj.modified_at = datetime.now()
        
        return CommandResult(
            success=True,
            message=f"Ссылка {reference} разрешена",
            modified_objects=[target_obj]
        )
    
    def _resolve_reference_internal(self, reference: str) -> Optional[ObjectNode]:
        """
        Разрешает ссылку на объект
        """
        if reference == "я":
            return self._find_current_user()
        elif reference in ["он", "она"]:
            return self._find_last_mentioned_person()
        elif reference == "это":
            return self._find_last_created_object()
        else:
            return self._find_object_by_name(reference)
    
    def _find_current_user(self) -> Optional[ObjectNode]:
        """
        Находит текущего пользователя
        """
        if self.system_state.context.current_user:
            return self._find_object_by_name(self.system_state.context.current_user)
        return None
    
    def _find_last_mentioned_person(self) -> Optional[ObjectNode]:
        """
        Находит последнего упомянутого человека
        """
        for obj_id in reversed(self.system_state.context.last_mentioned_objects):
            obj = self.system_state.objects.get(obj_id)
            if obj and obj.object_type == ObjectType.PERSON:
                return obj
        return None
    
    def _find_last_created_object(self) -> Optional[ObjectNode]:
        """
        Находит последний созданный объект
        """
        if not self.system_state.objects:
            return None
        
        return max(
            self.system_state.objects.values(),
            key=lambda x: x.created_at
        )
    
    def _find_object_by_name(self, name: str) -> Optional[ObjectNode]:
        """
        Находит объект по имени
        """
        for obj in self.system_state.objects.values():
            if obj.name == name:
                return obj
        return None
    
    def _determine_object_type(self, object_name: str) -> ObjectType:
        """
        Определяет тип объекта по имени
        """
        if object_name.lower() in ["person", "человек", "я", "он", "она"]:
            return ObjectType.PERSON
        elif object_name.lower() in ["apple", "яблоко", "яблоки"]:
            return ObjectType.OBJECT
        elif object_name.lower() in ["transfer", "передача"]:
            return ObjectType.TRANSFER
        else:
            return ObjectType.OBJECT
    
    def _update_context(self, result: CommandResult):
        """
        Обновляет контекст разговора
        """
        # Добавляем созданные объекты в историю
        for obj in result.created_objects:
            self.system_state.context.last_mentioned_objects.append(obj.id)
        
        # Ограничиваем историю
        if len(self.system_state.context.last_mentioned_objects) > 10:
            self.system_state.context.last_mentioned_objects = self.system_state.context.last_mentioned_objects[-10:]
    
    def _analyze_uncertainties(self, result: CommandResult):
        """
        Анализирует неопределенности и создает вопросы
        """
        for obj in result.created_objects:
            if obj.name.startswith("$"):
                # Создаем вопрос для неопределенности
                question_text = f"Что означает '{obj.name}'?"
                result.questions_generated.append(question_text)
    
    def get_system_tree(self) -> Dict[str, Any]:
        """
        Возвращает дерево объектов системы
        """
        objects = {}
        for obj_id, obj in self.system_state.objects.items():
            objects[obj_id] = {
                "name": obj.name,
                "type": obj.object_type.value,
                "state": obj.state.value,
                "properties": {name: prop.value for name, prop in obj.properties.items()}
            }
        
        return {
            "total_objects": len(objects),
            "object_types": list(set(obj["type"] for obj in objects.values())),
            "objects": objects,
            "context": {
                "current_user": self.system_state.context.current_user,
                "last_mentioned": self.system_state.context.last_mentioned_objects,
                "unresolved_questions": self.system_state.context.unresolved_questions
            }
        }

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
        Находит объект по имени
        """
        for obj in self.system_state.objects.values():
            if obj.name == name:
                return obj
        return None
    
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
        for op in ['+', '-', '*', '/']:
            if op in expression:
                left, right = expression.split(op, 1)
                left_val = self._evaluate_expression(left)
                right_val = self._evaluate_expression(right)
                return self.operators[op](left_val, right_val)
        
        # Если это число
        try:
            return float(expression)
        except ValueError:
            # Если это строка
            return expression 