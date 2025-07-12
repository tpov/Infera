from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from enum import Enum
import re

class CommandLevel(Enum):
    ABSTRACT = "abstract"      # Высокоуровневые команды
    LOGICAL = "logical"        # Логические связи
    CONCRETE = "concrete"      # Конкретные действия
    DETAILED = "detailed"      # Детальные операции

@dataclass
class ContextNode:
    """Узел контекстного дерева"""
    id: str
    concept: str
    level: CommandLevel
    properties: Dict[str, Any] = field(default_factory=dict)
    children: List['ContextNode'] = field(default_factory=list)
    parent: Optional['ContextNode'] = None
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_child(self, child: 'ContextNode'):
        """Добавляет дочерний узел"""
        child.parent = self
        self.children.append(child)
    
    def get_path(self) -> List[str]:
        """Возвращает путь от корня до узла"""
        path = [self.concept]
        current = self.parent
        while current:
            path.append(current.concept)
            current = current.parent
        return list(reversed(path))

@dataclass
class HierarchicalCommand:
    """Иерархическая команда"""
    level: CommandLevel
    command: str
    context_path: List[str]
    confidence: float
    dependencies: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)

class ContextTreeBuilder:
    """Строитель контекстного дерева"""
    
    def __init__(self):
        self.root = ContextNode("root", "system", CommandLevel.ABSTRACT)
        self.node_counter = 0
        
    def build_context_tree(self, user_query: str) -> ContextNode:
        """Строит контекстное дерево из запроса пользователя"""
        
        # Анализируем запрос на разных уровнях
        abstract_concepts = self._extract_abstract_concepts(user_query)
        logical_relations = self._extract_logical_relations(user_query)
        concrete_actions = self._extract_concrete_actions(user_query)
        detailed_operations = self._extract_detailed_operations(user_query)
        
        # Строим дерево
        self._build_abstract_level(abstract_concepts)
        self._build_logical_level(logical_relations)
        self._build_concrete_level(concrete_actions)
        self._build_detailed_level(detailed_operations)
        
        return self.root
    
    def _extract_abstract_concepts(self, query: str) -> List[str]:
        """Извлекает абстрактные концепции"""
        concepts = []
        query_lower = query.lower()
        
        # Абстрактные концепции
        abstract_keywords = [
            'система', 'процесс', 'функция', 'задача', 'цель', 'результат',
            'состояние', 'отношение', 'связь', 'взаимодействие', 'динамика',
            'структура', 'организация', 'управление', 'контроль', 'анализ'
        ]
        
        for keyword in abstract_keywords:
            if keyword in query_lower:
                concepts.append(keyword)
        
        return concepts
    
    def _extract_logical_relations(self, query: str) -> List[Dict[str, Any]]:
        """Извлекает логические отношения"""
        relations = []
        query_lower = query.lower()
        
        # Логические операторы
        logical_patterns = [
            (r'если\s+(\w+)\s+то\s+(\w+)', 'conditional'),
            (r'(\w+)\s+и\s+(\w+)', 'conjunction'),
            (r'(\w+)\s+или\s+(\w+)', 'disjunction'),
            (r'(\w+)\s+зависит\s+от\s+(\w+)', 'dependency'),
            (r'(\w+)\s+влияет\s+на\s+(\w+)', 'influence')
        ]
        
        for pattern, relation_type in logical_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                relations.append({
                    'type': relation_type,
                    'elements': list(match),
                    'confidence': 0.8
                })
        
        return relations
    
    def _extract_concrete_actions(self, query: str) -> List[str]:
        """Извлекает конкретные действия"""
        actions = []
        query_lower = query.lower()
        
        # Конкретные действия
        action_keywords = [
            'создать', 'удалить', 'изменить', 'установить', 'настроить',
            'включить', 'выключить', 'запустить', 'остановить', 'проверить',
            'измерить', 'считать', 'записать', 'прочитать', 'передать'
        ]
        
        for keyword in action_keywords:
            if keyword in query_lower:
                actions.append(keyword)
        
        return actions
    
    def _extract_detailed_operations(self, query: str) -> List[Dict[str, Any]]:
        """Извлекает детальные операции"""
        operations = []
        query_lower = query.lower()
        
        # Детальные операции с параметрами
        operation_patterns = [
            (r'(\w+)\s+=\s+(\d+)', 'assignment'),
            (r'(\w+)\s+в\s+(\w+)', 'location'),
            (r'(\w+)\s+со\s+значением\s+(\w+)', 'value_setting'),
            (r'(\w+)\s+количество\s+(\d+)', 'quantity_setting')
        ]
        
        for pattern, operation_type in operation_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                operations.append({
                    'type': operation_type,
                    'target': match[0],
                    'value': match[1],
                    'confidence': 0.9
                })
        
        return operations
    
    def _build_abstract_level(self, concepts: List[str]):
        """Строит абстрактный уровень дерева"""
        for concept in concepts:
            node = ContextNode(
                f"abstract_{self.node_counter}",
                concept,
                CommandLevel.ABSTRACT,
                confidence=0.8
            )
            self.root.add_child(node)
            self.node_counter += 1
    
    def _build_logical_level(self, relations: List[Dict[str, Any]]):
        """Строит логический уровень дерева"""
        for relation in relations:
            # Находим подходящий родительский узел
            parent = self._find_suitable_parent(relation['elements'][0])
            if parent:
                node = ContextNode(
                    f"logical_{self.node_counter}",
                    f"{relation['type']}_{relation['elements'][0]}_{relation['elements'][1]}",
                    CommandLevel.LOGICAL,
                    properties={'relation_type': relation['type']},
                    confidence=relation['confidence']
                )
                parent.add_child(node)
                self.node_counter += 1
    
    def _build_concrete_level(self, actions: List[str]):
        """Строит конкретный уровень дерева"""
        for action in actions:
            # Находим подходящий родительский узел
            parent = self._find_suitable_parent(action)
            if parent:
                node = ContextNode(
                    f"concrete_{self.node_counter}",
                    action,
                    CommandLevel.CONCRETE,
                    confidence=0.9
                )
                parent.add_child(node)
                self.node_counter += 1
    
    def _build_detailed_level(self, operations: List[Dict[str, Any]]):
        """Строит детальный уровень дерева"""
        for operation in operations:
            # Находим подходящий родительский узел
            parent = self._find_suitable_parent(operation['target'])
            if parent:
                node = ContextNode(
                    f"detailed_{self.node_counter}",
                    f"{operation['type']}_{operation['target']}",
                    CommandLevel.DETAILED,
                    properties=operation,
                    confidence=operation['confidence']
                )
                parent.add_child(node)
                self.node_counter += 1
    
    def _find_suitable_parent(self, concept: str) -> Optional[ContextNode]:
        """Находит подходящий родительский узел"""
        # Ищем узел с похожей концепцией
        for node in self._traverse_tree(self.root):
            if concept.lower() in node.concept.lower() or node.concept.lower() in concept.lower():
                return node
        
        # Если не нашли, возвращаем корень
        return self.root
    
    def _traverse_tree(self, node: ContextNode) -> List[ContextNode]:
        """Обходит дерево и возвращает все узлы"""
        nodes = [node]
        for child in node.children:
            nodes.extend(self._traverse_tree(child))
        return nodes

class HierarchicalCommandGenerator:
    """Генератор иерархических команд"""
    
    def __init__(self):
        self.tree_builder = ContextTreeBuilder()
        self.command_templates = self._load_command_templates()
    
    def generate_commands(self, user_query: str) -> List[HierarchicalCommand]:
        """Генерирует иерархические команды из запроса пользователя"""
        
        # Строим контекстное дерево
        context_tree = self.tree_builder.build_context_tree(user_query)
        
        # Генерируем команды для каждого уровня
        commands = []
        
        # Абстрактный уровень
        abstract_commands = self._generate_abstract_commands(context_tree)
        commands.extend(abstract_commands)
        
        # Логический уровень
        logical_commands = self._generate_logical_commands(context_tree)
        commands.extend(logical_commands)
        
        # Конкретный уровень
        concrete_commands = self._generate_concrete_commands(context_tree)
        commands.extend(concrete_commands)
        
        # Детальный уровень
        detailed_commands = self._generate_detailed_commands(context_tree)
        commands.extend(detailed_commands)
        
        return commands
    
    def _generate_abstract_commands(self, tree: ContextNode) -> List[HierarchicalCommand]:
        """Генерирует абстрактные команды"""
        commands = []
        
        for node in self._get_nodes_by_level(tree, CommandLevel.ABSTRACT):
            if node.concept == "система":
                command = HierarchicalCommand(
                    level=CommandLevel.ABSTRACT,
                    command="create system quantity 1",
                    context_path=node.get_path(),
                    confidence=node.confidence
                )
                commands.append(command)
            
            elif node.concept == "процесс":
                command = HierarchicalCommand(
                    level=CommandLevel.ABSTRACT,
                    command="create process quantity 1",
                    context_path=node.get_path(),
                    confidence=node.confidence
                )
                commands.append(command)
        
        return commands
    
    def _generate_logical_commands(self, tree: ContextNode) -> List[HierarchicalCommand]:
        """Генерирует логические команды"""
        commands = []
        
        for node in self._get_nodes_by_level(tree, CommandLevel.LOGICAL):
            relation_type = node.properties.get('relation_type', '')
            
            if relation_type == 'conditional':
                command = HierarchicalCommand(
                    level=CommandLevel.LOGICAL,
                    command="if condition then action",
                    context_path=node.get_path(),
                    confidence=node.confidence
                )
                commands.append(command)
            
            elif relation_type == 'dependency':
                command = HierarchicalCommand(
                    level=CommandLevel.LOGICAL,
                    command="set dependency source target",
                    context_path=node.get_path(),
                    confidence=node.confidence
                )
                commands.append(command)
        
        return commands
    
    def _generate_concrete_commands(self, tree: ContextNode) -> List[HierarchicalCommand]:
        """Генерирует конкретные команды"""
        commands = []
        
        for node in self._get_nodes_by_level(tree, CommandLevel.CONCRETE):
            if node.concept == "создать":
                command = HierarchicalCommand(
                    level=CommandLevel.CONCRETE,
                    command="create object quantity 1",
                    context_path=node.get_path(),
                    confidence=node.confidence
                )
                commands.append(command)
            
            elif node.concept == "удалить":
                command = HierarchicalCommand(
                    level=CommandLevel.CONCRETE,
                    command="delete object",
                    context_path=node.get_path(),
                    confidence=node.confidence
                )
                commands.append(command)
            
            elif node.concept == "изменить":
                command = HierarchicalCommand(
                    level=CommandLevel.CONCRETE,
                    command="modify object property value",
                    context_path=node.get_path(),
                    confidence=node.confidence
                )
                commands.append(command)
        
        return commands
    
    def _generate_detailed_commands(self, tree: ContextNode) -> List[HierarchicalCommand]:
        """Генерирует детальные команды"""
        commands = []
        
        for node in self._get_nodes_by_level(tree, CommandLevel.DETAILED):
            operation_type = node.properties.get('type', '')
            target = node.properties.get('target', '')
            value = node.properties.get('value', '')
            
            if operation_type == 'assignment':
                command = HierarchicalCommand(
                    level=CommandLevel.DETAILED,
                    command=f"set {target} value {value}",
                    context_path=node.get_path(),
                    confidence=node.confidence
                )
                commands.append(command)
            
            elif operation_type == 'quantity_setting':
                command = HierarchicalCommand(
                    level=CommandLevel.DETAILED,
                    command=f"set {target} quantity {value}",
                    context_path=node.get_path(),
                    confidence=node.confidence
                )
                commands.append(command)
        
        return commands
    
    def _get_nodes_by_level(self, tree: ContextNode, level: CommandLevel) -> List[ContextNode]:
        """Возвращает узлы определенного уровня"""
        nodes = []
        
        def traverse(node: ContextNode):
            if node.level == level:
                nodes.append(node)
            for child in node.children:
                traverse(child)
        
        traverse(tree)
        return nodes
    
    def _load_command_templates(self) -> Dict[str, List[str]]:
        """Загружает шаблоны команд"""
        return {
            CommandLevel.ABSTRACT.value: [
                "create system quantity {quantity}",
                "create process quantity {quantity}",
                "create function quantity {quantity}"
            ],
            CommandLevel.LOGICAL.value: [
                "if {condition} then {action}",
                "set dependency {source} {target}",
                "create relation {type} {from} {to}"
            ],
            CommandLevel.CONCRETE.value: [
                "create {object} quantity {quantity}",
                "delete {object}",
                "modify {object} {property} {value}"
            ],
            CommandLevel.DETAILED.value: [
                "set {object} {property} value {value}",
                "set {object} quantity {value}",
                "set {object} position {x} {y} {z}"
            ]
        }

class ContextAwareController:
    """Контроллер с учетом контекста"""
    
    def __init__(self):
        self.command_generator = HierarchicalCommandGenerator()
        self.context_history = []
        self.current_context = None
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Обрабатывает запрос с учетом контекста"""
        
        # Генерируем иерархические команды
        commands = self.command_generator.generate_commands(user_query)
        
        # Сортируем команды по уровню (от абстрактного к детальному)
        commands.sort(key=lambda x: list(CommandLevel).index(x.level))
        
        # Выполняем команды
        results = []
        for command in commands:
            result = self._execute_hierarchical_command(command)
            results.append(result)
        
        # Обновляем контекст
        self._update_context(user_query, commands, results)
        
        return {
            "user_query": user_query,
            "generated_commands": commands,
            "execution_results": results,
            "context_path": self._get_context_path(),
            "confidence_level": self._calculate_confidence(commands)
        }
    
    def _execute_hierarchical_command(self, command: HierarchicalCommand) -> Dict[str, Any]:
        """Выполняет иерархическую команду"""
        # Здесь должна быть интеграция с вашим существующим контроллером
        return {
            "command": command.command,
            "level": command.level.value,
            "context_path": command.context_path,
            "confidence": command.confidence,
            "success": True,  # Упрощенно
            "message": f"Выполнена команда уровня {command.level.value}"
        }
    
    def _update_context(self, query: str, commands: List[HierarchicalCommand], results: List[Dict[str, Any]]):
        """Обновляет контекст"""
        context_entry = {
            "query": query,
            "commands": commands,
            "results": results,
            "timestamp": datetime.now()
        }
        
        self.context_history.append(context_entry)
        
        # Ограничиваем историю
        if len(self.context_history) > 10:
            self.context_history = self.context_history[-10:]
    
    def _get_context_path(self) -> List[str]:
        """Возвращает путь контекста"""
        if not self.context_history:
            return []
        
        # Берем концепции из последних команд
        path = []
        for entry in self.context_history[-3:]:  # Последние 3 записи
            for command in entry["commands"]:
                path.extend(command.context_path)
        
        return list(set(path))  # Убираем дубликаты
    
    def _calculate_confidence(self, commands: List[HierarchicalCommand]) -> float:
        """Вычисляет общую уверенность"""
        if not commands:
            return 0.0
        
        total_confidence = sum(cmd.confidence for cmd in commands)
        return total_confidence / len(commands)

def test_hierarchical_system():
    """Тестирует иерархическую систему команд"""
    controller = ContextAwareController()
    
    test_queries = [
        "Создать систему автоматизации с 5 датчиками",
        "Если температура высокая, то включить вентиляцию",
        "Установить датчик в позицию x=10, y=20, z=0"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Запрос: {query}")
        print(f"{'='*60}")
        
        result = controller.process_query(query)
        
        print(f"Сгенерированные команды:")
        for cmd in result["generated_commands"]:
            print(f"  [{cmd.level.value}] {cmd.command} (уверенность: {cmd.confidence:.2f})")
        
        print(f"Путь контекста: {' -> '.join(result['context_path'])}")
        print(f"Общая уверенность: {result['confidence_level']:.2f}")

if __name__ == "__main__":
    test_hierarchical_system()