import re
from typing import Dict, Any, Optional, List

class CommandParser:
    def parse(self, command_str: str) -> Optional[Dict[str, Any]]:
        """
        Парсит строковую команду и возвращает структурированный словарь.
        """
        command_str = command_str.strip()
        parts = command_str.split(' ', 1)
        if not parts:
            return None

        command_verb = parts[0].upper()
        args_str = parts[1] if len(parts) > 1 else ""

        if command_verb == 'CREATE':
            return self._parse_create(args_str)
        elif command_verb == 'LINK':
            return self._parse_link(args_str)
        elif command_verb == 'UPDATE':
            return self._parse_update(args_str)
        elif command_verb == 'QUERY':
            return self._parse_query(args_str)
        else:
            print(f"Warning: Unknown command verb '{command_verb}'")
            return None

    def _parse_create(self, args_str: str) -> Optional[Dict[str, Any]]:
        """
        Парсит аргументы для команды CREATE.
        Формат: <type> <name> [WITH <key1>=<value1> ...]
        """
        pattern = re.compile(r"(\w+)\s+([\w_]+)(?:\s+WITH\s+(.*))?")
        match = pattern.match(args_str)
        if not match:
            return None

        obj_type, obj_name, with_block = match.groups()

        command = {
            'verb': 'CREATE',
            'type': obj_type,
            'name': obj_name,
            'attributes': {}
        }

        if with_block:
            # Парсим атрибуты
            # Простое разделение по пробелу, но обрабатывает строки в кавычках
            attr_pattern = re.compile(r"(\w+)=('[\w\s\d\.-]+'|[\w\d\.-]+)")
            for key, value in attr_pattern.findall(with_block):
                if value.startswith("'") and value.endswith("'"):
                    command['attributes'][key] = value[1:-1]
                elif '.' in value:
                    command['attributes'][key] = float(value)
                elif value.isdigit():
                    command['attributes'][key] = int(value)
                else:
                    command['attributes'][key] = value

        return command

    def _parse_link(self, args_str: str) -> Optional[Dict[str, Any]]:
        parts = args_str.split()
        if len(parts) != 3:
            return None
        return {'verb': 'LINK', 'source': parts[0], 'relation': parts[1], 'target': parts[2]}

    def _parse_update(self, args_str: str) -> Optional[Dict[str, Any]]:
        pattern = re.compile(r"([\w_]+)\s+SET\s+(\w+)=(.*)")
        match = pattern.match(args_str)
        if not match:
            return None
        name, key, value = match.groups()
        return {'verb': 'UPDATE', 'name': name, 'key': key, 'value': value.strip()}

    def _parse_query(self, args_str: str) -> Optional[Dict[str, Any]]:
        # Упрощенный парсер для QUERY
        parts = args_str.split(' COMPUTE ')
        query_part = parts[0]
        compute_part = parts[1] if len(parts) > 1 else None

        where_parts = query_part.split(' WHERE ')
        target = where_parts[0].strip()
        condition = where_parts[1].strip() if len(where_parts) > 1 else None

        command = {'verb': 'QUERY', 'target': target}
        if condition:
            command['where'] = condition
        if compute_part:
            command['compute'] = compute_part

        return command

import networkx as nx

class ProceduralController:
    def __init__(self):
        self.graph = nx.Graph()
        self.parser = CommandParser()
        self.node_counter = 0

    def execute(self, command_str: str):
        """
        Выполняет одну команду.
        """
        parsed_cmd = self.parser.parse(command_str)
        if not parsed_cmd:
            print(f"Error: Could not parse command: {command_str}")
            return

        verb = parsed_cmd['verb']
        if verb == 'CREATE':
            self._create(parsed_cmd)
        elif verb == 'LINK':
            self._link(parsed_cmd)
        elif verb == 'UPDATE':
            self._update(parsed_cmd)
        elif verb == 'QUERY':
            self._query(parsed_cmd)

    def _get_node_id_by_name(self, name: str) -> Optional[int]:
        # Возвращает первый найденный. Для уникальности имен это ок.
        for node_id, data in self.graph.nodes(data=True):
            if data.get('name') == name:
                return node_id
        return None

    def _create(self, cmd: Dict[str, Any]):
        node_name = cmd['name']
        if self._get_node_id_by_name(node_name) is not None:
            print(f"Warning: Node with name '{node_name}' already exists.")
            return

        node_id = self.node_counter
        self.node_counter += 1

        attributes = cmd['attributes']
        attributes['name'] = node_name
        attributes['type'] = cmd['type']

        self.graph.add_node(node_id, **attributes)
        print(f"Created node {node_id} with name '{node_name}' and attributes {attributes}")

    def _link(self, cmd: Dict[str, Any]):
        source_id = self._get_node_id_by_name(cmd['source'])
        target_id = self._get_node_id_by_name(cmd['target'])

        if source_id is None or target_id is None:
            print(f"Error: Cannot create link. One of the nodes not found: {cmd['source']} or {cmd['target']}")
            return

        self.graph.add_edge(source_id, target_id, relation=cmd['relation'])
        print(f"Linked {cmd['source']} and {cmd['target']} with relation '{cmd['relation']}'")


    def _update(self, cmd: Dict[str, Any]):
        node_id = self._get_node_id_by_name(cmd['name'])
        if node_id is None:
            print(f"Error: Node '{cmd['name']}' not found for update.")
            return

        key = cmd['key']
        value_str = cmd['value']

        # Обработка формул
        new_value = self._evaluate_formula(value_str, node_id, key)

        self.graph.nodes[node_id][key] = new_value
        print(f"Updated node '{cmd['name']}': set {key} to {new_value}")

    def _evaluate_formula(self, formula: str, context_node_id: int, context_key: str) -> Any:
        # PREVIOUS_VALUE
        if "PREVIOUS_VALUE" in formula:
            previous_value = self.graph.nodes[context_node_id].get(context_key, 0)
            # Безопасное выполнение eval
            formula = formula.replace("PREVIOUS_VALUE", str(previous_value))
            try:
                return eval(formula, {"__builtins__": {}}, {})
            except:
                return formula # Если вычисление не удалось, возвращаем как строку

        # Попытаемся преобразовать к числу
        try:
            return float(formula) if '.' in formula else int(formula)
        except ValueError:
            return formula

    def _query(self, cmd: Dict[str, Any]):
        target_type = cmd['target']

        # 1. Фильтрация нодов
        candidate_nodes = [
            (node_id, data) for node_id, data in self.graph.nodes(data=True)
            if data.get('type') == target_type
        ]

        # 2. Применение WHERE (упрощенно)
        if 'where' in cmd:
            key, value = cmd['where'].split('=')
            value = value.strip("'")
            candidate_nodes = [(node_id, data) for node_id, data in candidate_nodes if str(data.get(key)) == value]

        # 3. Применение COMPUTE
        if 'compute' in cmd:
            match = re.match(r"SUM\((\w+)\)", cmd['compute'])
            if match:
                attr_to_sum = match.group(1)
                total = sum(data.get(attr_to_sum, 0) for _, data in candidate_nodes)
                print(f"Query result: {total}")
            else:
                print(f"Query result: Unknown computation '{cmd['compute']}'")
        else:
            # Если нет COMPUTE, просто возвращаем найденные ноды
            print(f"Query result: {[data for _, data in candidate_nodes]}")


    def print_summary(self):
        print("\n--- Graph Summary ---")
        print(f"Nodes: {list(self.graph.nodes(data=True))}")
        print(f"Edges: {list(self.graph.edges(data=True))}")
        print("---------------------\n")


# Демонстрация работы контроллера
if __name__ == '__main__':
    controller = ProceduralController()

    commands = [
        "CREATE person user_1 WITH name='John'",
        "CREATE person user_2 WITH name='Alice'",
        "CREATE item apple_1 WITH owner='user_1' quantity=5 color='red'",
        "CREATE item apple_2 WITH owner='user_2' quantity=3 color='green'",
        "CREATE item apple_3 WITH owner='user_1' quantity=2 color='red'",

        # Запросы
        "QUERY person", # Найти всех людей
        "QUERY item WHERE owner='user_1'", # Найти все предметы юзера 1
        "QUERY item WHERE color='red' COMPUTE SUM(quantity)", # Сколько красных яблок?

        # Обновление с формулой
        "UPDATE apple_1 SET quantity=PREVIOUS_VALUE-1",
        "QUERY item WHERE name='apple_1'" # Проверить обновление
    ]

    for cmd in commands:
        print(f"\nExecuting: {cmd}")
        controller.execute(cmd)

    controller.print_summary()
