from neo4j import GraphDatabase, exceptions # type: ignore
from typing import List, Dict, Any, Optional
import os # Добавлен os для корректной работы NEO4J_URI

# TODO: Конфигурация подключения к Neo4j должна быть вынесена
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

class KnowledgeBaseManager:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("Successfully connected to Neo4j Knowledge Base.")
            self._ensure_constraints_and_indexes()
        except exceptions.AuthError as e:
            print(f"FATAL: Neo4j Authentication Error: {e}. URI: {uri}")
        except exceptions.ServiceUnavailable as e:
            print(f"FATAL: Neo4j Service Unavailable: {e}. URI: {uri}")
        except Exception as e:
            print(f"FATAL: Could not connect to Neo4j: {e}")

    def _ensure_constraints_and_indexes(self):
        if not self.driver: return
        queries = [
            "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT test_concept_name_unique IF NOT EXISTS FOR (tc:TestConcept) REQUIRE tc.name IS UNIQUE", # Для тестов
            "CREATE CONSTRAINT test_entity_name_unique IF NOT EXISTS FOR (te:TestEntity) REQUIRE te.name IS UNIQUE", # Для тестов
            "CREATE CONSTRAINT test_attribute_name_unique IF NOT EXISTS FOR (ta:TestAttribute) REQUIRE ta.name IS UNIQUE", # Для тестов
            "CREATE INDEX concept_name_index IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        ]
        with self.driver.session() as session:
            for query in queries:
                try:
                    session.run(query)
                except exceptions.ClientError as e:
                    if "already exists" not in str(e).lower() and "already been created" not in str(e).lower():
                        print(f"Warning: Failed constraint/index: {query}. Error: {e}")
    def close(self):
        if self.driver: self.driver.close(); print("Neo4j connection closed.")

    def add_node(self, label: str, properties: Dict[str, Any]) -> Optional[str]:
        if not self.driver: return None
        prop_str = ", ".join([f"`{key}`: ${key}" for key in properties.keys()]) # Экранируем ключи
        query = f"MERGE (n:{label} {{name: $name}}) SET n += {{ {prop_str} }} RETURN elementId(n) AS id"
        # Используем MERGE чтобы не создавать дубликаты по name, а обновлять свойства
        # Это предполагает, что 'name' является ключевым идентификатором.

        params_with_name = properties.copy()
        if 'name' not in params_with_name and 'id' in params_with_name : # Если id есть, а name нет
             params_with_name['name'] = params_with_name['id'] # Используем id как name для MERGE
        elif 'name' not in params_with_name:
            print(f"Error adding node: 'name' property is required for MERGE strategy. Properties: {properties}")
            return None

        try:
            with self.driver.session() as session:
                result = session.run(query, **params_with_name)
                record = result.single()
                return record["id"] if record else None
        except Exception as e:
            print(f"Error adding/merging node ({label}, {properties}): {e}")
            return None

    def add_relationship(self, from_node_label: str, from_node_name: str,
                         to_node_label: str, to_node_name: str,
                         relationship_type: str, rel_props: Optional[Dict[str, Any]] = None) -> bool:
        if not self.driver: return False

        rel_type_upper = relationship_type.upper()
        query = f"""
        MATCH (a:{from_node_label} {{name: $from_name}}), (b:{to_node_label} {{name: $to_name}})
        MERGE (a)-[r:{rel_type_upper}]->(b)
        """
        params = {"from_name": from_node_name, "to_name": to_node_name}

        if rel_props:
            set_clauses = []
            for key, value in rel_props.items():
                set_clauses.append(f"r.`{key}` = ${key}") # Экранируем ключи свойств
                params[key] = value
            if set_clauses:
                query += " SET " + ", ".join(set_clauses)

        try:
            with self.driver.session() as session:
                session.run(query, **params)
            return True
        except Exception as e:
            print(f"Error adding relationship {rel_type_upper} from {from_node_name} to {to_node_name}: {e}")
            return False

    def execute_cypher_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.driver: return []
        results = []
        try:
            with self.driver.session() as session:
                response = session.run(query, parameters)
                for record in response:
                    results.append(record.data())
        except Exception as e:
            print(f"Error executing Cypher query '{query[:100]}...': {e}")
        return results

    def get_isa_parents(self, node_name: str, node_label: str = "Entity") -> List[str]:
        if not self.driver: return []
        query = f"""
        MATCH (n:{node_label} {{name: $node_name}})-[:IS_A*]->(parent:Concept)
        RETURN DISTINCT parent.name AS parent_concept_name
        """
        results = self.execute_cypher_query(query, {"node_name": node_name})
        parent_names = [r["parent_concept_name"] for r in results if "parent_concept_name" in r]
        print(f"KB_INFO: ISA Parents for '{node_name}' ({node_label}): {parent_names}")
        return parent_names

    def get_entity_properties(self, entity_name: str, entity_label: str = "Entity") -> Dict[str, Any]:
        if not self.driver: return {}
        properties = {}
        query_node_props = f"MATCH (n:{entity_label} {{name: $node_name}}) RETURN properties(n) AS props"
        node_results = self.execute_cypher_query(query_node_props, {"node_name": entity_name})
        if node_results and node_results[0].get("props"):
            properties.update(node_results[0]["props"])

        query_rel_props = f"""
        MATCH (n:{entity_label} {{name: $node_name}})-[r:HAS_PROPERTY]->(a:Attribute)
        WHERE r.value IS NOT NULL AND a.name IS NOT NULL
        RETURN a.name AS attribute_name, r.value AS attribute_value
        """
        rel_results = self.execute_cypher_query(query_rel_props, {"node_name": entity_name})
        for record in rel_results:
            if "attribute_name" in record and "attribute_value" in record:
                properties[record["attribute_name"]] = record["attribute_value"]

        if properties: print(f"KB_INFO: Properties for '{entity_name}' ({entity_label}): {properties}")
        else: print(f"KB_INFO: No props found for '{entity_name}' ({entity_label}).")
        return properties

    def find_related_entities(self, node_name: str, relationship_type: str,
                              direction: str = "OUTGOING",
                              node_label: str = "Entity",
                              related_node_label: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.driver: return []
        rel_type_upper = relationship_type.upper()
        arrow_left, arrow_right = "", ""
        if direction.upper() == "OUTGOING": arrow_right = "->"
        elif direction.upper() == "INCOMING": arrow_left = "<-"

        related_label_str = f":{related_node_label}" if related_node_label else ""

        query = f"""
        MATCH (n:{node_label} {{name: $node_name}}){arrow_left}-[:{rel_type_upper}]-{arrow_right}(related{related_label_str})
        RETURN properties(related) AS related_props, elementId(related) as related_id
        """
        results = self.execute_cypher_query(query, {"node_name": node_name})

        related_entities = []
        for record in results:
            props = record.get("related_props", {})
            if "related_id" in record: props["_element_id"] = record["related_id"]
            related_entities.append(props)

        print(f"KB_INFO: Found {len(related_entities)} for '{node_name}' via '{direction} {rel_type_upper}' to '{related_label_str if related_label_str else 'any'}'.")
        return related_entities

    def check_relationship_exists(self, from_name: str, rel_type: str, to_name: str,
                                  from_label: str = "Entity", to_label: str = "Entity", direction: str = "OUTGOING") -> bool:
        if not self.driver: return False
        rel_type_upper = rel_type.upper()
        arrow_left, arrow_right = "", ""
        if direction.upper() == "OUTGOING": arrow_right = "->"
        elif direction.upper() == "INCOMING": arrow_left = "<-"
        # Если BOTH, стрелки остаются пустыми, Cypher поймет как ненаправленную связь для MATCH

        query = f"""
        MATCH (a:{from_label} {{name: $from_name}}){arrow_left}-[:{rel_type_upper}]{arrow_right}(b:{to_label} {{name: $to_name}})
        RETURN count(b) > 0 AS exists
        """
        # count(b) вместо count(r) чтобы не создавать лишние отношения если их нет, а просто проверить наличие пути
        results = self.execute_cypher_query(query, {"from_name": from_name, "to_name": to_name})

        exists_val = results[0]["exists"] if results and "exists" in results[0] else False
        print(f"KB_INFO: Rel {from_name}{arrow_left}-[{rel_type_upper}]-{arrow_right}{to_name} exists: {exists_val}")
        return exists_val

if __name__ == '__main__':
    print("Initializing KnowledgeBaseManager...")
    kb_manager = KnowledgeBaseManager()
    if kb_manager.driver:
        print("\n--- Clearing Potential Test Data (if any) ---")
        # Очищаем узлы, созданные в предыдущих тестах, чтобы избежать ошибок уникальности
        cleanup_labels = ["TestEntity", "TestConcept", "TestAttribute"]
        for label in cleanup_labels:
             kb_manager.execute_cypher_query(f"MATCH (n:{label}) DETACH DELETE n")
        print("Test data cleared.")

        # Добавление узлов для теста
        kb_manager.add_node("TestConcept", {"name": "Animal"})
        kb_manager.add_node("TestConcept", {"name": "Mammal"})
        kb_manager.add_node("TestConcept", {"name": "Carnivore"})
        kb_manager.add_node("TestEntity", {"name": "Tiger", "type": "Bengal Tiger", "can_roar": True})
        kb_manager.add_node("TestEntity", {"name": "Deer", "type": "Spotted Deer", "eats_grass": True})
        kb_manager.add_node("TestConcept", {"name": "Jungle", "type": "Habitat"})
        kb_manager.add_node("TestConcept", {"name": "Forest", "type": "Habitat"})
        kb_manager.add_node("TestEntity", {"name": "Apple", "category": "Fruit"})
        kb_manager.add_node("TestConcept", {"name": "FruitTree", "species": "Malus domestica"})
        kb_manager.add_node("TestAttribute", {"name": "fur_color"})
        kb_manager.add_node("TestAttribute", {"name": "color"})
        kb_manager.add_node("TestAttribute", {"name": "taste"})

        # Отношения IS_A
        kb_manager.add_relationship("TestEntity", "Tiger", "TestConcept", "Carnivore", "IS_A")
        kb_manager.add_relationship("TestEntity", "Tiger", "TestConcept", "Mammal", "IS_A")
        kb_manager.add_relationship("TestConcept", "Carnivore", "TestConcept", "Animal", "IS_A")
        kb_manager.add_relationship("TestConcept", "Mammal", "TestConcept", "Animal", "IS_A")

        # Отношения HAS_PROPERTY
        kb_manager.add_relationship("TestEntity", "Tiger", "TestAttribute", "fur_color", "HAS_PROPERTY", {"value": "orange and black stripes"})
        kb_manager.add_relationship("TestEntity", "Apple", "TestAttribute", "color", "HAS_PROPERTY", {"value": "red"})
        kb_manager.add_relationship("TestEntity", "Apple", "TestAttribute", "taste", "HAS_PROPERTY", {"value": "sweet"})

        # Отношения для find_related_entities и check_relationship_exists
        kb_manager.add_relationship("TestEntity", "Tiger", "TestEntity", "Deer", "EATS", {"frequency": "often"})
        kb_manager.add_relationship("TestEntity", "Tiger", "TestConcept", "Jungle", "LIVES_IN")
        kb_manager.add_relationship("TestEntity", "Deer", "TestConcept", "Forest", "LIVES_IN")
        kb_manager.add_relationship("TestEntity", "Apple", "TestConcept", "FruitTree", "GROWS_ON")

        print("\n--- Testing get_isa_parents ---")
        tiger_parents = kb_manager.get_isa_parents("Tiger", "TestEntity")
        assert all(p in tiger_parents for p in ["Carnivore", "Mammal", "Animal"])

        print("\n--- Testing get_entity_properties ---")
        tiger_props = kb_manager.get_entity_properties("Tiger", "TestEntity")
        assert tiger_props.get("type") == "Bengal Tiger" and tiger_props.get("can_roar") and tiger_props.get("fur_color") == "orange and black stripes"
        apple_props = kb_manager.get_entity_properties("Apple", "TestEntity")
        assert apple_props.get("category") == "Fruit" and apple_props.get("color") == "red" and apple_props.get("taste") == "sweet"

        print("\n--- Testing find_related_entities ---")
        eats_what = kb_manager.find_related_entities("Tiger", "EATS", "OUTGOING", "TestEntity", "TestEntity")
        assert len(eats_what) == 1 and eats_what[0].get("name") == "Deer"
        who_eats_deer = kb_manager.find_related_entities("Deer", "EATS", "INCOMING", "TestEntity", "TestEntity")
        assert len(who_eats_deer) == 1 and who_eats_deer[0].get("name") == "Tiger"
        lives_where_tiger = kb_manager.find_related_entities("Tiger", "LIVES_IN", node_label="TestEntity", related_node_label="TestConcept")
        assert len(lives_where_tiger) == 1 and lives_where_tiger[0].get("name") == "Jungle"

        print("\n--- Testing check_relationship_exists ---")
        assert kb_manager.check_relationship_exists("Tiger", "EATS", "Deer", "TestEntity", "TestEntity") is True
        assert kb_manager.check_relationship_exists("Deer", "EATS", "Tiger", "TestEntity", "TestEntity", direction="OUTGOING") is False # Deer не ест тигра
        assert kb_manager.check_relationship_exists("Deer", "EATS", "Tiger", "TestEntity", "TestEntity", direction="INCOMING") is True # Тигр ест оленя (входящая для оленя)
        assert kb_manager.check_relationship_exists("Tiger", "LIVES_IN", "Jungle", "TestEntity", "TestConcept") is True
        assert kb_manager.check_relationship_exists("Tiger", "EATS", "Unicorn", "TestEntity", "TestEntity") is False

        kb_manager.close()
    else:
        print("Could not run tests as Neo4j driver is not initialized.")
    print("\nKnowledgeBaseManager script finished.")
