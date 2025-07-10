from neo4j import GraphDatabase, ManagedTransaction, Query
import numpy as np

# Конфигурация Neo4j - данные для AuraDB от пользователя
NEO4J_URI = "neo4j+s://b1142466.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "q6qBUrZTTY9BfMcxwym1PqFXnZdAwNiMkGJjKGibm20"

driver = None

def get_driver():
    global driver
    if driver is None:
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session(database="neo4j") as session: # Указываем database для Aura
                session.run("RETURN 1")
            print(f"Successfully connected to Neo4j AuraDB: {NEO4J_URI}")
        except Exception as e:
            print(f"Failed to connect to Neo4j AuraDB: {e}")
            print("Please ensure your AuraDB instance is running and credentials are correct.")
            driver = None
            raise
    return driver

def close_driver():
    global driver
    if driver:
        driver.close()
        driver = None
        print("Neo4j AuraDB connection closed.")

def ensure_constraints(tx: ManagedTransaction):
    constraint_name = "constraint_word_text_unique"
    # Пытаемся создать ограничение с IF NOT EXISTS - это наиболее безопасный способ
    # для разных версий Neo4j, включая Aura.
    try:
        # Этот синтаксис должен работать для Neo4j 4.x и 5.x (Aura обычно на свежих версиях)
        tx.run(f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (w:Word) REQUIRE w.text IS UNIQUE")
        print(f"Ensured constraint (requested name: '{constraint_name}') for :Word(text).")
    except Exception as e_if_not_exists:
        # Если CREATE CONSTRAINT IF NOT EXISTS не поддерживается или вызывает ошибку по другой причине
        # (например, в очень старых версиях или специфических конфигурациях)
        # print(f"DEBUG: CREATE CONSTRAINT IF NOT EXISTS failed: {e_if_not_exists}")
        # Попробуем проверить существование и создать, если нужно, старым способом
        try:
            # Проверяем существование ограничения (синтаксис может варьироваться)
            # Этот запрос более общий для проверки
            found = False
            # Сначала SHOW CONSTRAINT (для Neo4j 4.0+). YIELD может отличаться для версий.
            try:
                # Используем более простой SHOW CONSTRAINT без YIELD для большей совместимости начальной проверки
                constraints_result = list(tx.run("SHOW CONSTRAINTS"))
                for record in constraints_result:
                    # Ищем по сути: ограничение уникальности на :Word(text)
                    # 'description' или 'details' могут содержать эту информацию
                    desc = record.get("description", "")
                    name = record.get("name", "")
                    # print(f"DEBUG: Constraint found: Name='{name}', Description='{desc}'")
                    if "Word" in desc and "text" in desc and "UNIQUENESS" in desc:
                        found = True
                        print(f"Uniqueness constraint for :Word(text) already exists (found by SHOW CONSTRAINTS): {name}")
                        break
                    # Иногда в новых версиях поля могут называться labelsOrTypes и properties
                    if record.get("labelsOrTypes") == ["Word"] and record.get("properties") == ["text"] and record.get("type") == "UNIQUENESS":
                        found = True
                        print(f"Uniqueness constraint for :Word(text) already exists (found by SHOW CONSTRAINTS with properties): {name}")
                        break
            except Exception as e_show:
                # Если SHOW CONSTRAINTS не сработал, возможно, это старая версия
                # print(f"DEBUG: SHOW CONSTRAINTS command failed: {e_show}. Trying CALL db.constraints().")
                try:
                    constraints_result = list(tx.run("CALL db.constraints()"))
                    for record in constraints_result:
                        desc = record.get("description", "")
                        # print(f"DEBUG: Constraint found (db.constraints): Description='{desc}'")
                        if "Word" in desc and "text" in desc and "UNIQUE" in desc: # Ищем суть
                            found = True
                            print("Uniqueness constraint for :Word(text) already exists (found by CALL db.constraints()).")
                            break
                except Exception as e_db_constraints:
                    print(f"Neither SHOW CONSTRAINTS nor CALL db.constraints() worked. Assuming constraint needs creation. Error: {e_db_constraints}")

            if not found:
                # Если не найдено, пытаемся создать старым способом (без IF NOT EXISTS)
                # Это может вызвать ошибку, если ограничение уже есть, но мы ее перехватим
                try:
                    tx.run("CREATE CONSTRAINT ON (w:Word) ASSERT w.text IS UNIQUE")
                    print("Created uniqueness constraint for :Word(text) using older syntax.")
                except Exception as e_create_old:
                    if "already exists" in str(e_create_old).lower():
                        print("Uniqueness constraint for :Word(text) already exists (caught by older CREATE CONSTRAINT ON).")
                    else:
                        # Если ошибка не "already exists", то это проблема
                        print(f"Failed to create constraint using older syntax, and it wasn't 'already exists': {e_create_old}")
                        # Также покажем ошибку от IF NOT EXISTS, если она была
                        print(f"Original error from 'CREATE CONSTRAINT IF NOT EXISTS' attempt: {e_if_not_exists}")
        except Exception as e_fallback_check:
            # Если даже фолбэк-проверка не удалась
            print(f"Fallback check/create for constraint failed: {e_fallback_check}")
            print(f"Original error from 'CREATE CONSTRAINT IF NOT EXISTS' attempt: {e_if_not_exists}")
            print("Continuing, assuming AuraDB might handle this or it's a non-critical warning for tests.")


def get_word_node(tx: ManagedTransaction, word_text: str):
    query = "MATCH (w:Word {text: $text}) RETURN w.text AS text, w.vector AS vector, w.priority AS priority, w.usage_count AS usage_count"
    result = tx.run(query, text=word_text)
    return result.single()

def create_word_node(tx: ManagedTransaction, word_text: str, vector: list[float], paragraph_priority: float):
    query = (
        "CREATE (w:Word {text: $text, vector: $vector, priority: $priority, usage_count: 1})"
        "RETURN w"
    )
    tx.run(query, text=word_text, vector=vector, priority=paragraph_priority)

def update_word_node(tx: ManagedTransaction, word_text: str,
                     paragraph_vector: list[float], paragraph_priority: float,
                     existing_vector: list[float], existing_priority: float,
                     usage_count: int, text_priority_setting: float):
    np_existing_vector = np.array(existing_vector)
    np_paragraph_vector = np.array(paragraph_vector)

    new_vector = np_existing_vector + (np_paragraph_vector - np_existing_vector) * (1 / (usage_count + 1)) * text_priority_setting
    new_priority = float(existing_priority) + (float(paragraph_priority) - float(existing_priority)) * (1 / (usage_count + 1))
    new_usage_count = usage_count + 1

    query = (
        "MATCH (w:Word {text: $text}) "
        "SET w.vector = $vector, w.priority = $priority, w.usage_count = $usage_count "
        "RETURN w"
    )
    tx.run(query, text=word_text, vector=new_vector.tolist(), priority=new_priority, usage_count=new_usage_count)

def add_relationship(tx: ManagedTransaction, word1_text: str, word2_text: str, sentence_id: str):
    query = (
        "MATCH (w1:Word {text: $text1}), (w2:Word {text: $text2}) "
        "MERGE (w1)-[r:PRECEDES {sentence_id: $sentence_id}]->(w2) "
        "RETURN type(r)"
    )
    tx.run(query, text1=word1_text, text2=word2_text, sentence_id=sentence_id)

def execute_batch(queries_with_params: list[tuple[str, dict]]):
    db_driver = get_driver()
    if not db_driver:
        print("Cannot execute batch, Neo4j driver not available.")
        return

    with db_driver.session(database="neo4j") as session:
        try:
            def batch_work(tx: ManagedTransaction):
                results = []
                for query_text, params in queries_with_params:
                    results.append(tx.run(query_text, **params))
                return results
            session.execute_write(batch_work)
        except Exception as e:
            print(f"Error executing batch on AuraDB: {e}")


if __name__ == '__main__':
    print("Starting graph_db.py self-test for Neo4j AuraDB...")
    try:
        db_driver_instance = get_driver()
        if not db_driver_instance:
            print("Exiting test due to Neo4j AuraDB connection failure.")

        if db_driver_instance:
            with db_driver_instance.session(database="neo4j") as session:
                print("Ensuring constraints (for AuraDB)...")
                session.execute_write(ensure_constraints)

                print("Cleaning up previous test data (if any)...")
                def cleanup_work(tx: ManagedTransaction):
                    # Добавляем _aura к тестовым словам, чтобы не пересекаться с локальными тестами, если они были
                    tx.run("MATCH (w:Word {text: 'test_word1_aura'}) DETACH DELETE w")
                    tx.run("MATCH (w:Word {text: 'test_word2_aura'}) DETACH DELETE w")
                session.execute_write(cleanup_work)
                print("Cleanup complete.")

            test_word1 = "test_word1_aura"
            test_vector1 = [0.1] * 768
            test_priority1 = 0.5

            test_word2 = "test_word2_aura"
            test_vector2 = [0.2] * 768
            test_priority2 = 0.8

            text_file_priority_setting = 0.3
            sentence_id_test = "test_sentence_aura_123"

            batch_ops = []

            print(f"Preparing to create '{test_word1}'...")
            batch_ops.append((
                "CREATE (w:Word {text: $text, vector: $vector, priority: $priority, usage_count: 1})",
                {"text": test_word1, "vector": test_vector1, "priority": test_priority1}
            ))

            print(f"Preparing to create '{test_word2}'...")
            batch_ops.append((
                "CREATE (w:Word {text: $text, vector: $vector, priority: $priority, usage_count: 1})",
                {"text": test_word2, "vector": test_vector2, "priority": test_priority2}
            ))

            print("Executing batch creation on AuraDB...")
            execute_batch(batch_ops)
            batch_ops = []

            with db_driver_instance.session(database="neo4j") as session:
                node1 = session.execute_read(get_word_node, test_word1)
                assert node1 is not None, f"Node '{test_word1}' was not created."
                assert node1["text"] == test_word1
                assert len(node1["vector"]) == 768
                assert abs(node1["priority"] - test_priority1) < 1e-9
                assert node1["usage_count"] == 1
                print(f"Node '{test_word1}' created and verified.")

                node2 = session.execute_read(get_word_node, test_word2)
                assert node2 is not None, f"Node '{test_word2}' was not created."
                print(f"Node '{test_word2}' created and verified.")

            node1_data = {}
            with db_driver_instance.session(database="neo4j") as session:
                 node1_data = session.execute_read(get_word_node, test_word1)

            print(f"Preparing to update '{test_word1}'...")
            paragraph_vec_update = [0.15] * 768
            paragraph_prio_update = 0.6

            existing_vec_for_update = node1_data["vector"]
            existing_prio_for_update = node1_data["priority"]
            existing_usage_for_update = node1_data["usage_count"]

            np_existing_vector = np.array(existing_vec_for_update)
            np_paragraph_vector = np.array(paragraph_vec_update)
            expected_new_usage = existing_usage_for_update + 1

            expected_new_vector = np_existing_vector + \
                                (np_paragraph_vector - np_existing_vector) * \
                                (1 / expected_new_usage) * text_file_priority_setting

            expected_new_priority = float(existing_prio_for_update) + \
                                    (float(paragraph_prio_update) - float(existing_prio_for_update)) * \
                                    (1 / expected_new_usage)

            batch_ops.append((
                "MATCH (w:Word {text: $text}) "
                "SET w.vector = $vector, w.priority = $priority, w.usage_count = $usage_count",
                {"text": test_word1,
                "vector": expected_new_vector.tolist(),
                "priority": expected_new_priority,
                "usage_count": expected_new_usage}
            ))
            print("Executing batch update on AuraDB...")
            execute_batch(batch_ops)
            batch_ops = []

            with db_driver_instance.session(database="neo4j") as session:
                updated_node1 = session.execute_read(get_word_node, test_word1)
                assert updated_node1 is not None, f"Node '{test_word1}' not found after update."
                assert updated_node1["usage_count"] == expected_new_usage
                assert np.allclose(np.array(updated_node1["vector"]), expected_new_vector), "Vector was not updated correctly."
                assert abs(updated_node1["priority"] - expected_new_priority) < 1e-9, "Priority was not updated correctly."
                print(f"Node '{test_word1}' updated and verified.")

            print(f"Preparing to add relationship between '{test_word1}' and '{test_word2}'...")
            batch_ops.append((
                "MATCH (w1:Word {text: $text1}), (w2:Word {text: $text2}) "
                "MERGE (w1)-[r:PRECEDES {sentence_id: $sentence_id}]->(w2)",
                {"text1": test_word1, "text2": test_word2, "sentence_id": sentence_id_test}
            ))
            print("Executing batch relationship creation on AuraDB...")
            execute_batch(batch_ops)
            batch_ops = []

            with db_driver_instance.session(database="neo4j") as session:
                def check_rel_work(tx: ManagedTransaction):
                    res = tx.run("MATCH (w1:Word {text: $t1})-[r:PRECEDES {sentence_id: $sid}]->(w2:Word {text: $t2}) RETURN type(r) AS rel_type",
                                t1=test_word1, t2=test_word2, sid=sentence_id_test)
                    return res.single()
                record = session.execute_read(check_rel_work)
                assert record is not None and record["rel_type"] == "PRECEDES", f"Relationship PRECEDES not created."
                print(f"Relationship PRECEDES verified.")

            print("graph_db.py self-test for AuraDB completed successfully!")

    except Exception as e:
        print(f"An error occurred during graph_db.py self-test for AuraDB: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if driver:
            close_driver()
