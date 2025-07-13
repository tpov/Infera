import nltk
import re
import numpy as np
from vectorizer import get_vector, cosine_similarity
import graph_db
from graph_db import execute_batch # Для пакетных операций

# Убедимся, что необходимые ресурсы nltk загружены (хотя это уже делалось при установке)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' not found. Downloading...")
    nltk.download('punkt', quiet=True)
except Exception: # Обработка других возможных ошибок при поиске nltk.data, если nltk не установлен.
    print("NLTK 'punkt' resource check failed. Ensure NLTK is installed and 'punkt' is available.")
    # В случае если nltk не установлен, последующие вызовы nltk.sent_tokenize упадут,
    # но это будет уже ошибкой времени выполнения, а не импорта.

# Константы
SIMILARITY_THRESHOLD = 0.95

def read_input_file(filepath: str) -> tuple[float, str]:
    """
    Читает входной текстовый файл.
    Первая строка должна быть 'priority: <число>'.
    Остальное - текст для обработки.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            priority_line = f.readline().strip()
            if not priority_line.startswith("priority:"):
                raise ValueError("Input file must start with 'priority: <number>'")

            try:
                priority_value = float(priority_line.split(":")[1].strip())
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid priority format. Expected 'priority: <number>'. Got: '{priority_line}'. Error: {e}")

            text_content = f.read()
            return priority_value, text_content
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error reading input file {filepath}: {e}")
        raise

def text_to_paragraphs_and_sentences(text: str, sentences_per_paragraph_threshold=10) -> list[list[list[str]]]:
    """
    Разделяет текст на абзацы, затем на предложения, затем на слова.
    Абзацы разделяются одной или несколькими пустыми строками.
    Если нет явных абзацев, то каждые `sentences_per_paragraph_threshold` предложений формируют абзац.
    """
    paragraphs_text = []
    # Разделение на абзацы по пустым строкам (одной или более)
    raw_paragraphs = re.split(r'\n\s*\n', text.strip())

    if len(raw_paragraphs) == 1 and not "\n\n" in text: # Если нет явных абзацев по двойному переносу строки
        # Делим по 10 предложений
        sentences = nltk.sent_tokenize(raw_paragraphs[0])
        for i in range(0, len(sentences), sentences_per_paragraph_threshold):
            paragraph_group = sentences[i:i + sentences_per_paragraph_threshold]
            paragraphs_text.append(" ".join(paragraph_group))
    else:
        # Используем абзацы, разделенные пустыми строками
        paragraphs_text = [p.strip() for p in raw_paragraphs if p.strip()]

    processed_paragraphs = []
    for para_text in paragraphs_text:
        if not para_text:
            continue
        sentences = nltk.sent_tokenize(para_text)
        paragraph_structure = []
        for sentence_text in sentences:
            if not sentence_text.strip():
                continue
            # Простое разделение по пробелам и удаление пунктуации для слов.
            # Можно улучшить, используя более сложный токенизатор слов из nltk,
            # но для данной задачи, возможно, этого достаточно.
            # Удаляем знаки препинания и приводим к нижнему регистру.
            words = [re.sub(r'[^\w\s-]', '', word).lower() for word in nltk.word_tokenize(sentence_text)]
            words = [word for word in words if word and word != '-'] # Убираем пустые строки и одиночные дефисы
            if words:
                paragraph_structure.append(words)
        if paragraph_structure:
            processed_paragraphs.append(paragraph_structure)

    return processed_paragraphs


def process_text(filepath: str):
    """
    Основная функция обработки текста.
    """
    print(f"Starting text processing for file: {filepath}")
    file_priority, text_content = read_input_file(filepath)
    print(f"File priority read: {file_priority}")

    paragraphs_data = text_to_paragraphs_and_sentences(text_content)
    if not paragraphs_data:
        print("No paragraphs found to process.")
        return

    # Инициализация Neo4j
    db_driver = graph_db.get_driver()
    if not db_driver:
        print("Could not connect to Neo4j. Aborting processing.")
        return

    # Убедимся, что ограничения существуют
    with db_driver.session(database="neo4j") as session:
        session.execute_write(graph_db.ensure_constraints)

    total_paragraphs = len(paragraphs_data)
    print(f"Total paragraphs to process: {total_paragraphs}")

    # Счетчик для сохранения каждые N абзацев (не используется с пакетной записью, но оставим для информации)
    # paragraph_counter_for_save = 0

    for i, paragraph_sentences in enumerate(paragraphs_data):
        print(f"\nProcessing paragraph {i+1}/{total_paragraphs}...")

        # Собираем текст всего абзаца для получения его вектора
        # Первый абзац (i=0) используется для получения вектора абзаца, как указано в задаче.
        # "программа берет первий абзац и строит из него многомерний вектор"
        # "затем удаляешь єтот абзац если будет мешать, и берешь следубщий, делаешь вектор"
        # Из этого следует, что каждый абзац имеет свой вектор.

        current_paragraph_text_for_vector = " ".join([" ".join(sent) for sent in paragraph_sentences])
        if not current_paragraph_text_for_vector.strip():
            print(f"Paragraph {i+1} is empty after joining sentences, skipping.")
            continue

        paragraph_vector = get_vector(current_paragraph_text_for_vector)
        # Приоритет абзаца для новых слов будет равен file_priority (как обсуждалось)
        paragraph_priority_for_new_words = file_priority

        batch_operations = [] # Собираем операции для этого абзаца

        for sentence_idx, sentence_words in enumerate(paragraph_sentences):
            if not sentence_words:
                continue

            # Генерируем уникальный ID для предложения (например, paragraph_index_sentence_index)
            # Это нужно для свойства ребра в Neo4j, чтобы различать связи из разных предложений.
            sentence_id = f"p{i}_s{sentence_idx}"

            print(f"  Processing sentence {sentence_idx+1}/{len(paragraph_sentences)}: \"{' '.join(sentence_words[:5])}...\"")

            previous_word_text = None
            for word_idx, word_text in enumerate(sentence_words):
                if not word_text: # Пропускаем пустые слова, если такие остались
                    continue

                # Получаем или создаем/обновляем узел слова
                # Используем транзакцию для каждой операции со словом, чтобы прочитать и затем записать
                # Это не очень эффективно, лучше собирать в батч, но требует более сложной логики
                # для чтения перед записью в батче. Пока сделаем так для ясности.
                # ----
                # Переделка на сбор операций в batch_operations:

                # 1. Проверить, существует ли слово
                # Это все еще требует чтения из БД до формирования батча записи.
                # Для оптимизации этого момента, можно было бы загрузить часть графа в память
                # или иметь более сложную логику отложенных операций.
                # Пока оставим чтение перед формированием операции для батча.

                node_data = None
                with db_driver.session(database="neo4j") as session:
                    node_data = session.execute_read(graph_db.get_word_node, word_text)

                if node_data:
                    # Слово существует, проверяем сходство векторов
                    existing_vector = node_data["vector"]
                    similarity = cosine_similarity(paragraph_vector, existing_vector)

                    if similarity >= SIMILARITY_THRESHOLD:
                        # Сходство высокое, обновляем вектор и приоритет слова
                        # print(f"    Word '{word_text}' exists. High similarity ({similarity:.2f}). Updating.")
                        # Формируем операцию обновления для батча
                        batch_operations.append((
                            "MATCH (w:Word {text: $text}) "
                            "SET w.vector = $new_vector, w.priority = $new_priority, w.usage_count = w.usage_count + 1",
                            {
                                "text": word_text,
                                "new_vector": (np.array(existing_vector) + (np.array(paragraph_vector) - np.array(existing_vector)) * (1 / (node_data["usage_count"] + 1)) * file_priority).tolist(),
                                "new_priority": float(node_data["priority"]) + (paragraph_priority_for_new_words - float(node_data["priority"])) * (1 / (node_data["usage_count"] + 1))
                            }
                        ))
                    else:
                        # Сходство низкое, не обновляем вектор, но увеличиваем счетчик использования
                        # И связь все равно создается.
                        # print(f"    Word '{word_text}' exists. Low similarity ({similarity:.2f}). Incrementing usage count only.")
                        batch_operations.append((
                            "MATCH (w:Word {text: $text}) SET w.usage_count = w.usage_count + 1",
                            {"text": word_text}
                        ))
                else:
                    # Слова нет, создаем новый узел
                    # print(f"    Word '{word_text}' is new. Creating node.")
                    batch_operations.append((
                        "CREATE (w:Word {text: $text, vector: $vector, priority: $priority, usage_count: 1})",
                        {"text": word_text, "vector": paragraph_vector, "priority": paragraph_priority_for_new_words}
                    ))

                # Добавляем связь с предыдущим словом, если оно есть
                if previous_word_text:
                    # print(f"      Adding relationship: '{previous_word_text}' -> '{word_text}'")
                    batch_operations.append((
                        "MATCH (w1:Word {text: $text1}), (w2:Word {text: $text2}) "
                        "MERGE (w1)-[r:PRECEDES {sentence_id: $sentence_id}]->(w2)",
                        {"text1": previous_word_text, "text2": word_text, "sentence_id": sentence_id}
                    ))

                previous_word_text = word_text

        # После обработки всех слов в абзаце, выполняем собранные операции батчем
        if batch_operations:
            print(f"  Executing batch of {len(batch_operations)} operations for paragraph {i+1}...")
            execute_batch(batch_operations) # execute_batch использует get_driver() и сессию внутри себя
            print(f"  Batch execution for paragraph {i+1} complete.")
        else:
            print(f"  No operations to execute for paragraph {i+1}.")

        # Логика сохранения каждые N абзацев (если бы не было батчей на каждый абзац)
        # paragraph_counter_for_save += 1
        # if paragraph_counter_for_save % 10 == 0:
        #     print(f"Processed 10 paragraphs (total {i+1}). 'Saving' graph (already saved by batches).")
        #     # Здесь могла бы быть логика сохранения состояния, если бы не постоянная запись в БД

    # Сбор метаданных после обработки
    print("\nText processing finished.")
    print("Collecting metadata...")
    metadata = {}
    with db_driver.session(database="neo4j") as session:
        num_nodes_result = session.run("MATCH (n:Word) RETURN count(n) AS count").single()
        metadata["node_count"] = num_nodes_result["count"] if num_nodes_result else 0

        num_rels_result = session.run("MATCH ()-[r:PRECEDES]->() RETURN count(r) AS count").single()
        metadata["relationship_count"] = num_rels_result["count"] if num_rels_result else 0

    print("Metadata:")
    print(f"  Total word nodes: {metadata['node_count']}")
    print(f"  Total PRECEDES relationships: {metadata['relationship_count']}")
    print(f"  Input file priority setting: {file_priority}")

    graph_db.close_driver()
    print("Processing complete. Neo4j connection closed.")


if __name__ == '__main__':
    # Создадим простой тестовый файл для проверки
    test_input_filename = "sample_input_tp.txt"
    with open(test_input_filename, "w", encoding="utf-8") as f:
        f.write("priority: 0.25\n")
        f.write("This is the first paragraph. It has two sentences.\n\n")
        f.write("This is the second paragraph. It is also short. A third sentence here.\n\n")
        f.write("A final paragraph with a repeated word. This paragraph is the end.")

    # Перед запуском этого теста убедитесь, что Neo4j AuraDB доступна
    # и vectorizer.py и graph_db.py находятся в той же директории.
    # Также, что vectorizer загрузит модель (может занять время при первом запуске).

    print(f"Running text_processor self-test with {test_input_filename}...")
    try:
        process_text(test_input_filename)
        print(f"\nSelf-test finished. Check Neo4j AuraDB for nodes 'this', 'is', 'first', 'paragraph', etc., and relationships.")
        print(f"Words like 'paragraph' should have usage_count > 1 if they appeared in multiple contexts suitable for merging/updating.")
    except Exception as e:
        print(f"Self-test failed: {e}")
        import traceback
        traceback.print_exc()

    # Пример для случая без явных абзацев (10 предложений)
    test_no_explicit_para_filename = "sample_no_para_tp.txt"
    with open(test_no_explicit_para_filename, "w", encoding="utf-8") as f:
        f.write("priority: 0.1\n")
        f.write("Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. ")
        f.write("Sentence 6. Sentence 7. Sentence 8. Sentence 9. Sentence 10. ")
        f.write("Sentence 11. Sentence 12. This should be a new paragraph group.")

    print(f"\nRunning text_processor self-test with {test_no_explicit_para_filename} (implicit paragraphs)...")
    try:
        process_text(test_no_explicit_para_filename)
        print(f"\nSelf-test for implicit paragraphs finished. Check Neo4j AuraDB.")
    except Exception as e:
        print(f"Self-test for implicit paragraphs failed: {e}")
        import traceback
        traceback.print_exc()
