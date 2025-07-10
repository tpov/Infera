import argparse
import sys
import os
from text_processor import process_text

def main():
    parser = argparse.ArgumentParser(description="Process a text file to build a graph in Neo4j.")
    parser.add_argument("input_file", help="Path to the input text file.")

    # Дополнительные аргументы можно добавить здесь, если потребуется
    # Например, URI Neo4j, пользователя, пароль, если не хотим их жестко кодировать
    # parser.add_argument("--neo4j_uri", default="neo4j+s://your_aura_uri.databases.neo4j.io", help="Neo4j URI")
    # parser.add_argument("--neo4j_user", default="neo4j", help="Neo4j Username")
    # parser.add_argument("--neo4j_password", help="Neo4j Password (will prompt if not provided and needed)")

    if len(sys.argv) == 1:
        # Если аргументы не переданы, печатаем справку и выходим
        # Это полезно, если пользователь просто запускает main.py без параметров
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    input_filepath = args.input_file

    if not os.path.exists(input_filepath):
        print(f"Error: Input file '{input_filepath}' not found.")
        sys.exit(1)

    if not os.path.isfile(input_filepath):
        print(f"Error: '{input_filepath}' is not a file.")
        sys.exit(1)

    try:
        print(f"Starting processing for {input_filepath}...")
        process_text(input_filepath)
        print(f"Successfully processed {input_filepath}.")
    except ValueError as ve: # Ошибки, которые мы сами генерируем (например, формат priority)
        print(f"ValueError: {ve}")
        sys.exit(1)
    except ConnectionError as ce: # Проблемы с подключением к Neo4j
        print(f"ConnectionError: Could not connect to the database. {ce}")
        print("Please ensure Neo4j is running and connection details are correct.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
