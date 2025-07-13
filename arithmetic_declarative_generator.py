#!/usr/bin/env python
"""
Генератор арифметических данных в ДЕКЛАРАТИВНОМ формате
"""
import json
import random
import re

class ArithmeticDeclarativeGenerator:
    def __init__(self):
        self.operators = ['+', '-', '*']
        self.operator_words = {'+': 'плюс', '-': 'минус', '*': 'умножить на'}
        print("Initialized ArithmeticDeclarativeGenerator.")

    def generate_arithmetic_expression(self, max_numbers=3):
        """Генерирует арифметическое выражение"""
        num_count = random.randint(2, max_numbers)
        numbers = [random.randint(1, 20) for _ in range(num_count)]
        operators = [random.choice(self.operators) for _ in range(num_count - 1)]
        
        # Создаём выражение
        expression_parts = []
        for i in range(num_count):
            expression_parts.append(str(numbers[i]))
            if i < len(operators):
                expression_parts.append(self.operator_words[operators[i]])
        
        expression_text = " ".join(expression_parts)
        natural_question = f"сколько будет {expression_text}"
        
        return {
            'numbers': numbers,
            'operators': operators,
            'expression_text': expression_text,
            'question': natural_question
        }

    def generate_declarative_output(self, arithmetic_data):
        """Генерирует декларативный вывод в формате OBJECT/INTENT"""
        declarations = []
        numbers = arithmetic_data['numbers']
        operators = arithmetic_data['operators']
        
        object_counter = 1
        
        # Создаём объекты для чисел и операции
        for i in range(len(numbers)):
            # Объект для числа
            declarations.append(f'OBJECT(name:"it{object_counter}",type:"virtual", count: "{numbers[i]}")')
            object_counter += 1
            
            # Если есть операция после этого числа
            if i < len(operators):
                operator_action = f"it{object_counter-1} {operators[i]}"
                declarations.append(f'INTENT(name:"it{object_counter}",type:"virtual", action: "{operator_action}")')
                object_counter += 1
        
        # Объект-вопрос
        declarations.append(f'OBJECT(name:"it{object_counter}",type:"virtual question", count: "?")')
        
        return declarations

    def generate_training_sample(self):
        """Генерирует один образец для обучения"""
        arithmetic_data = self.generate_arithmetic_expression()
        declarations = self.generate_declarative_output(arithmetic_data)
        
        return {
            'text': arithmetic_data['question'],
            'declarations': declarations,
            'raw_expression': arithmetic_data['expression_text']
        }

    def generate_dataset(self, num_samples=500):
        """Генерирует датасет"""
        dataset = []
        
        print(f"🔢 Генерируем {num_samples} арифметических примеров...")
        
        for i in range(num_samples):
            sample = self.generate_training_sample()
            if sample:
                dataset.append(sample)
                
                # Показываем первые 5 примеров
                if i < 5:
                    print(f"\n--- Пример {i+1} ---")
                    print(f"Вопрос: {sample['text']}")
                    print("Декларации:")
                    for decl in sample['declarations']:
                        print(f"  {decl}")
        
        return dataset

def main():
    generator = ArithmeticDeclarativeGenerator()
    
    # Генерируем данные
    dataset = generator.generate_dataset(500)
    
    # Конвертируем в формат для T5 (text-to-text)
    training_data = []
    for sample in dataset:
        # Входной текст
        input_text = f"convert to declarative: {sample['text']}"
        
        # Выходной текст (декларации через новую строку)
        output_text = "\n".join(sample['declarations'])
        
        training_data.append({
            'text': input_text,
            'declarations_text': output_text,
            'raw_question': sample['text']
        })
    
    # Сохраняем
    output_file = "arithmetic_declarative_training.jsonl"
    print(f"\n💾 Сохраняем {len(training_data)} примеров в {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ ГОТОВО! Сгенерировано {len(training_data)} арифметических примеров!")
    print(f"📊 Формат данных:")
    print(f"   Вход: 'convert to declarative: сколько будет 5 плюс 3'")
    print(f"   Выход: 'OBJECT(name:\"it1\",type:\"virtual\", count: \"5\")\\nINTENT(...)'")
    print(f"🧠 Нейросеть научится превращать арифметику в декларации!")

if __name__ == "__main__":
    main() 