import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple
import json
import re
from vectorizer import get_vector
from enhanced_controller import EnhancedCommandController

class CommandGenerator(nn.Module):
    """
    Нейросеть для генерации команд на основе входного текста
    """
    
    def __init__(self, input_dim=768, hidden_dim=512, max_commands=20):
        super(CommandGenerator, self).__init__()
        
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
        self.output_layer = nn.Linear(hidden_dim, 1000)  # Размер словаря команд
        
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
            
            # Следующий вход: всегда подаем encoded (фиксируем вход)
            current_input = encoded  # [batch_size, 1, hidden_dim]
        
        # После генерации outputs
        outputs = torch.stack(outputs, dim=1)  # [batch, max_len, vocab_size]
        # print shapes для отладки
        # print(f"outputs shape: {outputs.shape}, targets shape: {targets.shape}")
        return outputs

class CommandVocabulary:
    """
    Словарь для команд
    """
    
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_count = 0
        
        # Добавляем базовые токены
        self._add_basic_tokens()
    
    def _add_basic_tokens(self):
        """Добавляет базовые токены"""
        basic_tokens = [
            '<PAD>', '<START>', '<END>', '<UNK>',
            '[', ']', ',', '(', ')', '"', "'",
            'create', 'set', 'query', 'count', 'resolve',
            'if', 'then', 'else', 'when', 'do', 'define', 'as',
            'name', 'quantity', 'owner', 'place', 'time', 'value',
            'person', 'apple', 'transfer', 'question', 'it',
            'я', 'алина', 'поляна', 'яблоко', 'яблоки',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '+', '-', '*', '/', '=', '>', '<', '>=', '<=', '!=',
            'and', 'or', 'not', 'true', 'false'
        ]
        
        for token in basic_tokens:
            self.add_token(token)
    
    def add_token(self, token: str) -> int:
        """Добавляет токен в словарь"""
        if token not in self.token_to_id:
            self.token_to_id[token] = self.token_count
            self.id_to_token[self.token_count] = token
            self.token_count += 1
        return self.token_to_id[token]
    
    def get_token_id(self, token: str) -> int:
        """Преобразует токен в ID"""
        return self.token_to_id.get(token, self.token_to_id['<UNK>'])
    
    def get_token_by_id(self, token_id: int) -> str:
        """Преобразует ID в токен"""
        return self.id_to_token.get(token_id, '<UNK>')
    
    def encode_commands(self, commands: str) -> List[int]:
        """Кодирует команды в последовательность токенов"""
        # Простое токенизирование по пробелам и специальным символам
        tokens = re.findall(r'\[|\]|,|\(|\)|"[^"]*"|\'[^\']*\'|\w+', commands)
        token_ids = []
        
        for token in tokens:
            # Убираем кавычки из строк
            if (token.startswith('"') and token.endswith('"')) or \
               (token.startswith("'") and token.endswith("'")):
                token = token[1:-1]
            
            token_ids.append(self.get_token_id(token))
        
        return token_ids
    
    def decode_commands(self, token_ids: List[int]) -> str:
        """Декодирует последовательность токенов в команды"""
        tokens = []
        for token_id in token_ids:
            token = self.get_token_by_id(token_id)
            if token in ['<PAD>', '<START>', '<END>']:
                continue
            tokens.append(token)
        
        return ' '.join(tokens)
    
    def get_vocab_size(self) -> int:
        """Возвращает размер словаря"""
        return self.token_count

    def get_pad_id(self) -> int:
        return self.token_to_id['<PAD>']

class CommandTrainingSystem:
    """
    Система обучения для генерации команд
    """
    
    def __init__(self):
        self.vocabulary = CommandVocabulary()
        self.generator = CommandGenerator()
        self.controller = EnhancedCommandController()
        self.pad_id = self.vocabulary.get_pad_id()
        self.optimizer = optim.Adam(self.generator.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        
    def create_training_data(self) -> List[Dict[str, Any]]:
        """
        Создает данные для обучения на основе примеров
        """
        training_examples = [
            {
                "text": "У меня есть 2 яблока, у алины 3 яблока, мы пошли на поляну и съели по яблоку",
                "commands": "[create person name \"я\" quantity 1, create apple quantity 2 owner \"я\", create person name \"алина\" quantity 1, create apple quantity 3 owner \"алина\", create location name \"поляна\" quantity 1, set apple quantity value apple.quantity-1 where owner=\"я\", set apple quantity value apple.quantity-1 where owner=\"алина\"]"
            },
            {
                "text": "У меня есть 5 яблок, дал Алине 2, сколько осталось?",
                "commands": "[create person name \"я\" quantity 1, create apple quantity 5 owner \"я\", create person name \"алина\" quantity 1, create transfer from \"я\" to \"алина\" item \"apple\" quantity 2, set apple owner \"я\" quantity apple.quantity-2, query apple owner \"я\" quantity]"
            },
            {
                "text": "Меня зовут Петр, у меня 10 яблок",
                "commands": "[create person name \"Петр\" quantity 1, create apple quantity 10 owner \"Петр\"]"
            },
            {
                "text": "Алина дала мне 3 яблока, у меня было 2",
                "commands": "[create person name \"алина\" quantity 1, create person name \"я\" quantity 1, create apple quantity 2 owner \"я\", create transfer from \"алина\" to \"я\" item \"apple\" quantity 3, set apple owner \"я\" quantity apple.quantity+3]"
            },
            {
                "text": "Сколько у меня яблок?",
                "commands": "[query apple owner \"я\" quantity]"
            },
            {
                "text": "У нас с Алиной вместе 8 яблок",
                "commands": "[create person name \"я\" quantity 1, create person name \"алина\" quantity 1, create apple quantity 8 owner \"мы\"]"
            },
            {
                "text": "Я съел половину своих яблок",
                "commands": "[set apple owner \"я\" quantity apple.quantity/2]"
            },
            {
                "text": "Алина съела 2 яблока из своих 5",
                "commands": "[create person name \"алина\" quantity 1, create apple quantity 5 owner \"алина\", set apple owner \"алина\" quantity apple.quantity-2]"
            }
        ]
        
        # Векторизуем тексты и кодируем команды
        training_data = []
        for example in training_examples:
            vector = get_vector(example["text"])
            command_tokens = self.vocabulary.encode_commands(example["commands"])
            
            training_data.append({
                "vector": vector,
                "commands": command_tokens,
                "original_text": example["text"],
                "original_commands": example["commands"]
            })
        
        return training_data
    
    def train(self, epochs: int = 100, batch_size: int = 4):
        """
        Обучает модель генерации команд
        """
        print("Создание данных для обучения...")
        training_data = self.create_training_data()
        
        print(f"Создано {len(training_data)} примеров для обучения")
        print(f"Размер словаря: {self.vocabulary.get_vocab_size()}")
        
        # Переводим модель в режим обучения
        self.generator.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Обрабатываем данные батчами
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                
                # Подготавливаем данные
                vectors = torch.tensor([item["vector"] for item in batch], dtype=torch.float32)
                target_commands = [item["commands"] for item in batch]
                
                # Находим максимальную длину в батче
                max_len = max(len(commands) for commands in target_commands)
                print(f"target_commands lens: {[len(c) for c in target_commands]}, max_len: {max_len}")
                # Создаем тензор целей с паддингом
                targets = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
                for j, commands in enumerate(target_commands):
                    for k, token_id in enumerate(commands):
                        if k < max_len:
                            targets[j, k] = token_id
                
                # Обнуляем градиенты
                self.optimizer.zero_grad()
                
                # Прямой проход
                outputs = self.generator(vectors, max_length=max_len)

                # Отладка shapes
                print(f"outputs shape: {outputs.shape}, targets shape: {targets.shape}")
                print(f"outputs numel: {outputs.shape[0]*outputs.shape[1]}, targets numel: {targets.numel()}")
                assert outputs.shape[:2] == targets.shape, f"Mismatch: outputs {outputs.shape}, targets {targets.shape}"

                # Вычисляем потери
                loss = self.criterion(outputs.permute(0, 2, 1), targets)
                
                # Обратное распространение
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Выводим прогресс
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Эпоха {epoch + 1}/{epochs}, Средняя потеря: {avg_loss:.4f}")
        
        print("Обучение завершено!")
    
    def generate_commands(self, text: str, max_commands: int = 20) -> str:
        """
        Генерирует команды для заданного текста
        """
        # Векторизуем текст
        vector = get_vector(text)
        vector_tensor = torch.tensor([vector], dtype=torch.float32)
        
        # Переводим модель в режим оценки
        self.generator.eval()
        
        with torch.no_grad():
            # Генерируем команды
            outputs = self.generator(vector_tensor, max_length=max_commands)
            
            # Преобразуем в токены
            token_ids = torch.argmax(outputs, dim=-1)[0].tolist()
            
            # Декодируем команды
            commands = self.vocabulary.decode_commands(token_ids)
            
            return commands
    
    def test_generation(self, test_texts: List[str]):
        """
        Тестирует генерацию команд
        """
        print("\nТестирование генерации команд:")
        print("=" * 50)
        
        for text in test_texts:
            print(f"\nВходной текст: {text}")
            commands = self.generate_commands(text)
            print(f"Сгенерированные команды: {commands}")
            
            # Пытаемся выполнить команды
            try:
                result = self.controller.execute_commands(commands)
                print(f"Результат выполнения: {result.success}")
                if result.message:
                    print(f"Сообщение: {result.message}")
            except Exception as e:
                print(f"Ошибка выполнения: {str(e)}")
            
            print("-" * 30)

def main():
    """
    Основная функция для обучения и тестирования
    """
    print("СИСТЕМА ОБУЧЕНИЯ ГЕНЕРАЦИИ КОМАНД")
    print("=" * 50)
    
    # Создаем систему обучения
    training_system = CommandTrainingSystem()
    
    # Обучаем модель
    print("\nНачинаем обучение...")
    training_system.train(epochs=200)
    
    # Тестируем генерацию
    test_texts = [
        "У меня есть 3 яблока",
        "Алина дала мне 2 яблока",
        "Сколько у меня яблок?",
        "Мы съели по яблоку"
    ]
    
    training_system.test_generation(test_texts)

if __name__ == "__main__":
    main() 