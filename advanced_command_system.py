import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple
import json
import re
from vectorizer import get_vector
from enhanced_controller import EnhancedCommandController

class AdvancedCommandVocabulary:
    """
    Расширенный словарь команд для сложных сценариев
    """
    
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_count = 0
        
        # Добавляем базовые токены
        self._add_basic_tokens()
        # Добавляем финансовые токены
        self._add_financial_tokens()
        # Добавляем бизнес токены
        self._add_business_tokens()
        # Добавляем токены для неопределенностей
        self._add_uncertainty_tokens()
        # Добавляем токены для сценариев
        self._add_scenario_tokens()
    
    def _add_basic_tokens(self):
        """Базовые токены"""
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
    
    def _add_financial_tokens(self):
        """Финансовые токены"""
        financial_tokens = [
            'salary', 'зарплата', 'expense', 'расход', 'income', 'доход',
            'budget', 'бюджет', 'balance', 'остаток', 'savings', 'сбережения',
            'debt', 'долг', 'credit', 'кредит', 'loan', 'займ',
            'investment', 'инвестиция', 'profit', 'прибыль', 'loss', 'убыток',
            'tax', 'налог', 'rent', 'аренда', 'mortgage', 'ипотека',
            'food', 'еда', 'transport', 'транспорт', 'entertainment', 'развлечения',
            'utilities', 'коммунальные', 'insurance', 'страховка', 'medical', 'медицина'
        ]
        
        for token in financial_tokens:
            self.add_token(token)
    
    def _add_business_tokens(self):
        """Бизнес токены"""
        business_tokens = [
            'visitors', 'посетители', 'conversion', 'конверсия', 'revenue', 'выручка',
            'customers', 'клиенты', 'orders', 'заказы', 'sales', 'продажи',
            'marketing', 'маркетинг', 'advertising', 'реклама', 'campaign', 'кампания',
            'roi', 'окупаемость', 'margin', 'маржа', 'cost', 'стоимость',
            'price', 'цена', 'discount', 'скидка', 'commission', 'комиссия',
            'subscription', 'подписка', 'recurring', 'повторяющийся', 'lifetime', 'пожизненный'
        ]
        
        for token in business_tokens:
            self.add_token(token)
    
    def _add_uncertainty_tokens(self):
        """Токены для неопределенностей"""
        uncertainty_tokens = [
            'question', 'вопрос', 'assumption', 'предположение', 'confidence', 'уверенность',
            'unknown', 'неизвестно', 'maybe', 'возможно', 'probably', 'вероятно',
            'uncertain', 'неопределенный', 'clarify', 'уточнить', 'confirm', 'подтвердить',
            'who', 'кто', 'what', 'что', 'where', 'где', 'when', 'когда',
            'how', 'как', 'why', 'почему', 'which', 'какой', 'whose', 'чей'
        ]
        
        for token in uncertainty_tokens:
            self.add_token(token)
    
    def _add_scenario_tokens(self):
        """Токены для сценариев"""
        scenario_tokens = [
            'scenario', 'сценарий', 'option', 'вариант', 'pros', 'плюсы', 'cons', 'минусы',
            'compare', 'сравнить', 'choose', 'выбрать', 'recommend', 'рекомендовать',
            'timeline', 'временная_линия', 'milestone', 'этап', 'goal', 'цель',
            'progress', 'прогресс', 'deadline', 'срок', 'priority', 'приоритет',
            'risk', 'риск', 'mitigation', 'снижение_риска', 'probability', 'вероятность',
            'impact', 'влияние', 'strategy', 'стратегия', 'plan', 'план'
        ]
        
        for token in scenario_tokens:
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
    
    def get_pad_id(self) -> int:
        return self.token_to_id['<PAD>']
    
    def encode_commands(self, commands: str) -> List[int]:
        """Кодирует команды в последовательность токенов"""
        tokens = re.findall(r'\[|\]|,|\(|\)|"[^"]*"|\'[^\']*\'|\w+', commands)
        token_ids = []
        
        for token in tokens:
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

class AdvancedCommandGenerator(nn.Module):
    """
    Расширенная нейросеть для генерации команд
    """
    
    def __init__(self, input_dim=768, hidden_dim=512, max_commands=30):
        super(AdvancedCommandGenerator, self).__init__()
        
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
        self.output_layer = nn.Linear(hidden_dim, 1500)  # Увеличенный размер словаря
        
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
            
            # Следующий вход: всегда подаем encoded
            current_input = encoded  # [batch_size, 1, hidden_dim]
        
        # После генерации outputs
        outputs = torch.stack(outputs, dim=1)  # [batch, max_len, vocab_size]
        return outputs

class AdvancedCommandTrainingSystem:
    """
    Расширенная система обучения для генерации команд
    """
    
    def __init__(self):
        self.vocabulary = AdvancedCommandVocabulary()
        self.generator = AdvancedCommandGenerator()
        self.controller = EnhancedCommandController()
        self.pad_id = self.vocabulary.get_pad_id()
        self.optimizer = optim.Adam(self.generator.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        
    def create_advanced_training_data(self) -> List[Dict[str, Any]]:
        """
        Создает расширенные данные для обучения
        """
        training_examples = [
            # Финансовые задачи
            {
                "text": "Зарплата 50000, трачу 35000, сколько остается?",
                "commands": "[create person name \"я\" salary 50000, create expense category \"общие\" amount 35000 owner \"я\", query person name \"я\" balance]"
            },
            {
                "text": "У меня зарплата 80000, трачу на еду 20000, аренду 30000, сколько остается?",
                "commands": "[create person name \"я\" salary 80000, create expense category \"еда\" amount 20000 owner \"я\", create expense category \"аренда\" amount 30000 owner \"я\", query person name \"я\" balance]"
            },
            {
                "text": "Бюджет 100000 на ремонт квартиры",
                "commands": "[create budget total 100000 purpose \"ремонт\", create apartment owner \"я\"]"
            },
            
            # Бизнес-расчеты
            {
                "text": "1000 посетителей, конверсия 2%, чек 3000",
                "commands": "[create visitors quantity 1000, create conversion rate 0.02, create order_value average 3000, count revenue formula \"visitors * conversion * order_value\"]"
            },
            {
                "text": "500 клиентов, средний чек 5000, прибыль 20%",
                "commands": "[create customers quantity 500, create order_value average 5000, create margin rate 0.20, count revenue formula \"customers * order_value\", count profit formula \"revenue * margin\"]"
            },
            
            # Ресурсы и планирование
            {
                "text": "Ремонт 60 кв.м, бюджет 800000",
                "commands": "[create apartment area 60 owner \"я\", create budget total 800000 purpose \"ремонт\", create plan item \"материалы\" percent 40, create plan item \"работа\" percent 60]"
            },
            {
                "text": "Проект на 3 месяца, бюджет 500000",
                "commands": "[create project duration \"3_months\" budget 500000, create timeline goal \"завершить_проект\", create milestone month 1 target \"планирование\", create milestone month 2 target \"разработка\", create milestone month 3 target \"тестирование\"]"
            },
            
            # Неопределенности
            {
                "text": "мы будем это?",
                "commands": "[create question name \"кто_мы\" description \"Кто такие 'мы'?\", create question name \"что_это\" description \"Что означает 'это'?\", create question name \"что_будем\" description \"Что значит 'будем' в данном контексте?\"]"
            },
            {
                "text": "сколько это стоит?",
                "commands": "[create question name \"что_это\" description \"Что именно вы имеете в виду под 'это'?\", create question name \"какая_стоимость\" description \"Какую стоимость вы хотите узнать?\"]"
            },
            
            # Сценарии и варианты
            {
                "text": "купить машину за 500000 или взять кредит?",
                "commands": "[create scenario name \"покупка_авто\", create option name \"наличные\" cost 500000 pros [\"без_долгов\"] cons [\"большие_расходы\"], create option name \"кредит\" cost 600000 pros [\"быстро\"] cons [\"переплата\"], compare options by \"общая_стоимость\"]"
            },
            {
                "text": "выбрать работу с зарплатой 100000 или 80000 с перспективой роста?",
                "commands": "[create scenario name \"выбор_работы\", create option name \"высокая_зарплата\" salary 100000 pros [\"деньги\"] cons [\"нет_роста\"], create option name \"перспективная\" salary 80000 pros [\"рост\"] cons [\"меньше_денег\"], compare options by \"долгосрочная_выгода\"]"
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
        print("Создание расширенных данных для обучения...")
        training_data = self.create_advanced_training_data()
        
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
    
    def generate_commands(self, text: str, max_commands: int = 30) -> str:
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
        print("\nТестирование расширенной генерации команд:")
        print("=" * 60)
        
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
            
            print("-" * 40)

def main():
    """
    Основная функция для обучения и тестирования
    """
    print("РАСШИРЕННАЯ СИСТЕМА ОБУЧЕНИЯ ГЕНЕРАЦИИ КОМАНД")
    print("=" * 60)
    
    # Создаем систему обучения
    training_system = AdvancedCommandTrainingSystem()
    
    # Обучаем модель
    print("\nНачинаем обучение...")
    training_system.train(epochs=100)
    
    # Тестируем генерацию
    test_texts = [
        "Зарплата 60000, трачу 40000, сколько остается?",
        "1000 посетителей сайта, конверсия 3%, средний чек 2000",
        "мы будем это?",
        "купить квартиру за 5000000 или арендовать за 50000 в месяц?",
        "проект на 6 месяцев, бюджет 1000000"
    ]
    
    training_system.test_generation(test_texts)

if __name__ == "__main__":
    main() 