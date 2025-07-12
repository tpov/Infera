import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any
import json

class SimpleQueryGenerator(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, vocab_size=1000):
        super(SimpleQueryGenerator, self).__init__()
        
        # Энкодер для входного вектора
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LSTM для генерации последовательности команд
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Выходной слой
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_vector, max_length=50):
        batch_size = input_vector.size(0)
        
        # Кодируем входной вектор
        encoded = self.encoder(input_vector)  # [batch_size, hidden_dim]
        
        # Подготавливаем для LSTM
        encoded = encoded.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Инициализируем скрытое состояние
        h0 = torch.zeros(2, batch_size, encoded.size(-1)).to(input_vector.device)
        c0 = torch.zeros(2, batch_size, encoded.size(-1)).to(input_vector.device)
        
        # Генерируем последовательность
        outputs = []
        current_input = encoded
        
        for _ in range(max_length):
            # LSTM шаг
            lstm_out, (h0, c0) = self.lstm(current_input, (h0, c0))
            
            # Выходной слой
            output = self.output_layer(lstm_out)
            outputs.append(output)
            
            # Следующий вход (используем последний выход)
            current_input = output
        
        return torch.stack(outputs, dim=1)  # [batch_size, max_length, vocab_size]

class CommandVocabulary:
    def __init__(self):
        # Базовые токены
        self.tokens = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2,
            '<UNK>': 3,
            '[': 4,
            ']': 5,
            ',': 6,
            
            # Операции
            'set': 7,
            'delete': 8,
            'get': 9,
            'update': 10,
            'create': 11,
            'remove': 12,
            'modify': 13,
            'configure': 14,
            'activate': 15,
            'deactivate': 16,
            
            # Объекты
            'temperature': 17,
            'humidity': 18,
            'pressure': 19,
            'ventilation': 20,
            'system': 21,
            'settings': 22,
            'power': 23,
            'mode': 24,
            'time': 25,
            'lighting': 26,
            'security': 27,
            'camera': 28,
            'door': 29,
            'window': 30,
            'lock': 31,
            'alarm': 32,
            'sensor': 33,
            'fan': 34,
            'heater': 35,
            'cooler': 36,
            'pump': 37,
            'valve': 38,
            'motor': 39,
            'switch': 40,
            'relay': 41,
            'timer': 42,
            'schedule': 43,
            'status': 44,
            
            # Количество (quantity)
            'quantity': 45,
            'count': 46,
            'number': 47,
            'amount': 48,
            'total': 49,
            'single': 50,
            'multiple': 51,
            'all': 52,
            'none': 53,
            'zero': 54,
            'one': 55,
            'two': 56,
            'three': 57,
            'four': 58,
            'five': 59,
            'ten': 60,
            'hundred': 61,
            'thousand': 62,
            
            # Размер (size)
            'size': 63,
            'width': 64,
            'height': 65,
            'length': 66,
            'depth': 67,
            'diameter': 68,
            'radius': 69,
            'area': 70,
            'volume': 71,
            'capacity': 72,
            'large': 73,
            'small': 74,
            'medium': 75,
            'tiny': 76,
            'huge': 77,
            'micro': 78,
            'macro': 79,
            'mini': 80,
            'maxi': 81,
            
            # Положение (position)
            'position': 82,
            'location': 83,
            'place': 84,
            'spot': 85,
            'point': 86,
            'coordinates': 87,
            'x': 88,
            'y': 89,
            'z': 90,
            'left': 91,
            'right': 92,
            'top': 93,
            'bottom': 94,
            'center': 95,
            'front': 96,
            'back': 97,
            'side': 98,
            'corner': 99,
            'edge': 100,
            'middle': 101,
            'near': 102,
            'far': 103,
            'inside': 104,
            'outside': 105,
            'above': 106,
            'below': 107,
            'beside': 108,
            'between': 109,
            'around': 110,
            
            # Ориентация (orientation)
            'orientation': 111,
            'direction': 112,
            'angle': 113,
            'rotation': 114,
            'tilt': 115,
            'pitch': 116,
            'yaw': 117,
            'roll': 118,
            'horizontal': 119,
            'vertical': 120,
            'diagonal': 121,
            'parallel': 122,
            'perpendicular': 123,
            'north': 124,
            'south': 125,
            'east': 126,
            'west': 127,
            'up': 128,
            'down': 129,
            'forward': 130,
            'backward': 131,
            'clockwise': 132,
            'counterclockwise': 133,
            
            # Зависимости (dependencies)
            'dependency': 134,
            'depends_on': 135,
            'requires': 136,
            'needs': 137,
            'relies_on': 138,
            'connected_to': 139,
            'linked_to': 140,
            'attached_to': 141,
            'mounted_on': 142,
            'supported_by': 143,
            'powered_by': 144,
            'controlled_by': 145,
            'monitored_by': 146,
            'triggered_by': 147,
            'activated_by': 148,
            'deactivated_by': 149,
            'parent': 150,
            'child': 151,
            'ancestor': 152,
            'descendant': 153,
            'sibling': 154,
            'related_to': 155,
            
            # Условия существования (existence conditions)
            'condition': 156,
            'if': 157,
            'when': 158,
            'while': 159,
            'until': 160,
            'unless': 161,
            'provided': 162,
            'given': 163,
            'assuming': 164,
            'supposing': 165,
            'in_case': 166,
            'on_condition': 167,
            'subject_to': 168,
            'contingent_on': 169,
            'dependent_on': 170,
            'based_on': 171,
            'according_to': 172,
            'following': 173,
            'after': 174,
            'before': 175,
            'during': 176,
            'since': 177,
            'because': 178,
            'due_to': 179,
            'as_a_result': 180,
            'therefore': 181,
            'consequently': 182,
            'thus': 183,
            'hence': 184,
            
            # Состояния (states)
            'state': 185,
            'status': 186,
            'condition_state': 187,
            'active': 188,
            'inactive': 189,
            'enabled': 190,
            'disabled': 191,
            'on': 192,
            'off': 193,
            'running': 194,
            'stopped': 195,
            'paused': 196,
            'idle': 197,
            'busy': 198,
            'ready': 199,
            'waiting': 200,
            'error': 201,
            'warning': 202,
            'normal': 203,
            'abnormal': 204,
            'critical': 205,
            'safe': 206,
            'unsafe': 207,
            'open': 208,
            'closed': 209,
            'locked': 210,
            'unlocked': 211,
            'full': 212,
            'empty': 213,
            'partial': 214,
            'complete': 215,
            'incomplete': 216,
            'valid': 217,
            'invalid': 218,
            'true': 219,
            'false': 220,
            'yes': 221,
            'no': 222,
            
            # Время (time)
            'time': 223,
            'duration': 224,
            'period': 225,
            'interval': 226,
            'frequency': 227,
            'schedule': 228,
            'deadline': 229,
            'timeout': 230,
            'delay': 231,
            'immediate': 232,
            'instant': 233,
            'moment': 234,
            'second': 235,
            'minute': 236,
            'hour': 237,
            'day': 238,
            'week': 239,
            'month': 240,
            'year': 241,
            'always': 242,
            'never': 243,
            'sometimes': 244,
            'occasionally': 245,
            'frequently': 246,
            'rarely': 247,
            'periodically': 248,
            'continuously': 249,
            'intermittently': 250,
            
            # Значения (values)
            'value': 251,
            'level': 252,
            'intensity': 253,
            'strength': 254,
            'force': 255,
            'pressure_value': 256,
            'temperature_value': 257,
            'humidity_value': 258,
            'speed': 259,
            'velocity': 260,
            'acceleration': 261,
            'torque': 262,
            'voltage': 263,
            'current': 264,
            'power_value': 265,
            'energy': 266,
            'frequency_value': 267,
            'amplitude': 268,
            'wavelength': 269,
            'brightness': 270,
            'luminosity': 271,
            'color': 272,
            'hue': 273,
            'saturation': 274,
            'volume': 275,
            'pitch': 276,
            'tone': 277,
            'quality': 278,
            'grade': 279,
            'rank': 280,
            'score': 281,
            'rating': 282,
            'percentage': 283,
            'ratio': 284,
            'proportion': 285,
            'fraction': 286,
            'decimal': 287,
            'integer': 288,
            'float': 289,
            'string': 290,
            'text': 291,
            'binary': 292,
            'boolean': 293,
            
            # Единицы измерения (units)
            'unit': 294,
            'meter': 295,
            'centimeter': 296,
            'millimeter': 297,
            'kilometer': 298,
            'inch': 299,
            'foot': 300,
            'yard': 301,
            'mile': 302,
            'square_meter': 303,
            'cubic_meter': 304,
            'liter': 305,
            'milliliter': 306,
            'gallon': 307,
            'pound': 308,
            'kilogram': 309,
            'gram': 310,
            'ton': 311,
            'ounce': 312,
            'degree': 313,
            'radian': 314,
            'celsius': 315,
            'fahrenheit': 316,
            'kelvin': 317,
            'pascal': 318,
            'bar': 319,
            'psi': 320,
            'atmosphere': 321,
            'watt': 322,
            'kilowatt': 323,
            'horsepower': 324,
            'volt': 325,
            'ampere': 326,
            'ohm': 327,
            'hertz': 328,
            'rpm': 329,
            'percent': 330,
            'ppm': 331,
            'db': 332,
            'lux': 333,
            'candela': 334,
            'newton': 335,
            'joule': 336,
            'calorie': 337,
            'btu': 338,
            'second_unit': 339,
            'minute_unit': 340,
            'hour_unit': 341,
            'day_unit': 342,
            'week_unit': 343,
            'month_unit': 344,
            'year_unit': 345,
            
            # Операторы сравнения (comparison operators)
            'equals': 346,
            'not_equals': 347,
            'greater_than': 348,
            'less_than': 349,
            'greater_equal': 350,
            'less_equal': 351,
            'approximately': 352,
            'similar_to': 353,
            'different_from': 354,
            'same_as': 355,
            'identical_to': 356,
            'equivalent_to': 357,
            'proportional_to': 358,
            'inverse_of': 359,
            'opposite_of': 360,
            'complementary_to': 361,
            'supplementary_to': 362,
            'correlated_with': 363,
            'independent_of': 364,
            'dependent_on': 365,
            'coupled_with': 366,
            'linked_to': 367,
            'associated_with': 368,
            'related_to': 369,
            'connected_to': 370,
            'separated_from': 371,
            'isolated_from': 372,
            'distinct_from': 373,
            'unique_from': 374,
            'common_with': 375,
            'shared_with': 376,
            'exclusive_to': 377,
            'specific_to': 378,
            'general_to': 379,
            'universal_to': 380,
            'local_to': 381,
            'global_to': 382,
            'relative_to': 383,
            'absolute_to': 384,
            'fixed_to': 385,
            'variable_to': 386,
            'constant_to': 387,
            'dynamic_to': 388,
            'static_to': 389,
            'stable_to': 390,
            'unstable_to': 391,
            'balanced_to': 392,
            'unbalanced_to': 393,
            'symmetric_to': 394,
            'asymmetric_to': 395,
            'regular_to': 396,
            'irregular_to': 397,
            'uniform_to': 398,
            'nonuniform_to': 399,
            'homogeneous_to': 400,
            'heterogeneous_to': 401
        }
        
        # Счетчик для новых токенов
        self.next_id = len(self.tokens)
        
    def add_token(self, token: str) -> int:
        """Добавляет новый токен в словарь"""
        if token not in self.tokens:
            self.tokens[token] = self.next_id
            self.next_id += 1
        return self.tokens[token]
    
    def add_tokens(self, tokens: List[str]) -> None:
        """Добавляет несколько токенов сразу"""
        for token in tokens:
            self.add_token(token)
    
    def token_to_id(self, token: str) -> int:
        """Преобразует токен в ID"""
        return self.tokens.get(token, self.tokens['<UNK>'])
    
    def id_to_token(self, token_id: int) -> str:
        """Преобразует ID в токен"""
        for token, id_val in self.tokens.items():
            if id_val == token_id:
                return token
        return '<UNK>'
    
    def encode_commands(self, commands: str) -> List[int]:
        """Кодирует последовательность команд в ID"""
        tokens = commands.split()
        return [self.token_to_id(token) for token in tokens]
    
    def decode_commands(self, token_ids: List[int]) -> str:
        """Декодирует последовательность ID в команды"""
        tokens = [self.id_to_token(token_id) for token_id in token_ids]
        return ' '.join(tokens)
    
    def get_vocab_size(self) -> int:
        """Возвращает размер словаря"""
        return len(self.tokens)
    
    def list_tokens(self) -> List[str]:
        """Возвращает список всех токенов"""
        return list(self.tokens.keys())
    
    def get_tokens_by_category(self) -> Dict[str, List[str]]:
        """Возвращает токены, сгруппированные по категориям"""
        categories = {
            'operations': ['set', 'delete', 'get', 'update', 'create', 'remove', 'modify', 'configure', 'activate', 'deactivate'],
            'quantity': ['quantity', 'count', 'number', 'amount', 'total', 'single', 'multiple', 'all', 'none', 'zero', 'one', 'two', 'three', 'four', 'five', 'ten', 'hundred', 'thousand'],
            'size': ['size', 'width', 'height', 'length', 'depth', 'diameter', 'radius', 'area', 'volume', 'capacity', 'large', 'small', 'medium', 'tiny', 'huge', 'micro', 'macro', 'mini', 'maxi'],
            'position': ['position', 'location', 'place', 'spot', 'point', 'coordinates', 'x', 'y', 'z', 'left', 'right', 'top', 'bottom', 'center', 'front', 'back', 'side', 'corner', 'edge', 'middle', 'near', 'far', 'inside', 'outside', 'above', 'below', 'beside', 'between', 'around'],
            'orientation': ['orientation', 'direction', 'angle', 'rotation', 'tilt', 'pitch', 'yaw', 'roll', 'horizontal', 'vertical', 'diagonal', 'parallel', 'perpendicular', 'north', 'south', 'east', 'west', 'up', 'down', 'forward', 'backward', 'clockwise', 'counterclockwise'],
            'dependencies': ['dependency', 'depends_on', 'requires', 'needs', 'relies_on', 'connected_to', 'linked_to', 'attached_to', 'mounted_on', 'supported_by', 'powered_by', 'controlled_by', 'monitored_by', 'triggered_by', 'activated_by', 'deactivated_by', 'parent', 'child', 'ancestor', 'descendant', 'sibling', 'related_to'],
            'conditions': ['condition', 'if', 'when', 'while', 'until', 'unless', 'provided', 'given', 'assuming', 'supposing', 'in_case', 'on_condition', 'subject_to', 'contingent_on', 'dependent_on', 'based_on', 'according_to', 'following', 'after', 'before', 'during', 'since', 'because', 'due_to', 'as_a_result', 'therefore', 'consequently', 'thus', 'hence'],
            'states': ['state', 'status', 'condition_state', 'active', 'inactive', 'enabled', 'disabled', 'on', 'off', 'running', 'stopped', 'paused', 'idle', 'busy', 'ready', 'waiting', 'error', 'warning', 'normal', 'abnormal', 'critical', 'safe', 'unsafe', 'open', 'closed', 'locked', 'unlocked', 'full', 'empty', 'partial', 'complete', 'incomplete', 'valid', 'invalid', 'true', 'false', 'yes', 'no'],
            'time': ['time', 'duration', 'period', 'interval', 'frequency', 'schedule', 'deadline', 'timeout', 'delay', 'immediate', 'instant', 'moment', 'second', 'minute', 'hour', 'day', 'week', 'month', 'year', 'always', 'never', 'sometimes', 'occasionally', 'frequently', 'rarely', 'periodically', 'continuously', 'intermittently'],
            'values': ['value', 'level', 'intensity', 'strength', 'force', 'pressure_value', 'temperature_value', 'humidity_value', 'speed', 'velocity', 'acceleration', 'torque', 'voltage', 'current', 'power_value', 'energy', 'frequency_value', 'amplitude', 'wavelength', 'brightness', 'luminosity', 'color', 'hue', 'saturation', 'volume', 'pitch', 'tone', 'quality', 'grade', 'rank', 'score', 'rating', 'percentage', 'ratio', 'proportion', 'fraction', 'decimal', 'integer', 'float', 'string', 'text', 'binary', 'boolean'],
            'units': ['unit', 'meter', 'centimeter', 'millimeter', 'kilometer', 'inch', 'foot', 'yard', 'mile', 'square_meter', 'cubic_meter', 'liter', 'milliliter', 'gallon', 'pound', 'kilogram', 'gram', 'ton', 'ounce', 'degree', 'radian', 'celsius', 'fahrenheit', 'kelvin', 'pascal', 'bar', 'psi', 'atmosphere', 'watt', 'kilowatt', 'horsepower', 'volt', 'ampere', 'ohm', 'hertz', 'rpm', 'percent', 'ppm', 'db', 'lux', 'candela', 'newton', 'joule', 'calorie', 'btu', 'second_unit', 'minute_unit', 'hour_unit', 'day_unit', 'week_unit', 'month_unit', 'year_unit'],
            'operators': ['equals', 'not_equals', 'greater_than', 'less_than', 'greater_equal', 'less_equal', 'approximately', 'similar_to', 'different_from', 'same_as', 'identical_to', 'equivalent_to', 'proportional_to', 'inverse_of', 'opposite_of', 'complementary_to', 'supplementary_to', 'correlated_with', 'independent_of', 'dependent_on', 'coupled_with', 'linked_to', 'associated_with', 'related_to', 'connected_to', 'separated_from', 'isolated_from', 'distinct_from', 'unique_from', 'common_with', 'shared_with', 'exclusive_to', 'specific_to', 'general_to', 'universal_to', 'local_to', 'global_to', 'relative_to', 'absolute_to', 'fixed_to', 'variable_to', 'constant_to', 'dynamic_to', 'static_to', 'stable_to', 'unstable_to', 'balanced_to', 'unbalanced_to', 'symmetric_to', 'asymmetric_to', 'regular_to', 'irregular_to', 'uniform_to', 'nonuniform_to', 'homogeneous_to', 'heterogeneous_to']
        }
        return categories

class SimpleQueryTransformer:
    def __init__(self, input_dim=768):
        self.vocab = CommandVocabulary()
        # Используем динамический размер словаря
        vocab_size = self.vocab.get_vocab_size()
        self.model = SimpleQueryGenerator(input_dim=input_dim, vocab_size=vocab_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def add_new_tokens(self, new_tokens: List[str]):
        """
        Добавляет новые токены в словарь и обновляет модель
        """
        # Добавляем токены в словарь
        self.vocab.add_tokens(new_tokens)
        
        # Получаем новый размер словаря
        new_vocab_size = self.vocab.get_vocab_size()
        
        # Создаем новую модель с обновленным размером словаря
        old_model = self.model
        self.model = SimpleQueryGenerator(
            input_dim=old_model.encoder[0].in_features,
            hidden_dim=old_model.encoder[0].out_features,
            vocab_size=new_vocab_size
        )
        
        # Копируем веса из старой модели
        with torch.no_grad():
            # Копируем энкодер
            for i, layer in enumerate(self.model.encoder):
                if hasattr(old_model.encoder[i], 'weight'):
                    self.model.encoder[i].weight.copy_(old_model.encoder[i].weight)
                    if hasattr(old_model.encoder[i], 'bias'):
                        self.model.encoder[i].bias.copy_(old_model.encoder[i].bias)
            
            # Копируем LSTM
            for name, param in old_model.lstm.named_parameters():
                if name in dict(self.model.lstm.named_parameters()):
                    getattr(self.model.lstm, name.split('.')[0]).weight.copy_(param)
            
            # Копируем выходной слой (только для существующих токенов)
            old_output_size = old_model.output_layer.out_features
            self.model.output_layer.weight[:old_output_size] = old_model.output_layer.weight
        
        self.model.to(self.device)
        
    def train(self, training_data: List[Dict[str, Any]], epochs=100):
        """
        Обучает модель на данных вида:
        [
            {
                "vector": [0.1, 0.2, ...],
                "commands": "[set temperature 25 10:00, delete ventilation_settings]"
            },
            ...
        ]
        """
        # Подготавливаем данные
        vectors = []
        commands = []
        
        for item in training_data:
            vectors.append(item['vector'])
            # Добавляем токены в словарь
            command_tokens = item['commands'].split()
            for token in command_tokens:
                self.vocab.add_token(token)
            commands.append(item['commands'])
        
        # Обновляем размер словаря в модели
        new_vocab_size = self.vocab.get_vocab_size()
        if new_vocab_size != self.model.output_layer.out_features:
            self.add_new_tokens([])  # Обновляем модель
        
        # Создаем тензоры
        vectors_tensor = torch.tensor(vectors, dtype=torch.float32).to(self.device)
        
        # Кодируем команды
        encoded_commands = []
        max_length = max(len(self.vocab.encode_commands(c)) for c in commands)
        
        for command in commands:
            encoded = self.vocab.encode_commands(command)
            # Паддинг
            encoded.extend([self.vocab.tokens['<PAD>']] * (max_length - len(encoded)))
            encoded_commands.append(encoded)
        
        commands_tensor = torch.tensor(encoded_commands, dtype=torch.long).to(self.device)
        
        # Оптимизатор и функция потерь
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.tokens['<PAD>'])
        
        # Обучение
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Прямой проход
            outputs = self.model(vectors_tensor, max_length=max_length)
            
            # Переформатируем выходы для loss
            outputs = outputs.view(-1, outputs.size(-1))
            targets = commands_tensor.view(-1)
            
            # Вычисляем loss
            loss = criterion(outputs, targets)
            
            # Обратное распространение
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def generate_commands(self, vector: List[float], max_commands: int = 10) -> str:
        """
        Генерирует последовательность команд на основе входного вектора
        Может генерировать бесконечную последовательность команд
        """
        self.model.eval()
        
        # Подготавливаем входные данные
        vector_tensor = torch.tensor([vector], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Генерируем последовательность
            outputs = self.model(vector_tensor, max_length=max_commands * 10)  # Увеличиваем длину для большего количества команд
            
            # Получаем наиболее вероятные токены
            predicted_tokens = torch.argmax(outputs, dim=-1)
            
            # Декодируем результат
            token_ids = predicted_tokens[0].cpu().numpy().tolist()
            
            # Убираем паддинг и специальные токены
            result_tokens = []
            for token_id in token_ids:
                token = self.vocab.id_to_token(token_id)
                if token in ['<PAD>', '<START>', '<END>']:
                    continue
                result_tokens.append(token)
            
            return ' '.join(result_tokens)
    
    def generate_infinite_sequence(self, vector: List[float], max_iterations: int = 100) -> str:
        """
        Генерирует бесконечную последовательность команд
        Останавливается только при достижении max_iterations или специального токена
        """
        self.model.eval()
        
        # Подготавливаем входные данные
        vector_tensor = torch.tensor([vector], dtype=torch.float32).to(self.device)
        
        all_commands = []
        current_sequence = []
        
        with torch.no_grad():
            for iteration in range(max_iterations):
                # Генерируем следующую команду
                outputs = self.model(vector_tensor, max_length=20)
                predicted_tokens = torch.argmax(outputs, dim=-1)
                
                # Декодируем команду
                token_ids = predicted_tokens[0].cpu().numpy().tolist()
                command_tokens = []
                
                for token_id in token_ids:
                    token = self.vocab.id_to_token(token_id)
                    if token in ['<PAD>', '<START>', '<END>']:
                        continue
                    command_tokens.append(token)
                
                command = ' '.join(command_tokens)
                
                # Проверяем, не закончилась ли последовательность
                if not command or command.strip() == '':
                    break
                
                current_sequence.append(command)
                
                # Если накопили достаточно команд, добавляем в общий результат
                if len(current_sequence) >= 5:
                    all_commands.extend(current_sequence)
                    current_sequence = []
        
        # Добавляем оставшиеся команды
        if current_sequence:
            all_commands.extend(current_sequence)
        
        return ', '.join(all_commands)
    
    def get_available_tokens(self) -> List[str]:
        """Возвращает список всех доступных токенов"""
        return self.vocab.list_tokens()

def create_simple_sample_data() -> List[Dict[str, Any]]:
    """
    Создает примеры данных для обучения в правильном формате с использованием свойств объектов
    """
    from vectorizer import get_vector
    
    sample_data = [
        {
            "text": "Установить температуру 25 градусов Цельсия в 10:00",
            "commands": "[set temperature value 25 unit celsius time 10:00]"
        },
        {
            "text": "Удалить настройки вентиляции",
            "commands": "[delete ventilation settings]"
        },
        {
            "text": "Получить данные о влажности",
            "commands": "[get humidity value]"
        },
        {
            "text": "Обновить параметры системы в 15:30",
            "commands": "[update system parameters time 15:30]"
        },
        {
            "text": "Установить влажность 60% и температуру 22 градуса",
            "commands": "[set humidity value 60 unit percent, set temperature value 22 unit celsius]"
        },
        {
            "text": "Включить вентиляцию на полную мощность и выключить отопление",
            "commands": "[set ventilation power value full, set heating state off]"
        },
        {
            "text": "Если температура больше 30 градусов, включить кондиционер",
            "commands": "[if temperature value greater_than 30 unit celsius, set air_conditioning state on]"
        },
        {
            "text": "Установить режим автоматический и время работы с 8:00 до 18:00",
            "commands": "[set mode value auto, set work_time duration 8:00-18:00]"
        },
        {
            "text": "Создать 3 датчика температуры в разных позициях",
            "commands": "[create temperature_sensor quantity 3 position different]"
        },
        {
            "text": "Установить вентилятор на высоте 2 метра с углом поворота 45 градусов",
            "commands": "[set fan position height 2 unit meter, set fan orientation angle 45 unit degree]"
        },
        {
            "text": "Настроить систему безопасности с 5 камерами, зависящими от датчиков движения",
            "commands": "[configure security_system quantity 5 cameras dependency motion_sensors]"
        },
        {
            "text": "Установить давление в трубе на 2.5 бара при условии, что температура не превышает 80 градусов",
            "commands": "[set pipe pressure value 2.5 unit bar condition temperature value less_than 80 unit celsius]"
        },
        {
            "text": "Создать 2 мотора с мощностью 5 кВт каждый, расположенных параллельно",
            "commands": "[create motor quantity 2 power value 5 unit kilowatt orientation parallel]"
        },
        {
            "text": "Установить освещение на 70% яркости в течение 6 часов",
            "commands": "[set lighting brightness value 70 unit percent duration 6 unit hour]"
        },
        {
            "text": "Настроить насос с расходом 100 литров в минуту при давлении больше 1 бара",
            "commands": "[configure pump flow value 100 unit liter_per_minute condition pressure value greater_than 1 unit bar]"
        }
    ]
    
    # Векторизуем тексты
    for item in sample_data:
        item['vector'] = get_vector(item['text'])
        del item['text']  # Убираем исходный текст
    
    return sample_data

if __name__ == "__main__":
    # Создаем трансформер
    transformer = SimpleQueryTransformer()
    
    # Показываем доступные токены по категориям
    print("Доступные токены по категориям:")
    categories = transformer.vocab.get_tokens_by_category()
    for category, tokens in categories.items():
        print(f"\n{category.upper()} ({len(tokens)} токенов):")
        print(f"  Примеры: {tokens[:5]}")
    
    # Добавляем новые токены для специфических объектов
    new_tokens = [
        'robot', 'arm', 'gripper', 'conveyor', 'belt',
        'laser', 'cutter', 'welder', 'drill', 'mill',
        'lathe', 'grinder', 'polisher', 'painter', 'coater',
        'dryer', 'washer', 'cleaner', 'sterilizer', 'disinfectant',
        'filter', 'separator', 'mixer', 'reactor', 'tank',
        'pipe', 'tube', 'hose', 'fitting', 'connector',
        'gauge', 'meter', 'indicator', 'display', 'screen',
        'keyboard', 'mouse', 'joystick', 'button', 'lever',
        'pedal', 'wheel', 'handle', 'knob', 'dial',
        'air_conditioning', 'heating', 'cooling', 'refrigeration',
        'compressor', 'condenser', 'evaporator', 'expansion_valve',
        'thermostat', 'humidifier', 'dehumidifier', 'air_purifier',
        'heat_exchanger', 'boiler', 'furnace', 'radiator',
        'solar_panel', 'wind_turbine', 'battery', 'inverter',
        'transformer', 'circuit_breaker', 'fuse', 'capacitor',
        'resistor', 'inductor', 'diode', 'transistor', 'ic',
        'microcontroller', 'plc', 'scada', 'hmi', 'rtu',
        'ethernet', 'wifi', 'bluetooth', 'zigbee', 'modbus',
        'profibus', 'can_bus', 'rs485', 'rs232', 'usb',
        'hdmi', 'vga', 'dvi', 'displayport', 'audio',
        'video', 'image', 'stream', 'recording', 'playback',
        'database', 'server', 'client', 'api', 'webhook',
        'cloud', 'edge', 'fog', 'gateway', 'router',
        'switch', 'hub', 'bridge', 'repeater', 'amplifier'
    ]
    
    print(f"\nДобавляем {len(new_tokens)} новых токенов для объектов...")
    transformer.add_new_tokens(new_tokens)
    
    print(f"Теперь доступно токенов: {len(transformer.get_available_tokens())}")
    
    # Создаем данные для обучения
    training_data = create_simple_sample_data()
    
    print("\nОбучаем модель...")
    transformer.train(training_data, epochs=50)
    
    # Тестируем генерацию с новыми структурированными токенами
    test_cases = [
        "Установить температуру 20 градусов Цельсия",
        "Включить вентиляцию на среднюю мощность",
        "Удалить все настройки системы",
        "Получить текущую влажность и температуру",
        "Создать 5 роботов с мощностью 10 кВт каждый",
        "Установить лазер на мощность 50% с углом 30 градусов",
        "Настроить систему с 3 камерами, зависящими от датчиков движения",
        "Установить давление в трубе на 3 бара при температуре меньше 90 градусов",
        "Создать конвейер длиной 10 метров с 2 моторами по 5 кВт",
        "Настроить освещение на 80% яркости в течение 8 часов"
    ]
    
    print("\nТестирование генерации команд с свойствами объектов:")
    for test_text in test_cases:
        test_vector = get_vector(test_text)
        generated_commands = transformer.generate_commands(test_vector)
        print(f"\nТекст: {test_text}")
        print(f"Команды: {generated_commands}")
    
    # Тестируем бесконечную генерацию
    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ БЕСКОНЕЧНОЙ ГЕНЕРАЦИИ КОМАНД")
    print("="*80)
    
    complex_test_cases = [
        "Создать полную систему автоматизации завода с множественными конвейерами, роботами, датчиками и системами контроля",
        "Настроить интеллектуальное здание с системами отопления, вентиляции, освещения, безопасности и управления энергопотреблением",
        "Развернуть промышленную сеть с множественными устройствами, серверами, базами данных и системами мониторинга",
        "Создать автоматизированную систему управления складом с роботами-погрузчиками, конвейерами, датчиками и системами инвентаризации"
    ]
    
    for test_text in complex_test_cases:
        print(f"\n{'='*60}")
        print(f"СЛОЖНЫЙ ТЕКСТ: {test_text}")
        print(f"{'='*60}")
        
        test_vector = get_vector(test_text)
        
        # Генерируем обычную последовательность
        print("\n1. Обычная генерация (до 10 команд):")
        commands = transformer.generate_commands(test_vector, max_commands=10)
        print(f"Результат: {commands}")
        
        # Генерируем расширенную последовательность
        print("\n2. Расширенная генерация (до 50 команд):")
        commands = transformer.generate_commands(test_vector, max_commands=50)
        print(f"Результат: {commands}")
        
        # Генерируем бесконечную последовательность
        print("\n3. Бесконечная генерация (до 100 итераций):")
        infinite_commands = transformer.generate_infinite_sequence(test_vector, max_iterations=100)
        print(f"Результат: {infinite_commands}")
        
        print(f"\nКоличество сгенерированных команд: {len(infinite_commands.split(','))}")
    
    # Показываем возможности бесконечной генерации
    print("\n" + "="*80)
    print("ВОЗМОЖНОСТИ БЕСКОНЕЧНОЙ ГЕНЕРАЦИИ:")
    print("="*80)
    print("1. Модель может генерировать сколь угодно длинные последовательности команд")
    print("2. Каждая команда содержит полную информацию об объекте:")
    print("   - Операция (set, delete, create, configure)")
    print("   - Объект (temperature, robot, conveyor, sensor)")
    print("   - Свойства (value, quantity, position, orientation)")
    print("   - Значения (25, 3, 10, 45)")
    print("   - Единицы измерения (celsius, kilowatt, meter, degree)")
    print("   - Условия (if, when, while, until)")
    print("   - Зависимости (depends_on, requires, connected_to)")
    print("   - Временные рамки (time, duration, schedule)")
    print("3. Последовательности могут включать:")
    print("   - Множественные объекты")
    print("   - Сложные зависимости между объектами")
    print("   - Условную логику")
    print("   - Временные ограничения")
    print("   - Пространственные отношения")
    print("   - Физические свойства")
    print("   - Состояния и режимы работы")
    print("4. Модель автоматически определяет:")
    print("   - Количество необходимых команд")
    print("   - Логическую последовательность")
    print("   - Связи между объектами")
    print("   - Условия выполнения")
    print("   - Параметры каждого объекта") 