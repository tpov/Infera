from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from enum import Enum
import re
import random

class AnalysisCycle(Enum):
    SURFACE = "surface"           # Поверхностный анализ
    PATTERN = "pattern"           # Поиск паттернов
    HIDDEN = "hidden"             # Скрытые смыслы
    RELATIONAL = "relational"     # Отношения между элементами
    INFERENTIAL = "inferential"   # Логические выводы
    SYNTHETIC = "synthetic"       # Синтез информации

@dataclass
class AnalysisResult:
    """Результат анализа"""
    cycle_type: AnalysisCycle
    insights: List[str]
    confidence: float
    patterns_found: List[str]
    hidden_meanings: List[str]
    relationships: List[Dict[str, Any]]
    next_focus: Optional[str] = None

@dataclass
class FocusedContext:
    """Фокусированный контекст для анализа"""
    primary_focus: str
    secondary_focus: List[str]
    context_window: List[str]
    analysis_depth: int
    confidence_threshold: float
    max_patterns: int

class CyclicAnalysisEngine:
    """Движок циклического анализа"""
    
    def __init__(self, max_cycles: int = 10):
        self.max_cycles = max_cycles
        self.analysis_history = []
        self.pattern_database = {}
        self.relationship_graph = {}
        
    def analyze_with_cycles(self, user_query: str) -> Dict[str, Any]:
        """Анализирует запрос через циклы анализа"""
        
        print(f"Начинаем циклический анализ запроса: {user_query}")
        
        current_context = self._create_initial_context(user_query)
        analysis_results = []
        
        for cycle_num in range(self.max_cycles):
            print(f"\n--- Цикл анализа {cycle_num + 1} ---")
            
            # Определяем тип анализа для текущего цикла
            cycle_type = self._determine_cycle_type(cycle_num, current_context)
            
            # Выполняем анализ
            result = self._perform_cycle_analysis(user_query, cycle_type, current_context)
            analysis_results.append(result)
            
            # Обновляем контекст на основе результатов
            current_context = self._update_context_from_analysis(current_context, result)
            
            # Проверяем, нужно ли остановиться
            if self._should_stop_analysis(result, cycle_num):
                print(f"Останавливаем анализ на цикле {cycle_num + 1}")
                break
        
        # Синтезируем финальный результат
        final_result = self._synthesize_results(analysis_results, user_query)
        
        return final_result
    
    def _create_initial_context(self, query: str) -> FocusedContext:
        """Создает начальный контекст"""
        # Извлекаем ключевые слова
        keywords = self._extract_keywords(query)
        
        return FocusedContext(
            primary_focus=keywords[0] if keywords else "general",
            secondary_focus=keywords[1:3] if len(keywords) > 1 else [],
            context_window=keywords[:5],
            analysis_depth=1,
            confidence_threshold=0.7,
            max_patterns=5
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Извлекает ключевые слова из запроса"""
        # Простая реализация - в реальности нужен более сложный анализ
        words = re.findall(r'\w+', query.lower())
        
        # Фильтруем стоп-слова
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'к', 'у', 'о', 'об', 'за'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Ограничиваем количество
    
    def _determine_cycle_type(self, cycle_num: int, context: FocusedContext) -> AnalysisCycle:
        """Определяет тип анализа для цикла"""
        
        if cycle_num == 0:
            return AnalysisCycle.SURFACE
        
        elif cycle_num == 1:
            return AnalysisCycle.PATTERN
        
        elif cycle_num == 2:
            return AnalysisCycle.HIDDEN
        
        elif cycle_num == 3:
            return AnalysisCycle.RELATIONAL
        
        elif cycle_num == 4:
            return AnalysisCycle.INFERENTIAL
        
        else:
            return AnalysisCycle.SYNTHETIC
    
    def _perform_cycle_analysis(self, query: str, cycle_type: AnalysisCycle, context: FocusedContext) -> AnalysisResult:
        """Выполняет анализ для конкретного цикла"""
        
        if cycle_type == AnalysisCycle.SURFACE:
            return self._surface_analysis(query, context)
        
        elif cycle_type == AnalysisCycle.PATTERN:
            return self._pattern_analysis(query, context)
        
        elif cycle_type == AnalysisCycle.HIDDEN:
            return self._hidden_meaning_analysis(query, context)
        
        elif cycle_type == AnalysisCycle.RELATIONAL:
            return self._relational_analysis(query, context)
        
        elif cycle_type == AnalysisCycle.INFERENTIAL:
            return self._inferential_analysis(query, context)
        
        else:  # SYNTHETIC
            return self._synthetic_analysis(query, context)
    
    def _surface_analysis(self, query: str, context: FocusedContext) -> AnalysisResult:
        """Поверхностный анализ"""
        insights = []
        patterns = []
        
        # Анализируем структуру запроса
        if '?' in query:
            insights.append("Запрос содержит вопрос")
            patterns.append("question_pattern")
        
        if any(word in query.lower() for word in ['если', 'то', 'когда']):
            insights.append("Запрос содержит условную логику")
            patterns.append("conditional_pattern")
        
        if any(word in query.lower() for word in ['создать', 'установить', 'настроить']):
            insights.append("Запрос содержит действия создания")
            patterns.append("creation_pattern")
        
        return AnalysisResult(
            cycle_type=AnalysisCycle.SURFACE,
            insights=insights,
            confidence=0.8,
            patterns_found=patterns,
            hidden_meanings=[],
            relationships=[],
            next_focus="pattern_detection"
        )
    
    def _pattern_analysis(self, query: str, context: FocusedContext) -> AnalysisResult:
        """Анализ паттернов"""
        patterns = []
        hidden_meanings = []
        
        # Ищем повторяющиеся паттерны
        words = query.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Находим частые слова
        for word, freq in word_freq.items():
            if freq > 1:
                patterns.append(f"repetition_{word}")
        
        # Анализируем синтаксические паттерны
        if re.search(r'\w+\s+и\s+\w+', query):
            patterns.append("conjunction_pattern")
        
        if re.search(r'\w+\s+или\s+\w+', query):
            patterns.append("disjunction_pattern")
        
        # Ищем скрытые смыслы в паттернах
        if "система" in query.lower() and "автоматизация" in query.lower():
            hidden_meanings.append("Пользователь хочет создать автоматизированную систему")
        
        return AnalysisResult(
            cycle_type=AnalysisCycle.PATTERN,
            insights=[f"Найдено {len(patterns)} паттернов"],
            confidence=0.7,
            patterns_found=patterns,
            hidden_meanings=hidden_meanings,
            relationships=[],
            next_focus="hidden_meaning"
        )
    
    def _hidden_meaning_analysis(self, query: str, context: FocusedContext) -> AnalysisResult:
        """Анализ скрытых смыслов"""
        hidden_meanings = []
        relationships = []
        
        # Анализируем эмоциональный подтекст
        emotional_words = {
            'хорошо': 'positive_emotion',
            'плохо': 'negative_emotion',
            'нужно': 'necessity',
            'важно': 'importance',
            'срочно': 'urgency'
        }
        
        for word, emotion_type in emotional_words.items():
            if word in query.lower():
                hidden_meanings.append(f"Эмоциональный контекст: {emotion_type}")
        
        # Анализируем намерения
        intention_words = {
            'создать': 'creation_intent',
            'улучшить': 'improvement_intent',
            'решить': 'problem_solving_intent',
            'оптимизировать': 'optimization_intent'
        }
        
        for word, intent_type in intention_words.items():
            if word in query.lower():
                hidden_meanings.append(f"Намерение: {intent_type}")
        
        # Анализируем отношения между концепциями
        concepts = self._extract_keywords(query)
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                relationships.append({
                    'type': 'conceptual_relation',
                    'source': concept1,
                    'target': concept2,
                    'strength': 0.6
                })
        
        return AnalysisResult(
            cycle_type=AnalysisCycle.HIDDEN,
            insights=[f"Обнаружено {len(hidden_meanings)} скрытых смыслов"],
            confidence=0.6,
            patterns_found=[],
            hidden_meanings=hidden_meanings,
            relationships=relationships,
            next_focus="relational_analysis"
        )
    
    def _relational_analysis(self, query: str, context: FocusedContext) -> AnalysisResult:
        """Анализ отношений"""
        relationships = []
        
        # Анализируем причинно-следственные связи
        if 'если' in query.lower() and 'то' in query.lower():
            # Извлекаем условие и следствие
            if_match = re.search(r'если\s+([^,]+?)\s+то\s+([^,]+)', query.lower())
            if if_match:
                condition = if_match.group(1).strip()
                consequence = if_match.group(2).strip()
                relationships.append({
                    'type': 'causal_relation',
                    'source': condition,
                    'target': consequence,
                    'strength': 0.9
                })
        
        # Анализируем зависимости
        dependency_patterns = [
            (r'(\w+)\s+зависит\s+от\s+(\w+)', 'dependency'),
            (r'(\w+)\s+влияет\s+на\s+(\w+)', 'influence'),
            (r'(\w+)\s+связан\s+с\s+(\w+)', 'association')
        ]
        
        for pattern, relation_type in dependency_patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                relationships.append({
                    'type': relation_type,
                    'source': match[0],
                    'target': match[1],
                    'strength': 0.8
                })
        
        return AnalysisResult(
            cycle_type=AnalysisCycle.RELATIONAL,
            insights=[f"Обнаружено {len(relationships)} отношений"],
            confidence=0.8,
            patterns_found=[],
            hidden_meanings=[],
            relationships=relationships,
            next_focus="inferential_analysis"
        )
    
    def _inferential_analysis(self, query: str, context: FocusedContext) -> AnalysisResult:
        """Логический анализ"""
        insights = []
        
        # Делаем логические выводы на основе предыдущих анализов
        if context.primary_focus == "система":
            insights.append("Пользователь работает с системной архитектурой")
        
        if "автоматизация" in query.lower():
            insights.append("Требуется автоматизация процессов")
        
        if "датчик" in query.lower():
            insights.append("Необходимо управление сенсорами")
        
        # Анализируем логические операторы
        logical_operators = ['и', 'или', 'не', 'если', 'то', 'когда']
        found_operators = [op for op in logical_operators if op in query.lower()]
        
        if found_operators:
            insights.append(f"Используются логические операторы: {', '.join(found_operators)}")
        
        return AnalysisResult(
            cycle_type=AnalysisCycle.INFERENTIAL,
            insights=insights,
            confidence=0.7,
            patterns_found=[],
            hidden_meanings=[],
            relationships=[],
            next_focus="synthesis"
        )
    
    def _synthetic_analysis(self, query: str, context: FocusedContext) -> AnalysisResult:
        """Синтетический анализ"""
        # Объединяем все предыдущие результаты
        all_insights = []
        all_patterns = []
        all_hidden_meanings = []
        all_relationships = []
        
        # Собираем данные из истории анализа
        for result in self.analysis_history[-3:]:  # Последние 3 результата
            all_insights.extend(result.insights)
            all_patterns.extend(result.patterns_found)
            all_hidden_meanings.extend(result.hidden_meanings)
            all_relationships.extend(result.relationships)
        
        # Синтезируем общий вывод
        synthetic_insight = self._synthesize_insights(all_insights, all_patterns, all_hidden_meanings)
        
        return AnalysisResult(
            cycle_type=AnalysisCycle.SYNTHETIC,
            insights=[synthetic_insight],
            confidence=0.9,
            patterns_found=list(set(all_patterns)),
            hidden_meanings=list(set(all_hidden_meanings)),
            relationships=all_relationships,
            next_focus=None
        )
    
    def _synthesize_insights(self, insights: List[str], patterns: List[str], hidden_meanings: List[str]) -> str:
        """Синтезирует общий вывод"""
        if not insights and not patterns and not hidden_meanings:
            return "Недостаточно данных для синтеза"
        
        synthesis_parts = []
        
        if insights:
            synthesis_parts.append(f"Основные наблюдения: {len(insights)}")
        
        if patterns:
            synthesis_parts.append(f"Обнаружено паттернов: {len(patterns)}")
        
        if hidden_meanings:
            synthesis_parts.append(f"Скрытых смыслов: {len(hidden_meanings)}")
        
        return "; ".join(synthesis_parts)
    
    def _update_context_from_analysis(self, context: FocusedContext, result: AnalysisResult) -> FocusedContext:
        """Обновляет контекст на основе результатов анализа"""
        
        # Обновляем фокус на основе найденных паттернов
        if result.patterns_found:
            new_focus = result.patterns_found[0] if result.patterns_found else context.primary_focus
        else:
            new_focus = context.primary_focus
        
        # Увеличиваем глубину анализа
        new_depth = context.analysis_depth + 1
        
        # Обновляем порог уверенности
        new_threshold = min(0.95, context.confidence_threshold + 0.05)
        
        return FocusedContext(
            primary_focus=new_focus,
            secondary_focus=context.secondary_focus,
            context_window=context.context_window,
            analysis_depth=new_depth,
            confidence_threshold=new_threshold,
            max_patterns=context.max_patterns
        )
    
    def _should_stop_analysis(self, result: AnalysisResult, cycle_num: int) -> bool:
        """Определяет, нужно ли остановить анализ"""
        
        # Останавливаем, если достигли максимального количества циклов
        if cycle_num >= self.max_cycles - 1:
            return True
        
        # Останавливаем, если уверенность высокая
        if result.confidence > 0.9:
            return True
        
        # Останавливаем, если нет новых инсайтов
        if not result.insights and not result.patterns_found:
            return True
        
        return False
    
    def _synthesize_results(self, analysis_results: List[AnalysisResult], original_query: str) -> Dict[str, Any]:
        """Синтезирует финальные результаты"""
        
        # Собираем все результаты
        all_insights = []
        all_patterns = []
        all_hidden_meanings = []
        all_relationships = []
        
        for result in analysis_results:
            all_insights.extend(result.insights)
            all_patterns.extend(result.patterns_found)
            all_hidden_meanings.extend(result.hidden_meanings)
            all_relationships.extend(result.relationships)
        
        # Убираем дубликаты
        all_insights = list(set(all_insights))
        all_patterns = list(set(all_patterns))
        all_hidden_meanings = list(set(all_hidden_meanings))
        
        # Вычисляем общую уверенность
        total_confidence = sum(result.confidence for result in analysis_results)
        avg_confidence = total_confidence / len(analysis_results) if analysis_results else 0.0
        
        return {
            "original_query": original_query,
            "analysis_cycles": len(analysis_results),
            "total_insights": len(all_insights),
            "total_patterns": len(all_patterns),
            "total_hidden_meanings": len(all_hidden_meanings),
            "total_relationships": len(all_relationships),
            "average_confidence": avg_confidence,
            "insights": all_insights,
            "patterns": all_patterns,
            "hidden_meanings": all_hidden_meanings,
            "relationships": all_relationships,
            "cycle_details": [
                {
                    "cycle": i + 1,
                    "type": result.cycle_type.value,
                    "confidence": result.confidence,
                    "insights_count": len(result.insights)
                }
                for i, result in enumerate(analysis_results)
            ]
        }

class CommandGeneratorFromAnalysis:
    """Генератор команд на основе анализа"""
    
    def __init__(self):
        self.analysis_engine = CyclicAnalysisEngine()
        self.command_templates = self._load_command_templates()
    
    def _load_command_templates(self) -> Dict[str, List[str]]:
        """Загружает шаблоны команд"""
        return {
            "creation": [
                "create {object} quantity {quantity}",
                "set {object} value {value}",
                "configure {object} {property} {value}"
            ],
            "conditional": [
                "if {condition} then {action}",
                "when {condition} do {action}",
                "set condition {condition} action {action}"
            ],
            "pattern": [
                "set pattern_handler quantity 1",
                "create pattern_detector quantity 1",
                "configure pattern_analyzer {pattern_type}"
            ]
        }
    
    def generate_commands_from_analysis(self, user_query: str) -> str:
        """Генерирует команды на основе циклического анализа"""
        
        # Выполняем анализ
        analysis_result = self.analysis_engine.analyze_with_cycles(user_query)
        
        # Генерируем команды на основе результатов
        commands = self._generate_commands_from_insights(analysis_result)
        
        return commands
    
    def _generate_commands_from_insights(self, analysis_result: Dict[str, Any]) -> str:
        """Генерирует команды из инсайтов анализа"""
        commands = []
        
        # Анализируем паттерны
        for pattern in analysis_result['patterns']:
            if 'creation' in pattern:
                commands.append("create system quantity 1")
            elif 'conditional' in pattern:
                commands.append("if condition then action")
            elif 'repetition' in pattern:
                commands.append("set pattern_handler quantity 1")
        
        # Анализируем скрытые смыслы
        for meaning in analysis_result['hidden_meanings']:
            if 'автоматизация' in meaning.lower():
                commands.append("create automation_system quantity 1")
            elif 'эмоциональный' in meaning.lower():
                commands.append("set emotional_state value positive")
            elif 'намерение' in meaning.lower():
                commands.append("create intent_handler quantity 1")
        
        # Анализируем отношения
        for relationship in analysis_result['relationships']:
            if relationship['type'] == 'causal_relation':
                commands.append(f"set dependency {relationship['source']} {relationship['target']}")
            elif relationship['type'] == 'influence':
                commands.append(f"create influence_handler quantity 1")
        
        return f"[{', '.join(commands)}]" if commands else ""

def test_cyclic_analysis():
    """Тестирует циклический анализ"""
    generator = CommandGeneratorFromAnalysis()
    
    test_queries = [
        "Создать систему автоматизации с датчиками температуры",
        "Если температура высокая, то включить вентиляцию",
        "Нужно улучшить производительность системы"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Тестируем запрос: {query}")
        print(f"{'='*80}")
        
        commands = generator.generate_commands_from_analysis(query)
        print(f"Сгенерированные команды: {commands}")

if __name__ == "__main__":
    test_cyclic_analysis()