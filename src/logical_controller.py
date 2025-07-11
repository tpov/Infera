import json
import random
import os
from typing import Dict, Any, List, Optional, Tuple

try:
    from state_manager import StateManager
    from controller import NLUController
    from nlg_model import NLGModel, THINKING_OUTPUT_SEPARATOR
except ImportError:
    if __name__ == '__main__':
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.state_manager import StateManager
        from src.controller import NLUController
        from src.nlg_model import NLGModel, THINKING_OUTPUT_SEPARATOR
    else:
        raise

class LogicalController:
    def __init__(self, nlu_controller: NLUController, state_manager: StateManager, nlg_model: NLGModel,
                 inference_engine = None, kb_manager = None): # Добавляем inference_engine и kb_manager как опциональные
        self.nlu_controller = nlu_controller
        self.state_manager = state_manager
        self.nlg_model = nlg_model
        self.inference_engine = inference_engine # Будет None, если не передан
        if self.inference_engine is None:
            try:
                from inference_engine import SimpleInferenceEngine # Попытка импорта по умолчанию
                self.inference_engine = SimpleInferenceEngine()
                print("LogicalController: Initialized default SimpleInferenceEngine.")
            except ImportError:
                print("LogicalController: SimpleInferenceEngine not available, inference will be skipped.")
                self.inference_engine = None

        self.kb_manager = kb_manager
        self.MAX_THINKING_ITERATIONS = 5
        print("LogicalController initialized.")

    def _handle_nlu_result(self, nlu_result: Dict[str, Any],
                           current_thinking_goal: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str]]:
        action_response_detail = "Action processed."
        nlg_task_type_override = None

        if current_thinking_goal:
            pass

        intent = nlu_result.get("intent")
        slots = nlu_result.get("slots", [])

        if intent == "NLU_ERROR": return f"NLU Error: {nlu_result.get('error', 'Unknown')}", "GENERATE_RESPONSE"
        if intent == "UNKNOWN_INTENT": return "Sorry, I couldn't determine the intent.", "GENERATE_RESPONSE"
        if intent == "OUT_OF_SCOPE": return "I can manage items... Not sure about that.", "GENERATE_RESPONSE"

        if intent == "ADD_ENTITY_WITH_DETAILS" or intent == "ADD_ENTITY_COUNT":
            entity_name = next((s["value"] for s in slots if s["type"] == "ENTITY_NAME"), None)
            count_str = next((s["value"] for s in slots if s["type"] == "COUNT"), None)
            location = next((s["value"] for s in slots if s["type"] == "LOCATION"), None)
            attribute_value = next((s["value"] for s in slots if s["type"] == "ATTRIBUTE_VALUE"), None)
            count_int = None
            if count_str:
                if count_str.isdigit(): count_int = int(count_str)
                elif count_str.lower() in ["one", "a", "an"]: count_int = 1
                elif count_str.lower() == "two": count_int = 2
                else:
                    try: count_int = int(float(count_str))
                    except ValueError: print(f"Warning: Could not parse count_str '{count_str}' to int.")

            if entity_name and count_int is not None:
                attrs_to_add = {"count": count_int}
                if location: attrs_to_add["location"] = location
                if attribute_value: attrs_to_add["color"] = attribute_value
                existing_entity = self.state_manager.get_entity(entity_name)
                op_type = "updated"
                if not existing_entity:
                    op_type = "added"; self.state_manager.add_or_update_entity(entity_name, attrs_to_add)
                else:
                    new_c = existing_entity.get("count", 0) + count_int if intent == "ADD_ENTITY_COUNT" or "count" in attrs_to_add else existing_entity.get("count", 0)
                    final_attrs = {**existing_entity, **attrs_to_add, "count": new_c}; self.state_manager.add_or_update_entity(entity_name, final_attrs)
                action_response_detail = f"Entity '{entity_name}' {op_type}. Current: {self.state_manager.get_entity(entity_name)}"
            else:
                action_response_detail = f"Could not add/update: missing name or valid count. Name='{entity_name}', CountStr='{count_str}'"
                if not entity_name: nlu_result["meta_missing_slot"] = "entity_name"; nlg_task_type_override = "REQUEST_CLARIFICATION"
                elif count_int is None and count_str is not None : nlu_result["meta_missing_slot"] = f"valid count for '{count_str}'"; nlg_task_type_override = "REQUEST_CLARIFICATION"
                elif count_int is None : nlu_result["meta_missing_slot"] = "count"; nlg_task_type_override = "REQUEST_CLARIFICATION"
                if intent == "ADD_ENTITY_WITH_DETAILS" and not nlg_task_type_override:
                    if not location: nlu_result["meta_missing_slot"] = "location"; nlg_task_type_override = "REQUEST_CLARIFICATION"
                    elif not attribute_value: nlu_result["meta_missing_slot"] = "attribute"; nlg_task_type_override = "REQUEST_CLARIFICATION"

        elif intent == "REMOVE_ENTITY_COUNT":
            entity_name = next((s["value"] for s in slots if s["type"] == "ENTITY_NAME"), None)
            count_str = next((s["value"] for s in slots if s["type"] == "COUNT"), "1"); count_int = 1
            if count_str.isdigit(): count_int = int(count_str)
            if entity_name:
                ex_entity = self.state_manager.get_entity(entity_name)
                if ex_entity:
                    cur_c = ex_entity.get("count",0); new_c = max(0,cur_c-count_int)
                    self.state_manager.add_or_update_entity(entity_name, {"count":new_c})
                    action_response_detail = f"Removed {cur_c-new_c} of '{entity_name}'. Left: {new_c}."
                else: action_response_detail = f"Entity '{entity_name}' not found."
            else: action_response_detail = "Which entity to remove?"; nlu_result["meta_missing_slot"] = "entity name"; nlg_task_type_override = "REQUEST_CLARIFICATION"

        elif intent == "QUERY_COUNT":
            entity_name = next((s["value"] for s in slots if s["type"] == "ENTITY_NAME"), None)
            if entity_name:
                data = self.state_manager.get_entity(entity_name)
                action_response_detail = f"Count of '{entity_name}': {data.get('count','unknown')}." if data else f"'{entity_name}' not found."
            else: action_response_detail = "Which entity's count?"; nlu_result["meta_missing_slot"] = "entity name"; nlg_task_type_override = "REQUEST_CLARIFICATION"

        elif intent == "QUERY_LOCATION":
             entity_name = next((s["value"] for s in slots if s["type"] == "ENTITY_NAME"), None)
             if entity_name:
                data = self.state_manager.get_entity(entity_name)
                action_response_detail = f"'{entity_name}' is at '{data.get('location','unknown')}'." if data else f"'{entity_name}' not found."
             else: action_response_detail = "Which entity's location?"; nlu_result["meta_missing_slot"] = "entity name"; nlg_task_type_override = "REQUEST_CLARIFICATION"

        elif intent == "QUERY_ATTRIBUTE":
            entity_name = next((s["value"] for s in slots if s["type"] == "ENTITY_NAME"), None)
            attr_type = next((s["value"] for s in slots if s["type"] == "ATTRIBUTE_TYPE"), None)
            if entity_name and attr_type:
                data = self.state_manager.get_entity(entity_name)
                action_response_detail = f"{attr_type} of '{entity_name}': {data.get(attr_type.lower(),'unknown')}." if data else f"'{entity_name}' not found."
            else: action_response_detail = "Missing entity or attribute type."; nlu_result["meta_missing_slot"] = "entity/attribute"; nlg_task_type_override = "REQUEST_CLARIFICATION"

        elif intent == "GENERAL_GREETING": action_response_detail = "User greeted."
        elif intent == "GENERAL_GOODBYE": action_response_detail = "User said goodbye."
        elif intent == "QUERY_BUSINESS_PLAN":
            action_response_detail = "Processing business plan query."
            if self._is_complex_query_requiring_thought(nlu_result, self.state_manager.get_current_state()):
                 nlg_task_type_override = "THINK_AND_PLAN"
            else:
                nlu_result["meta_missing_slot"] = "your experience and skills"; nlg_task_type_override = "REQUEST_CLARIFICATION"
        else:
            action_response_detail = f"Intent '{intent}' received, specific logic pending."

        # === Вызов Inference Engine после обработки основного NLU и обновления состояния ===
        if self.inference_engine and not current_thinking_goal: # Не вызываем inference внутри "думания" пока, чтобы не усложнять снимки
            print("DEBUG: Running inference engine after NLU handling...")
            num_inferences = self.inference_engine.infer(self.state_manager, kb_manager=self.kb_manager)
            print(f"DEBUG: Inference engine made {num_inferences} new inferences/actions.")
            if num_inferences > 0:
                action_response_detail += f" (Additionally, {num_inferences} related facts/updates were processed by inference.)"
        # =================================================================================

        if current_thinking_goal and current_thinking_goal.get("current_snapshot_id"):
            self.state_manager.create_snapshot(current_thinking_goal["current_snapshot_id"])
        return action_response_detail, nlg_task_type_override

    def _is_complex_query_requiring_thought(self, nlu_result: Dict[str, Any], current_state: Dict[str, Any]) -> bool:
        intent = nlu_result.get("intent")
        if intent == "QUERY_BUSINESS_PLAN":
            user_profile = current_state.get("entities", {}).get("user_profile", {})
            # Считаем сложным, если есть хоть какая-то информация для старта "думания"
            return bool(user_profile.get("experience") or user_profile.get("skills") or nlu_result.get("slots"))
        # TODO: Добавить другие интенты, которые могут требовать "думания"
        return False

    def _create_thinking_goal(self, nlu_result: Dict[str, Any], base_snapshot_id: str) -> Dict[str, Any]:
        goal_id = f"think_{nlu_result.get('intent','task')}_{random.randint(1000,9999)}"
        solution_criteria = None
        if nlu_result.get("intent") == "QUERY_BUSINESS_PLAN":
            solution_criteria = {
                "type": "ENTITY_WITH_ATTRIBUTES",
                "entity_name": "business_plan_draft",
                "required_attributes": ["strategy", "budget_estimation", "marketing_channels", "target_audience_defined", "key_partnerships"]
            }
        goal = {
            "goal_id": goal_id, "goal_type": "SYSTEM_THINKING_PROCESS",
            "original_user_nlu": dict(nlu_result),
            "current_iteration_nlu": dict(nlu_result),
            "status": "thinking_initiated", "iteration": 0,
            "max_iterations": self.MAX_THINKING_ITERATIONS,
            "base_snapshot_id": base_snapshot_id,
            "current_snapshot_id": base_snapshot_id,
            "parent_nlu_for_ideas": dict(nlu_result), # NLU, на котором генерировался текущий набор идей
            "ideas_to_explore": [], "current_idea_index": 0,
            "history_ideas_tried": {},  # { "idea_text": {"status": "...", "snapshot_id": "..."}}
            "accumulated_solution_paths": [], # Список словарей с информацией о найденных решениях
            "solution_snapshot_id": None, "solution_details": None, # Для лучшего/финального решения
            "pending_user_question": None,
            "solution_criteria": solution_criteria
        }
        self.state_manager.add_goal(goal)
        print(f"Created thinking goal: {goal_id} based on snapshot {base_snapshot_id} with criteria: {solution_criteria}")
        return goal

    def _is_solution_found(self, thinking_goal: Dict[str, Any], current_state_for_check: Dict[str, Any]) -> bool:
        criteria = thinking_goal.get("solution_criteria")
        if not criteria: return False # Если критерии не заданы, решение не может быть найдено этим методом

        if criteria.get("type") == "ENTITY_WITH_ATTRIBUTES":
            entity_name = criteria.get("entity_name")
            required_attributes = criteria.get("required_attributes", [])
            entity_data = current_state_for_check.get("entities", {}).get(entity_name)
            if not entity_data: return False

            all_found = True
            for attr in required_attributes:
                if not entity_data.get(attr): # Проверяем наличие и не-пустоту (None, "")
                    all_found = False; break
            if all_found:
                print(f"DEBUG _is_solution_found: Criteria MET for '{entity_name}'. Details: {entity_data}")
                return True
        return False

    def _thinking_loop_iteration(self, thinking_goal: Dict[str, Any]) -> str:
        """Одна итерация цикла думания. Может быть рекурсивной."""

        if thinking_goal.get("status") == "thinking_initiated":
            thinking_goal["iteration"] = 0 # Инкремент до 1 будет ниже

        thinking_goal["iteration"] += 1
        current_iteration = thinking_goal["iteration"]
        goal_id = thinking_goal['goal_id']
        # Обновляем статус цели в StateManager в начале каждой итерации
        current_goal_status = f"thinking_iteration_{current_iteration}"
        thinking_goal["status"] = current_goal_status
        self.state_manager.update_goal_status(goal_id, current_goal_status, "goal_id")


        if current_iteration > thinking_goal.get("max_iterations", self.MAX_THINKING_ITERATIONS):
            print(f"Thinking limit reached for goal {goal_id}.")
            thinking_goal["status"] = "thinking_limit_reached"
            self.state_manager.update_goal_status(goal_id, thinking_goal["status"], "goal_id")
            if thinking_goal.get("base_snapshot_id"): self.state_manager.restore_snapshot(thinking_goal["base_snapshot_id"])
            # Сообщаем NLG о лимите и передаем накопленные решения, если есть
            nlu_for_final_resp = thinking_goal["original_user_nlu"].copy()
            nlu_for_final_resp["meta_prompt_info"] = "Thinking limit reached."
            if thinking_goal["accumulated_solution_paths"]:
                 nlu_for_final_resp["meta_solutions"] = thinking_goal["accumulated_solution_paths"]
            return self.nlg_model.generate_text(nlu_for_final_resp, self.state_manager.get_current_state(), "GENERATE_RESPONSE")

        # Снимок, на котором генерируем/обрабатываем идеи на ЭТОМ УРОВНЕ "думания"
        parent_snapshot_for_this_level_ideas = thinking_goal["current_snapshot_id"]
        if not self.state_manager.restore_snapshot(parent_snapshot_for_this_level_ideas):
            error_msg = f"Snap err @ iter {current_iteration}"; print(error_msg)
            thinking_goal["status"] = "thinking_error_snapshot_restore"; self.state_manager.update_goal_status(goal_id, thinking_goal["status"], "goal_id")
            return error_msg

        print(f"Thinking iter {current_iteration} on parent_snap {parent_snapshot_for_this_level_ideas} for goal {goal_id}")

        # Шаг 1: Если нет текущих идей для исследования или все исследованы, генерируем новые
        if thinking_goal.get("current_idea_index", 0) >= len(thinking_goal.get("ideas_to_explore", [])):
            thinking_goal.update({"ideas_to_explore": [], "current_idea_index": 0})
            # NLU для генерации идей берется из current_iteration_nlu
            thinking_goal["parent_nlu_for_ideas"] = dict(thinking_goal["current_iteration_nlu"])

            nlg_out = self.nlg_model.generate_text(
                thinking_goal["current_iteration_nlu"],
                self.state_manager.get_current_state(), # Состояние из parent_snapshot_for_this_level_ideas
                task_type="THINK_AND_PLAN"
            )
            print(f"NLG generated new ideas/q: {nlg_out}")
            if THINKING_OUTPUT_SEPARATOR in nlg_out:
                q, ideas_str = nlg_out.split(THINKING_OUTPUT_SEPARATOR, 1)
                thinking_goal["pending_user_question"] = q.strip()
                if ideas_str: thinking_goal["ideas_to_explore"] = [i.strip() for i in ideas_str.split(';') if i.strip()]
            else:
                thinking_goal["pending_user_question"] = nlg_out.strip()

        # Шаг 2: Обрабатываем следующую доступную идею
        ideas_this_level = thinking_goal.get("ideas_to_explore", [])
        current_idea_idx_this_level = thinking_goal.get("current_idea_index", 0)

        if current_idea_idx_this_level < len(ideas_this_level):
            idea_to_explore = ideas_this_level[current_idea_idx_this_level]

            if idea_to_explore in thinking_goal["history_ideas_tried"]:
                print(f"Skipping already explored idea in this session: '{idea_to_explore}'")
                thinking_goal["current_idea_index"] += 1
                self.state_manager.restore_snapshot(parent_snapshot_for_this_level_ideas)
                thinking_goal["current_snapshot_id"] = parent_snapshot_for_this_level_ideas
                return self._thinking_loop_iteration(thinking_goal)

            # Создаем снимок для ветки этой идеи, отталкиваясь от parent_snapshot_for_this_level_ideas
            idea_branch_snap_id = f"{parent_snapshot_for_this_level_ideas}_idea{current_idea_idx_this_level}"
            # StateManager УЖЕ в состоянии parent_snapshot_for_this_level_ideas
            self.state_manager.create_snapshot(idea_branch_snap_id)

            if not self.state_manager.restore_snapshot(idea_branch_snap_id): # Переключаемся на ветку идеи
                 thinking_goal["history_ideas_tried"][idea_to_explore] = {"status": "error_snapshot_restore_branch", "snapshot_id": idea_branch_snap_id}
                 thinking_goal["current_idea_index"] += 1 # Пробуем следующую
                 self.state_manager.restore_snapshot(parent_snapshot_for_this_level_ideas)
                 thinking_goal["current_snapshot_id"] = parent_snapshot_for_this_level_ideas
                 return self._thinking_loop_iteration(thinking_goal)

            print(f"Exploring idea ({current_idea_idx_this_level + 1}/{len(ideas_this_level)}): '{idea_to_explore}' on snap '{idea_branch_snap_id}'")
            nlu_for_idea = self.nlu_controller.predict_nlu(idea_to_explore)

            # Обновляем current_snapshot_id в thinking_goal ПЕРЕД вызовом _handle_nlu_result,
            # чтобы _handle_nlu_result знал, какой снимок обновлять.
            thinking_goal["current_snapshot_id"] = idea_branch_snap_id
            action_detail, next_nlg_task = self._handle_nlu_result(nlu_for_idea, thinking_goal)
            print(f"Result of processing idea '{idea_to_explore}': {action_detail}")
            thinking_goal["history_ideas_tried"][idea_to_explore] = {"status": "explored", "snapshot_id": idea_branch_snap_id, "nlu_result": nlu_for_idea}

            if self._is_solution_found(thinking_goal, self.state_manager.get_current_state()):
                solution_state = dict(self.state_manager.get_current_state())
                thinking_goal["accumulated_solution_paths"].append({
                    "idea_path": [k for k,v in thinking_goal["history_ideas_tried"].items() if v.get("status") != "skipped"], # Упрощенный путь
                    "final_state": solution_state, "triggering_idea": idea_to_explore
                })
                thinking_goal["history_ideas_tried"][idea_to_explore]["status"] = "solution_branch_found"
                print(f"Solution branch found via '{idea_to_explore}'. Snapshot: {idea_branch_snap_id}.")
                # После нахождения решения в ветке, восстанавливаем родительский снимок и пробуем следующую идею
                self.state_manager.restore_snapshot(parent_snapshot_for_this_level_ideas)
                thinking_goal["current_snapshot_id"] = parent_snapshot_for_this_level_ideas
                thinking_goal["current_iteration_nlu"] = thinking_goal["parent_nlu_for_ideas"]
                thinking_goal["current_idea_index"] += 1 # Переходим к следующей идее из списка
                return self._thinking_loop_iteration(thinking_goal)

            if next_nlg_task == "REQUEST_CLARIFICATION":
                thinking_goal.update({"status": "thinking_paused_needs_user_clarification",
                                      "pending_user_question": self.nlg_model.generate_text(nlu_for_idea, self.state_manager.get_current_state(), "REQUEST_CLARIFICATION"),
                                      "current_snapshot_id": idea_branch_snap_id}) # Остаемся на этом снимке для уточнения
                thinking_goal["history_ideas_tried"][idea_to_explore]["status"] = "clarification_needed_in_branch"
                self.state_manager.update_goal_status(goal_id, thinking_goal["status"], "goal_id")
                return thinking_goal["pending_user_question"]

            # Идея обработана, не решение, не уточнение. Углубляемся (рекурсивный вызов).
            thinking_goal.update({
                "current_iteration_nlu": nlu_for_idea,
                "current_snapshot_id": idea_branch_snap_id,
                "ideas_to_explore": [], "current_idea_index": 0,
                "parent_nlu_for_ideas": nlu_for_idea # NLU для следующего уровня генерации идей
            })
            thinking_goal["history_ideas_tried"][idea_to_explore]["status"] = "deepening_from_this_idea"

            # StateManager УЖЕ в состоянии idea_branch_snap_id
            deeper_result = self._thinking_loop_iteration(thinking_goal) # Рекурсия

            # После возврата из углубления (независимо от результата там, кроме глобального решения/паузы/лимита)
            # мы должны вернуться к parent_snapshot_for_this_level_ideas и попробовать следующую идею из списка.
            if thinking_goal["status"] in ["thinking_solution_found", "thinking_paused_needs_user_clarification", "thinking_limit_reached"]:
                return deeper_result

            print(f"Branch for idea '{idea_to_explore}' ended. Returning to parent snap {parent_snapshot_for_this_level_ideas}")
            self.state_manager.restore_snapshot(parent_snapshot_for_this_level_ideas)
            thinking_goal["current_snapshot_id"] = parent_snapshot_for_this_level_ideas
            thinking_goal["current_iteration_nlu"] = thinking_goal["parent_nlu_for_ideas"]
            thinking_goal["ideas_to_explore"] = ideas_this_level
            # current_idea_index уже был инкрементирован, так что следующий вызов _thinking_loop_iteration возьмет следующую идею.
            return self._thinking_loop_iteration(thinking_goal)

        # Все идеи из текущего набора (ideas_this_level) были обработаны.
        if thinking_goal["accumulated_solution_paths"]:
            thinking_goal["status"] = "thinking_completed_with_solutions"
            self.state_manager.update_goal_status(goal_id, thinking_goal["status"], "goal_id")
            if thinking_goal.get("base_snapshot_id"): self.state_manager.restore_snapshot(thinking_goal["base_snapshot_id"])
            summary = f"Found {len(thinking_goal['accumulated_solution_paths'])} solution(s). E.g., via idea '{thinking_goal['accumulated_solution_paths'][0]['triggering_idea']}' leading to state: " + json.dumps(thinking_goal['accumulated_solution_paths'][0].get("final_state",{}).get("entities",{}).get("business_plan_draft",{}))
            return self.nlg_model.generate_text({"intent": "MULTIPLE_SOLUTIONS_FOUND", "slots": [{"type":"summary", "value": summary}]}, self.state_manager.get_current_state(), "GENERATE_RESPONSE")

        final_q = thinking_goal.get("pending_user_question", "I've explored these options. How should we proceed?")
        thinking_goal["status"] = "thinking_stuck_needs_user_or_new_strategy"
        self.state_manager.update_goal_status(goal_id, thinking_goal["status"], "goal_id")
        self.state_manager.restore_snapshot(parent_snapshot_for_this_level_ideas)
        thinking_goal["current_snapshot_id"] = parent_snapshot_for_this_level_ideas
        return final_q

    def process_user_input(self, user_text: str) -> str:
        # ... (логика process_user_input остается в основном прежней)
        print(f"\nLogicalController processing user input: '{user_text}'")
        active_goals = self.state_manager.get_active_goals()
        pending_goal = next((g for g in active_goals if g.get("status") and \
            ("pending_user_clarification" in g.get("status") or \
             g["status"] in ["thinking_stuck_needs_user_or_new_strategy", "thinking_ended_nlg_no_ideas"])), None)

        nlu_result_to_process = None # Этот NLU будет либо от нового ввода, либо от ответа на уточнение

        if pending_goal: # Пользователь отвечает на вопрос системы
            goal_id = pending_goal['goal_id']
            print(f"Input is response for goal: {goal_id} (status: {pending_goal['status']})")

            snap_to_restore = pending_goal.get("current_snapshot_id") # Снимок, на котором остановились для вопроса
            if snap_to_restore and self.state_manager.restore_snapshot(snap_to_restore):
                print(f"Restored snap {snap_to_restore} for goal {goal_id}.")

            nlu_user_resp = self.nlu_controller.predict_nlu(user_text)
            print(f"NLU of user's response: {nlu_user_resp}")

            if pending_goal.get("goal_type") == "SYSTEM_THINKING_PROCESS":
                pending_goal["current_iteration_nlu"] = nlu_user_resp # Обновляем NLU контекст для "думания"
                # Применяем NLU ответа пользователя к текущему состоянию снимка "думания"
                self._handle_nlu_result(nlu_user_resp, pending_goal) # Это обновит снимок pending_goal["current_snapshot_id"]

                pending_goal.update({"ideas_to_explore": [], "current_idea_index": 0,
                                     "status": f"thinking_iteration_{pending_goal['iteration']}"}) # Сброс идей, т.к. контекст изменился
                self.state_manager.update_goal_status(goal_id, pending_goal["status"], "goal_id")
                return self._thinking_loop_iteration(pending_goal) # Продолжаем "думать"

            else: # Это было простое уточнение CLARIFY_FOR_INTENT
                original_nlu = pending_goal.get("original_nlu_result", {})
                updated_slots = list(original_nlu.get("slots", []))
                # TODO: Более умное слияние слотов из nlu_user_resp в updated_slots
                # Например, если missing_slot_type был известен, ищем его в nlu_user_resp
                missing_slot_type = pending_goal.get("missing_slot_type")
                clarified = False
                for r_slot in nlu_user_resp.get("slots",[]):
                    if missing_slot_type and r_slot.get("type") == missing_slot_type:
                        # Нашли нужный слот для уточнения
                        # Нужно обновить его в updated_slots или добавить
                        slot_replaced = False
                        for i, o_slot in enumerate(updated_slots):
                            if o_slot.get("type") == missing_slot_type:
                                updated_slots[i] = r_slot; slot_replaced = True; break
                        if not slot_replaced: updated_slots.append(r_slot)
                        clarified = True; break
                    elif not missing_slot_type and len(nlu_user_resp.get("slots",[])) == 1: # Если не ждали конкретный, но NLU что-то одно извлек
                         updated_slots.append(r_slot); clarified = True; break
                if not clarified and nlu_user_resp.get("intent") not in ["OUT_OF_SCOPE", "UNKNOWN_INTENT"]: # Если ничего не извлекли, но текст есть
                    if missing_slot_type: updated_slots.append({"type": missing_slot_type, "value": user_text}) # Грубо берем весь текст

                nlu_result_to_process = {"intent": original_nlu.get("intent"), "slots": updated_slots}
                self.state_manager.remove_goal(goal_id, criteria="goal_id")
                print(f"Re-processing with clarified NLU: {nlu_result_to_process}")
                # Этот nlu_result_to_process пойдет на обычную обработку ниже
        else:
            nlu_result_to_process = self.nlu_controller.predict_nlu(user_text)

        # --- Основной поток обработки NLU результата (нового или уточненного) ---
        print(f"NLU Result to process: {nlu_result_to_process}")
        action_detail, nlg_task_override = self._handle_nlu_result(nlu_result_to_process, None)
        current_task_for_nlg = nlg_task_override if nlg_task_override else "GENERATE_RESPONSE"

        should_start_new_thinking = self._is_complex_query_requiring_thought(nlu_result_to_process, self.state_manager.get_current_state()) and \
                                   current_task_for_nlg == "THINK_AND_PLAN" and \
                                   not pending_goal # Не начинаем новое думание, если только что обработали ответ на уточнение

        if should_start_new_thinking:
            print(f"Initiating NEW thinking: {nlu_result_to_process.get('intent')}")
            base_snap_id = f"base_think_{random.randint(1000,9999)}"
            self.state_manager.create_snapshot(base_snap_id)
            thinking_goal = self._create_thinking_goal(nlu_result_to_process, base_snap_id)
            return self._thinking_loop_iteration(thinking_goal)

        elif current_task_for_nlg == "REQUEST_CLARIFICATION":
            final_response = self.nlg_model.generate_text(nlu_result_to_process, self.state_manager.get_current_state(), "REQUEST_CLARIFICATION")
            missing_slot = nlu_result_to_process.get("meta_missing_slot", "details")
            # Проверяем, нет ли уже АКТИВНОЙ цели на уточнение для этого же NLU и слота
            existing_goal = next((g for g in self.state_manager.get_active_goals() if \
                                   g.get("goal_type") == "CLARIFY_FOR_INTENT" and \
                                   g.get("original_nlu_result",{}).get("intent") == nlu_result_to_process.get("intent") and \
                                   g.get("missing_slot_type") == missing_slot and \
                                   g.get("status") == "pending_user_clarification"), None)
            if not existing_goal:
                gid = f"clarify_{nlu_result_to_process.get('intent','unk')}_{random.randint(1000,9999)}"
                self.state_manager.add_goal({"goal_id": gid, "goal_type": "CLARIFY_FOR_INTENT",
                                             "original_nlu_result": dict(nlu_result_to_process),
                                             "missing_slot_type": missing_slot,
                                             "status": "pending_user_clarification",
                                             "pending_user_question": final_response })
            else: # Если такая цель уже есть, просто обновляем вопрос (хотя он должен быть тем же)
                existing_goal.update({"pending_user_question": final_response, "status": "pending_user_clarification"})
        else:
            final_response = self.nlg_model.generate_text(nlu_result_to_process, self.state_manager.get_current_state(), "GENERATE_RESPONSE")

        print(f"LogicalController final response: {final_response}")
        return final_response

if __name__ == '__main__':
    print("Initializing components for LogicalController test...")
    intent_model_path_check = "models/intent_classifier_bert_en"
    slot_model_path_check = "models/slot_filler_bert_en"
    if not (os.path.exists(intent_model_path_check) and os.path.exists(slot_model_path_check)):
        print(f"ERROR: NLU models not found. Run training.")
    else:
        # Инициализация InferenceEngine и KnowledgeBaseManager (пока kb_manager - заглушка)
        # kb_mng = KnowledgeBaseManager() # Если Neo4j настроен
        kb_mng = None
        inf_engine = SimpleInferenceEngine() # Загрузит свои правила по умолчанию

        state_mng = StateManager(); nlu_ctrl = NLUController(None); nlg_mdl = NLGModel()
        if not (nlu_ctrl.tokenizer and nlu_ctrl.intent_model and nlu_ctrl.slot_model and nlg_mdl.tokenizer and nlg_mdl.model):
            print("ERROR: NLU or NLG models not fully loaded.")
        else:
            logical_ctrl = LogicalController(nlu_ctrl, state_mng, nlg_mdl, inf_engine, kb_mng)
            print("\nSCENARIO: Business Plan Query with potential inference and thinking")
            state_mng.reset_state(); state_mng._snapshots = {}
            # Добавим начальную цель, чтобы правило R002 в InferenceEngine могло сработать
            # Это лучше делать через NLU, но для теста пока так:
            # logical_ctrl.process_user_input("I want to create a business plan for an online education platform.")
            # Вместо этого, зададим вопрос, который инициирует QUERY_BUSINESS_PLAN

            state_mng.add_or_update_entity("user_profile", {"experience": "10 years in marketing", "skills": "digital advertising, SEO", "interests": "eco-friendly products"})

            resp1 = logical_ctrl.process_user_input("I want to start a new business for eco-friendly handmade soaps. Can you help me think through it?")
            print(f"User: I want to start a new business for eco-friendly handmade soaps. Can you help me think through it?\nSystem: {resp1}")
            print("Current State Facts (after 1st turn, potentially with inference):", state_mng.get_facts())
            print("Active goals (after 1st turn):", json.dumps(state_mng.get_active_goals(), indent=2))

            active_goals_after_bp1 = state_mng.get_active_goals()
            if active_goals_after_bp1 and (pending_goal := next((g for g in active_goals_after_bp1 if g.get("status") and ("pending_user_clarification" in g.get("status") or "thinking_stuck" in g.get("status"))),None) ):
                user_clarification_input = ""
                current_question = pending_goal.get("pending_user_question","").lower()

                # Имитируем несколько обменов для заполнения business_plan_draft
                if "budget" in current_question: user_clarification_input = "My starting budget is $5000. I will also define marketing channels."
                elif "audience" in current_question: user_clarification_input = "The target audience is young professionals."
                elif "marketing" in current_question: user_clarification_input = "For marketing I will use social media and local markets."
                elif "partners" in current_question: user_clarification_input = "Key partners could be local eco-stores."
                elif "strategy" in current_question: user_clarification_input = "My strategy is online sales and unique packaging."
                # Добавим еще одно условие для _is_solution_found
                elif "key_partnerships" in current_question: user_clarification_input = "For key partnerships, I'll contact craft fair organizers."


                if user_clarification_input:
                    print(f"\nUser (clarification): {user_clarification_input}")
                    resp2 = logical_ctrl.process_user_input(user_clarification_input)
                    print(f"System: {resp2}")
                    print("Goals:", json.dumps(state_mng.get_active_goals(), indent=2))
                    print("Entities:", state_mng.get_all_entities())

                    # Можно добавить еще одну итерацию диалога, если система снова задаст вопрос
                    # ...

    print("\nLogicalController test finished.")
