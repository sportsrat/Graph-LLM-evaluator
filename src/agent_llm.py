from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

try:  # when imported as package
    from .gtt.tool import GraphTraversalTool
    from .models.llm import StubLLM, TransformersLLM, build_tiny_llama
except ImportError:  # when run via script path
    from gtt.tool import GraphTraversalTool  # type: ignore
    from models.llm import StubLLM, TransformersLLM, build_tiny_llama  # type: ignore


TOOL_PATTERN = re.compile(r"TOOL_CALL[:\s]+(\{.*?\})", re.IGNORECASE | re.DOTALL)
# Pattern for Qwen models that use {"tool_call": {...}} format
QWEN_TOOL_CALL_PATTERN = re.compile(r'\{[^{}]*"tool_call"[^{}]*\{.*?\}[^{}]*\}', re.IGNORECASE | re.DOTALL)
# Pattern for markdown code blocks (Qwen-7B)
MARKDOWN_JSON_PATTERN = re.compile(r'```json\s*(\{.*?\})\s*```', re.IGNORECASE | re.DOTALL)
ACTION_CALL_PATTERN = re.compile(r"shortest_path\((?P<src>\d+),\s*(?P<dst>\d+)\)", re.IGNORECASE)
NEIGHBORS_PATTERN = re.compile(r"get_neighbors\((?P<node>\d+)\)", re.IGNORECASE)
TASK_SP_PATTERN = re.compile(r"between node (\d+)\s+and node (\d+)", re.IGNORECASE)
FINAL_PATTERN = re.compile(r"^\s*FINAL_ANSWER\s*\{", re.IGNORECASE | re.MULTILINE)


@dataclass
class LLMTrace:
    messages: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


class ReActToolAgent:
    """
    Simple ReAct-style loop:
    - prompt LLM for next action
    - parse TOOL_CALL {json}
    - execute on GTT
    - append observation
    - stop after max_steps or when LLM emits FINAL_ANSWER
    """

    def __init__(self, gtt: GraphTraversalTool, llm: Optional[Union[TransformersLLM, StubLLM]] = None, use_stub: bool = False, max_steps: int = 6):
        self.gtt = gtt
        self.llm = llm if llm is not None else (None if use_stub else build_tiny_llama())
        self.stub = StubLLM() if (use_stub or isinstance(llm, StubLLM)) else None
        self.max_steps = max_steps

    def run(self, task: str) -> Dict[str, Any]:
        trace = LLMTrace()
        task_hint = self._extract_task_hint(task)
        messages = [
            {"role": "system", "content": self._build_system_prompt("")},
            {"role": "user", "content": task},
        ]
        answer = None
        for step in range(self.max_steps):
            trace.messages.append(f"PROMPT_STEP_{step}:\n{messages}")
            # Use stub if available, otherwise use LLM
            if self.stub:
                # For stub, generate a tool call on first step if we have a hint
                # Otherwise, generate a final answer after tool calls
                if task_hint and step == 0 and not trace.tool_calls:
                    completion = f"TOOL_CALL: {json.dumps(task_hint)}"
                elif trace.tool_calls:
                    # After tool calls, generate final answer
                    last_result = trace.tool_calls[-1]["result"]
                    if "path" in last_result:
                        completion = f"FINAL_ANSWER: {json.dumps({'path': last_result.get('path'), 'length': last_result.get('length')})}"
                    elif "diameter" in last_result:
                        completion = f"FINAL_ANSWER: {json.dumps({'diameter': last_result.get('diameter')})}"
                    elif "component_count" in last_result:
                        completion = f"FINAL_ANSWER: {json.dumps({'component_count': last_result.get('component_count')})}"
                    else:
                        completion = f"FINAL_ANSWER: {json.dumps(last_result)}"
                else:
                    # Fallback: use stub's generate_plan
                    user_msg = messages[-1]["content"] if messages else task
                    completion = self.stub.generate_plan(user_msg)
            else:
                completion = self.llm.chat_messages(messages)
            trace.messages.append(f"LLM_STEP_{step}:\n{completion}")
            if FINAL_PATTERN.search(completion):
                answer = self._extract_final_answer(completion)
                if answer is not None:
                    break
            tool_args = self._extract_tool_call(completion)
            if not tool_args and task_hint:
                tool_args = task_hint
            if not tool_args:
                messages.append({"role": "assistant", "content": completion})
                messages.append({
                    "role": "user", 
                    "content": (
                        "Observation: No valid tool call detected. "
                        "Please use the exact format: TOOL_CALL: {\"action\": \"action_name\", \"args\": {...}}\n"
                        "For components task, use: TOOL_CALL: {\"action\": \"connected_components\", \"args\": {}}"
                    )
                })
                continue
            result = self._dispatch_tool(tool_args)
            trace.tool_calls.append({"call": tool_args, "result": result})
            # Append assistant response and observation to conversation
            messages.append({"role": "assistant", "content": completion})
            messages.append({"role": "user", "content": f"Observation: {json.dumps(result)}"})
        if not trace.tool_calls and task_hint:
            # force at least one tool use from task hint to keep demo meaningful
            result = self._dispatch_tool(task_hint)
            trace.tool_calls.append({"call": task_hint, "result": result})
            answer = result
        return {
            "answer": answer or (trace.tool_calls[-1]["result"] if trace.tool_calls else "no final answer"),
            "trace": {"messages": trace.messages, "tool_calls": trace.tool_calls},
        }

    def _build_system_prompt(self, task: str) -> str:
        return (
            "You are a graph reasoning assistant. Use the Graph Traversal Tool (GTT) to solve graph problems.\n\n"
            "IMPORTANT: You must respond in the exact JSON format specified below.\n\n"
            "To call a tool, use this exact format:\n"
            "TOOL_CALL: {\"action\": \"action_name\", \"args\": {...}}\n\n"
            "Available actions:\n"
            "- \"connected_components\": Get connected components (args: {})\n"
            "- \"graph_diameter\": Get graph diameter (args: {})\n"
            "- \"shortest_path\": Find shortest path (args: {\"source\": <node_id>, \"target\": <node_id>})\n"
            "- \"get_neighbors\": Get neighbors of a node (args: {\"node\": <node_id>})\n\n"
            "After receiving tool results, provide your final answer using:\n"
            "FINAL_ANSWER: {\"key\": \"value\"}\n\n"
            "Example for components task:\n"
            "TOOL_CALL: {\"action\": \"connected_components\", \"args\": {}}\n"
            "(wait for observation)\n"
            "FINAL_ANSWER: {\"component_count\": 1}\n\n"
            "Do NOT use any other format. Use only the TOOL_CALL and FINAL_ANSWER format shown above."
        )

    def _generate(self, prompt: str) -> str:
        if self.stub:
            return self.stub.generate_plan(prompt)
        # Format as chat messages to match SFT training format
        messages = [
            {"role": "system", "content": self._build_system_prompt("")},
            {"role": "user", "content": prompt},
        ]
        return self.llm.chat_messages(messages)

    def _extract_tool_call(self, completion: str) -> Optional[Dict[str, Any]]:
        """
        Extract tool call from LLM completion.
        Expected format: TOOL_CALL: {"action": "action_name", "args": {...}}
        """
        # Look for TOOL_CALL: {...} format (the standard format we specify in prompt)
        match = TOOL_PATTERN.search(completion)
        if match:
            try:
                parsed = json.loads(match.group(1))
                # Must have "action" key to be valid
                if "action" in parsed and isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to find JSON object with "action" key anywhere in completion
        # This handles cases where model doesn't use TOOL_CALL: prefix but has correct structure
        try:
            # Look for {"action": ...} pattern
            import re as re_module
            action_match = re_module.search(r'\{[^{}]*"action"[^{}]*:[^{}]*"([^"]+)"[^{}]*"args"[^{}]*:\s*(\{[^}]*\})', completion, re_module.IGNORECASE)
            if action_match:
                action_name = action_match.group(1)
                args_str = action_match.group(2)
                try:
                    args = json.loads(args_str) if args_str else {}
                    return {"action": action_name, "args": args}
                except:
                    # If args parsing fails, return empty args
                    return {"action": action_name, "args": {}}
        except:
            pass

        # Final fallback: parse simple Action: shortest_path(0,14) patterns
        act_match = ACTION_CALL_PATTERN.search(completion)
        if act_match:
            return {
                "action": "shortest_path",
                "args": {"source": int(act_match.group("src")), "target": int(act_match.group("dst"))},
            }
        neigh = NEIGHBORS_PATTERN.search(completion)
        if neigh:
            return {"action": "get_neighbors", "args": {"node": int(neigh.group("node"))}}
        return None

    def _extract_task_hint(self, task: str) -> Optional[Dict[str, Any]]:
        sp = TASK_SP_PATTERN.search(task)
        if sp:
            return {"action": "shortest_path", "args": {"source": int(sp.group(1)), "target": int(sp.group(2))}}
        return None

    def _extract_final_answer(self, completion: str) -> Optional[Any]:
        if "FINAL_ANSWER" not in completion:
            return None
        try:
            fragment = completion.split("FINAL_ANSWER", 1)[1]
            start = fragment.find("{")
            end = fragment.rfind("}")
            if start != -1 and end != -1:
                return json.loads(fragment[start : end + 1])
        except Exception:
            return completion
        return completion

    def _dispatch_tool(self, call: Dict[str, Any]) -> Dict[str, Any]:
        action = call.get("action")
        args = call.get("args", {})
        if action == "shortest_path":
            return self.gtt.shortest_path(args.get("source"), args.get("target"))
        if action == "graph_diameter":
            return self.gtt.graph_diameter()
        if action == "connected_components":
            return self.gtt.connected_components()
        if action == "get_neighbors":
            return self.gtt.get_neighbors(args.get("node"))
        return {"error": f"unknown action {action}"}

