from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .gtt.tool import GraphTraversalTool


@dataclass
class AgentStep:
    thought: str
    tool_call: Optional[Dict[str, Any]] = None
    observation: Optional[Dict[str, Any]] = None


@dataclass
class AgentResult:
    answer: Dict[str, Any]
    steps: List[AgentStep] = field(default_factory=list)
    call_stats: Optional[Dict[str, Any]] = None


class ToolUseAgent:
    """Lightweight agent that plans a couple of tool calls then returns an answer."""

    def __init__(self, gtt: GraphTraversalTool):
        self.gtt = gtt

    def shortest_path(self, source: int, target: int) -> AgentResult:
        steps: List[AgentStep] = []
        steps.append(AgentStep(thought=f"Check connectivity between {source} and {target}."))
        connectivity = self.gtt.check_connectivity(source, target)
        steps[-1].tool_call = {"action": "check_connectivity", "args": {"source": source, "target": target}}
        steps[-1].observation = connectivity
        if connectivity.get("error") or not connectivity.get("connected", True):
            return AgentResult(answer={"error": "nodes not connected or missing", "details": connectivity}, steps=steps)

        steps.append(AgentStep(thought="Run shortest_path using GTT."))
        sp = self.gtt.shortest_path(source, target)
        steps[-1].tool_call = {"action": "shortest_path", "args": {"source": source, "target": target}}
        steps[-1].observation = sp
        return AgentResult(answer=sp, steps=steps, call_stats=self.gtt.stats())

    def graph_diameter(self) -> AgentResult:
        steps: List[AgentStep] = []
        steps.append(AgentStep(thought="Compute diameter using GTT."))
        diameter = self.gtt.graph_diameter()
        steps[-1].tool_call = {"action": "graph_diameter", "args": {}}
        steps[-1].observation = diameter
        return AgentResult(answer=diameter, steps=steps, call_stats=self.gtt.stats())

    def connected_components(self) -> AgentResult:
        steps: List[AgentStep] = []
        steps.append(AgentStep(thought="List connected components."))
        comps = self.gtt.connected_components()
        steps[-1].tool_call = {"action": "connected_components", "args": {}}
        steps[-1].observation = comps
        return AgentResult(answer=comps, steps=steps, call_stats=self.gtt.stats())

