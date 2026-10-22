from enum import Enum
from typing import Any, Dict, List

from acua.envs.biology.tools import finish, nominate_genes, select_genes, select_indices
from acua.envs.utils import function_to_tool_schema

# BrowserGym tools require gymnasium — degrade gracefully when absent.
try:
    from acua.envs.browsergym.tools import (
        click,
        fill,
        go_back,
        go_forward,
        goto,
        hover,
        keyboard_press,
        new_tab,
        report_infeasible,
        scroll,
        select_option,
        send_msg_to_user,
        tab_close,
        tab_focus,
    )
    _webarena_tools: set = {
        scroll,
        keyboard_press,
        click,
        fill,
        hover,
        tab_focus,
        new_tab,
        go_back,
        go_forward,
        goto,
        tab_close,
        select_option,
        send_msg_to_user,
        report_infeasible,
    }
except ImportError:
    _webarena_tools = set()

# OSWorld tools are also an optional dependency.
try:
    from acua.envs.osworld.tools import (
        double_click,
        done,
        fail,
        key,
        left_click,
        left_click_drag,
        middle_click,
        mouse_move,
        right_click,
        triple_click,
        type_text,
        wait,
    )
    from acua.envs.osworld.tools import scroll as os_world_scroll

    _osworld_tools: set = {
        left_click,
        right_click,
        middle_click,
        double_click,
        triple_click,
        mouse_move,
        left_click_drag,
        type_text,
        key,
        os_world_scroll,
        wait,
        done,
        fail,
    }
except ImportError:
    _osworld_tools = set()


class ActionSet(Enum):
    WEBARENA = _webarena_tools
    OSWORLD = _osworld_tools

    # Orchestrator / single-agent: submits the final gene batch to the env.
    BIOLOGY = {
        select_genes,
        select_indices,
        finish,
    }

    # Specialist agents: propose genes with reasoning; do NOT write to the env
    # directly.  The BiologyOrchestratorAgent collects nominations from all
    # specialists and calls select_genes() once per cycle.
    BIOLOGY_NOMINATE = {
        nominate_genes,
        finish,
    }


def to_tool_dict(action_set: ActionSet) -> List[Dict[str, Any]]:
    res = []
    for action_function in action_set.value:
        res.append(function_to_tool_schema(action_function))
    return res
