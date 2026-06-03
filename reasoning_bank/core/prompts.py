"""Built-in prompt templates for memory induction, customizable per domain."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Domain-adapted prompt templates
# ---------------------------------------------------------------------------

# Domain descriptor maps: used to fill the domain context in prompts.
_DOMAIN_DESCRIPTIONS = {
    "web": "web navigation",
    "coding": "fixing issues in code repositories",
    "general": "completing tasks in an interactive environment",
}


def _make_single_si(status: str = "success", domain: str = "web") -> str:
    """Build a system prompt for single-trajectory induction.

    Args:
        status: ``"success"`` or ``"fail"``.
        domain: ``"web"``, ``"coding"``, or ``"general"``.
    """
    desc = _DOMAIN_DESCRIPTIONS.get(domain, _DOMAIN_DESCRIPTIONS["general"])

    if status == "success":
        trajectory_desc = "how an agent **successfully accomplished the task**"
        trajectory_noun = "successful trajectory"
        procedure_adj = "concrete, actionable procedures"
        thinking_hint = "Analyze why the trajectory was successful and what insights can be extracted..."
        content_hint = "the insights learned to successfully accomplishing similar tasks in the future"
    else:
        trajectory_desc = "how an agent attempted to resolve the task but **failed**"
        trajectory_noun = "failed trajectory"
        procedure_adj = "concrete, actionable recovery procedures"
        thinking_hint = "Reflect on why the trajectory failed and what lessons can be learned..."
        content_hint = (
            "the insights learned to avoid such failures and successfully accomplishing similar tasks in the future"
        )

    return f"""\
You are an expert in {desc}. You will be given a user query, the corresponding
trajectory that represents **{trajectory_desc}**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items
based on the agent's {trajectory_noun}.
The goal of summarized memory items is to be helpful and generalizable for
future similar tasks.

## Important notes
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer {procedure_adj} over abstract principles.
    Do not embed specific product names, queries, or literal string contents
    from the task.

## Output Format
First, reason inside <thinking> tags. Then output ONLY the final memory items
in the Markdown format shown below.

<thinking>
{thinking_hint}
</thinking>

# Memory Item 1
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing {content_hint}>
"""


def _make_parallel_si(domain: str = "web") -> str:
    desc = _DOMAIN_DESCRIPTIONS.get(domain, _DOMAIN_DESCRIPTIONS["general"])
    return f"""\
You are an expert in {desc}. You will be given a user query and multiple
trajectories showing how an agent attempted the task.
Some trajectories may be successful, and others may have failed.

## Guidelines
Your goal is to **compare and contrast** these trajectories to identify the
most useful and generalizable strategies as memory items.
Use **self-contrast reasoning**:
  - Identify patterns and strategies that consistently led to success.
  - Identify mistakes or inefficiencies from failed trajectories and formulate
    preventative strategies.
  - Prefer strategies that generalize beyond specific pages or exact wording.

## Important notes
  - You can extract *at most 5* memory items from all trajectories combined.
  - Do not repeat similar or overlapping items.
  - Do not mention specific websites, queries, or string contents — focus on
    generalizable behaviors and reasoning patterns.
  - Make sure each memory item captures **actionable** and **transferable**
    insights.

## Output Format
First, reason inside <thinking> tags. Then output ONLY the final memory items
in the Markdown format shown below.

<thinking>
Compare the trajectories. Why did some succeed while others failed? What
patterns emerge?
</thinking>

# Memory Item 1
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-5 sentences describing the insights learned to avoid such failures
  and successfully accomplishing similar tasks in the future>
"""


# ---------------------------------------------------------------------------
# Backward-compatible aliases (original WebArena prompts)
# ---------------------------------------------------------------------------

SUCCESSFUL_SI = _make_single_si("success", "web")
FAILED_SI = _make_single_si("fail", "web")
PARALLEL_SI = _make_parallel_si("web")


def get_system_prompt(status: str, domain: str = "web") -> str:
    """Return the appropriate system prompt for single-trajectory induction."""
    return _make_single_si(status, domain)


def get_scaling_prompt(domain: str = "web") -> str:
    """Return the system prompt for multi-trajectory scaling induction."""
    return _make_parallel_si(domain)
