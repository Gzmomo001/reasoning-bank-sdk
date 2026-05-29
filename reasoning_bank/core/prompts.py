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


def _make_successful_si(domain: str = "web") -> str:
    desc = _DOMAIN_DESCRIPTIONS.get(domain, _DOMAIN_DESCRIPTIONS["general"])
    return f"""\
You are an expert in {desc}. You will be given a user query, the corresponding trajectory that represents **how an agent successfully accomplished the task**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first think why the trajectory is successful, and then summarize the insights.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable procedures over abstract principles. Do not embed specific product names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing similar tasks in the future>
```
"""


def _make_failed_si(domain: str = "web") -> str:
    desc = _DOMAIN_DESCRIPTIONS.get(domain, _DOMAIN_DESCRIPTIONS["general"])
    return f"""\
You are an expert in {desc}. You will be given a user query, the corresponding trajectory that represents **how an agent attempted to resolve the task but failed**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable recovery procedures over abstract principles. Do not embed specific product names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to avoid such failures and successfully accomplishing similar tasks in the future>
```
"""


def _make_parallel_si(domain: str = "web") -> str:
    desc = _DOMAIN_DESCRIPTIONS.get(domain, _DOMAIN_DESCRIPTIONS["general"])
    return f"""\
You are an expert in {desc}. You will be given a user query and multiple trajectories showing how an agent attempted the task.
Some trajectories may be successful, and others may have failed.

## Guidelines
Your goal is to **compare and contrast** these trajectories to identify the most useful and generalizable strategies as memory items.
Use **self-contrast reasoning**:
  - Identify patterns and strategies that consistently led to success.
  - Identify mistakes or inefficiencies from failed trajectories and formulate preventative strategies.
  - Prefer strategies that generalize beyond specific pages or exact wording.

## Important notes
  - Think first: Why did some trajectories succeed while others failed?
  - You can extract *at most 5* memory items from all trajectories combined.
  - Do not repeat similar or overlapping items.
  - Do not mention specific websites, queries, or string contents — focus on generalizable behaviors and reasoning patterns.
  - Make sure each memory item captures **actionable** and **transferable** insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-5 sentences describing the insights learned to avoid such failures and successfully accomplishing similar tasks in the future>
```
"""


# ---------------------------------------------------------------------------
# Backward-compatible aliases (original WebArena prompts)
# ---------------------------------------------------------------------------

SUCCESSFUL_SI = _make_successful_si("web")
FAILED_SI = _make_failed_si("web")
PARALLEL_SI = _make_parallel_si("web")


def get_system_prompt(status: str, domain: str = "web") -> str:
    """Return the appropriate system prompt for single-trajectory induction."""
    if status == "success":
        return _make_successful_si(domain)
    return _make_failed_si(domain)


def get_scaling_prompt(domain: str = "web") -> str:
    """Return the system prompt for multi-trajectory scaling induction."""
    return _make_parallel_si(domain)
