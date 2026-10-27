# Skills Module

The Skills module provides integration with the Browser Use API to fetch and execute skills.

## Basic Usage

```python
import asyncio
from browser_use.skills import SkillService

async def main():
    skill_ids = ['skill-id-1', 'skill-id-2']

    # Initialize service
    service = SkillService(skill_ids=skill_ids, api_key='your-api-key')

    # Get all loaded skills (auto-initializes on first call)
    skills = await service.get_all_skills()

    # Execute a skill (auto-initializes if needed)
    result = await service.execute_skill(
        skill_id='skill-id-1',
        parameters={'param1': 'value1'}
    )

    if result.success:
        print(f'Success! Result: {result.result}')

    # Cleanup
    await service.close()

asyncio.run(main())
```

## Local skills (record once, replay without an LLM)

```python
from browser_use import Agent
from browser_use.skills import LocalSkillStore

# Record: a successful run is saved under <BROWSER_USE_CONFIG_DIR>/local_skills/<name>/ (default ~/.config/browseruse) (history.json + SKILL.md)
agent = Agent(task='Log in to the demo site', llm=llm, record_skill='demo-login')
await agent.run()

# Find by intent, then replay with new values for the detected parameters (see SKILL.md / skill.parameters)
store = LocalSkillStore()
skill, score = store.find('sign in to demo')[0]
await store.run(skill.name, llm=llm, variables={'email': 'other@example.com'})
```

Replay re-resolves elements against the live DOM and only calls the LLM for fallback AI steps and the final
summary. Secrets passed via `sensitive_data` are redacted in `history.json` and `SKILL.md`.

## Token efficiency knobs

- `Agent(image_optimizer=True)` (or an `ImageOptimizerSettings`) fits screenshots into a per-provider box,
  encodes them lossy when that is smaller, memoizes by content, and with `skip_unchanged=True` omits frames
  that are perceptually identical to the previous one.
- The state message is laid out stable -> volatile with an Anthropic cache breakpoint after the append-only
  history, so each step reads the previous step's prefix from cache instead of re-billing it.

## `bu-agent` command

```bash
uv tool install .          # from a checkout; puts `bu-agent` on your PATH
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY / GOOGLE_API_KEY (auto-detected), or --model provider:model

bu-agent run "find the cheapest laptop on example.com" --save-skill laptop-price
bu-agent skills list | find "laptop" | show laptop-price | delete laptop-price
bu-agent skills run laptop-price --set email=other@example.com
```

`run` enables the image optimizer by default (`--no-optimize-images`, `--skip-unchanged`, `--no-vision` to tune).
The separate `browser-use` / `bu` command is the Browser Harness CLI and is unchanged.
