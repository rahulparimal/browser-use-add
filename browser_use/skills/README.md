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

## Confidential data, local model switching, vision modes

```bash
bu-agent vault set example.com password        # prompts; stored Fernet-encrypted in SQLite (0600 files)
bu-agent vault list                            # names only, never values
bu-agent run "log in to example.com" --use-vault --local-model ollama:llama3.1
bu-agent run "read the front page" --vision auto   # screenshot only when the agent asks (on | off | auto)
bu-agent calibrate --model openai:gpt-6-luna   # measure real image token cost; estimates then use it
```

- **Vault:** the cloud LLM sees `<secret>password</secret>` placeholders; real values are substituted locally, only
  on matching domains. Protects values at rest, not against malware running as your user. Set `BU_VAULT_KEY` to
  supply your own key instead of the `vault.key` file.
- **`--local-model`:** a router sends a step to the local model (screenshots stripped) when the page host matches a
  vault domain or the page text contains an email, Luhn-valid card number or SSN-shaped number; otherwise to the cloud
  model. It fails closed: if the local model errors, the step is NOT sent to the cloud unless `--local-fallback-cloud`.
  Detection is heuristic; private data the patterns miss (names, balances) on a non-vault domain still reaches the cloud.
  Small local models are weaker at the structured output the agent needs; expect more failed steps.
