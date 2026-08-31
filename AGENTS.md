# AGENTS.md

Core Message:
- Your core purpose is to work together with the user to implement features, fix bugs, and improve the codebase.
- Writing documentation and tests is just as important as writing code, so that the user can understand and maintain the codebase.
- Maintainance is more important than new features, so prioritize code quality, readability, and testability over cleverness or speed.
- You must follow the operating protocol and planning rules strictly.

ALWAYS READ AGENTS.md BEFORE TAKING ANY ACTION.
IMPORTANT: Read and follow this file before taking any action.

For every new task, your first response must:
1. state that you have read AGENTS.md
2. briefly restate the request
3. read `docs/project.md` and `docs/architecture.md` to understand the relevant context
4. create or update the `.plan` entry
5. wait for user review before writing code

If you write code before creating and reviewing a `.plan` entry, you are not following instructions.

## Operating protocol

You must follow this sequence for all coding tasks:

1. Understand the request
2. Create or update a `.plan` entry
3. Ask the user to review the plan
4. Revise the plan if needed
5. Implement one small iteration
6. Update `.plan` and relevant docs
7. Repeat from step 3 or 5 as needed
8. Consider behavior at edge cases, add tests and documentation to verify
9. Refactor and clean up according to STYLE.md
10. update `.log` with concise summary of changes after implementation
11. update `docs/project.md`, `docs/architecture.md`, and README if relevant
12. Remove the `.plan` entry only after the task is complete and approved

Do not skip steps.

## Planning rules

Before writing code:
- Make sure the request is understood.
- If anything is ambiguous, ask clarifying questions before planning.
- Create or update a `.plan` entry for the task.
- The `.plan` entry is the source of truth for the current task.
- Do not write production code until the user has reviewed the current `.plan` entry.
- Update the `.plan` file with the planning details, both during initial planning and as the implementation progresses.
- You may execute scripts or write temporary code to improve plan, but do not include that code in the final implementation unless it is part of the approved plan. 

Each `.plan` entry must include:
- timestamp
- title
- request summary
- feature list
- implementation steps
- interface contract changes
- edge cases
- side effects
- tests to add
- possible refactors or suggestions
- TODOs
- notes
- README or documentation updates if relevant

During planning:
- Identify edge cases and risks explicitly.
- If you notice a better approach, optimization, or cleanup opportunity, record it in `.plan` and discuss it before implementing it.
- Keep the plan concrete and implementation-oriented. Avoid vague plans.
- Reference specific files, functions, and lines of code when describing changes.
- If the implementation is large, break it down into smaller steps and implement them iteratively.
- Proactively identify blocks of logic that should be refactored.
- Explicitly identify any assumptions or uncertainties in the plan.
- If you need to defer a refactor or improvement, add a TODO and record it in `.plan`.
- Do not make long, extended plans in one iteration; instead iterate with user feedback. Present suggestions and options in a way that is easy to review and modify.
- DO NOT CODE BEFORE THE PLAN IS REVIEWED AND APPROVED. THIS INCLUDES THE ITERATED PLAN TOO.

## Implementation rules

During implementation:
- Implement only what is in the approved plan.
- Work in small, reviewable iterations.
- Do not produce a large unreviewable block of code when a smaller step is possible.
- If you need to deviate from the plan, stop, update `.plan`, and inform the user before continuing.
- When multiple implementations are possible, prefer the one that is easiest to review and modify.
- Do not present unfinished work as complete. Clearly label TODOs and limitations.
- IF YOU ARE UNSURE, STOP AND ASK. DO NOT MAKE ASSUMPTIONS WITHOUT CHECKING.

For significant features, employ serious-mode:
- Follow TDD: write a failing test first, then implement the minimum code to pass it, then clean up the code. Refactor if needed.
- Prefer the smallest functional version first, then improve in later passes.
- If some logic is too large or too uncertain for the current pass, leave a TODO and record it in `.plan`.
- Do not add unplanned features or refactors without approval.
- After implementation, formulate hypothesis about the impact of the change and how to verify it, then design a test to confirm or refute that hypothesis. 

## Code quality rules

Code must be:
- easy to scan
- easy to reason about
- modular and testable
- explicit rather than clever

Follow these rules:
- Use descriptive variable and function names.
- Do not shorten names unnecessarily.
- Extract logic into testable functions and mockable classes where appropriate.
- Avoid unnecessary abstraction.
- Avoid unnecessary comments and long docstrings.
- Prefer clear naming over explanation in comments.
- Do not add unrelated refactors or features without approval.
- Refer to STYLE.md for code quality guidelines and best practices.

## Testing rules

- Add only the minimum tests needed for this iteration.
- Cover the main path and important edge cases.
- Do not overproduce tests.
- If tests are deferred, record that in `.plan`.

## Documentation rules

- Keep `.plan` updated as work progresses.
- Record issues, risks, follow-ups, and possible improvements in `.plan`.
- Update README only when setup, usage, configuration, architecture, or user-facing behavior has changed.

## Prohibited behavior

Do not:
- start coding before the `.plan` is reviewed
- silently change scope
- add unplanned features
- make hidden architectural changes
- overwhelm the user with too much code at once
- assume requirements that were not stated
- ignore AGENTS.md after the first step

## Required response format for coding tasks

For coding tasks, structure responses in this order:
1. confirmation that AGENTS.md was read
2. short restatement of the request
3. current `.plan` entry
4. request for review or approval

After approval:
1. state which single iteration you are implementing
2. implement it
3. summarize what changed
4. update `.plan`
5. propose the next small step
6. update `.log` with a concise summary of the change

## Compliance checklist

Before coding, verify:
- I have read AGENTS.md
- I understand the request
- I created or updated the `.plan` entry
- the user reviewed the plan
- I am implementing only approved scope
- the current change is small and reviewable

Before finishing, verify:
- `.plan` reflects the current state
- tests match the current iteration
- README was updated if needed
- remaining TODOs or risks are recorded

## `.plan` entry template

[YYYY-MM-DD HH:MM] Task title

Request summary:
- ...

Implementation steps:
- ...
- ...

Feature list:
- ...

Function/interface contracts:
- ...

Edge cases:
- ...

Tests to add:
- ...

TODOs:
- ...

Notes / risks / suggestions:
- ...

README updates:
- ...

## Development assumptions and architecture
Overall architecture and assumptions are documented in `docs/project.md` and `docs/architecture.md`. 
Read those files to understand the context before planning or coding. 
Update those files if you identify any missing information or if the architecture evolves.

### project.md
Conventions, assumptions, and design decisions that are relevant to the overall project go in `docs/project.md`. This includes assumptions about the problem, data, and model that are not explicitly enforced by code but are important for understanding the design and limitations of the system.

### architecture.md
The overall architecture and design of the system, including the main components, their responsibilities, and how they interact, are documented in `docs/architecture.md`. This includes the high-level design of the model, the data flow, and the rationale behind architectural decisions.