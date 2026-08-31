# STYLE.md

Follow this file for all code changes.

Write code that is easy to read, easy to test, and easy to modify. Prefer explicit, standalone logic over clever or tightly coupled code.

IMPORTANT: The code should be beautiful. Apply aesthetic sense to the structure of the code.

## Functions and structure

- Write code that is easy to follow at a glance.
- Use small, standalone functions with descriptive names.
- Functions should do one thing and have clear inputs and outputs.
- Group together related logic into functions or classes.
- Prefer explicit data flow over hidden state.
- Prefer standalone functions over classes unless a class is clearly justified by state or interface needs.
- Keep orchestration separate from implementation details.
- Introduce helper functions and classes to improve modularity.
- Use polymorphism and design patterns where they will simplify complex conditional logic.
- Use data structures to group related data or logic if it will improve clarity.
- Duplicated logic should be refactored into a shared function or class method.
- If there are many if-statements, consider refactoring into smaller functions or using polymorphism.

## Dependency injection and testability

- Core logic should be injectable and testable in isolation.
- Pass dependencies explicitly instead of constructing them inside core logic.
- Separate pure logic from I/O, filesystem access, model loading, networking, and other side effects.
- Write code so that important logic can be unit tested without heavy setup or real external systems.
- If a class is needed, keep it focused and inject its dependencies.

## Naming

- Use descriptive variable, function, and class names.
- Avoid unnecessary abbreviations.
- Avoid one-letter variable names except for very local conventional cases.
- Name things so the code is understandable while skimming.
- Do not shorten names to the point of losing clarity.

## Tensor shape comments

- Tensor variables should include shape comments when the dimensions are not obvious.
- Use concise shape comments in the form `# <B, N, D>`.
- Define dimension symbols nearby when they first appear.

Example:
`# B = batch size`
`# N = number of joints`
`# D = feature dimension`

- Keep shape notation consistent across the codebase.
- Update shape comments when the tensor shape changes.
- Do not add shape comments for trivial values or dimensions that are already obvious.

## Comments

- Only add comments where it is absolutely necessary.
- Comments should be sparse and useful.
- Comments should explain intent and assumptions.
- Do not use comments to restate what the code literally does.
- Prefer good names and functions over explanatory comments.
- Use comments when future readers would otherwise miss an important constraint or reason.

## Docstrings

- Avoid docstrings by default.
- Use docstrings mainly for public-facing methods, public APIs, or interfaces with non-obvious contracts.
- Do not add docstrings to every internal helper.
- For internal code, prefer clear naming, type hints, and a small number of useful comments.

## Style priorities

When choosing between multiple implementations, prefer the one that is:
- easier to review
- easier to test
- easier to modify
- easier to understand at a glance

Avoid overengineering, unnecessary abstraction, and large multi-purpose functions.