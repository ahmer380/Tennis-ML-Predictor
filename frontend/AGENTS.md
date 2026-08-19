# TennisDuel Frontend — AI Development Instructions

## 1. Scope

These instructions apply only to the `frontend/` directory.

Do not modify files outside `frontend/` unless explicitly requested.

The frontend is an Angular application using TypeScript, SCSS, and Vitest.

---

## 2. General Principles

- Keep implementations simple, focused, and maintainable.
- Implement only what the task requires. Do not introduce speculative functionality.
- Inspect existing code before creating new components, services, utilities, or abstractions.
- Reuse existing functionality where appropriate.
- Avoid unnecessary refactoring or changes unrelated to the task.
- Preserve existing behaviour unless the task explicitly requires changing it.
- Prefer readable and explicit code over clever or overly abstract solutions.

---

## 3. Angular

- Use standalone Angular components and modern Angular patterns.
- Keep components focused on presentation and user interaction.
- Put reusable business/application logic in services rather than components.
- Keep templates simple and readable; avoid complex business logic in templates.
- Use Angular's built-in functionality where practical before introducing new libraries.
- Use TypeScript's type system consistently.
- Avoid `any` unless there is a clear justification.

---

## 4. API & Backend

The backend is a separate FastAPI application outside `frontend/`.

- Treat the backend as an external API.
- Do not modify backend code during frontend tasks unless explicitly requested.
- Keep API communication within appropriate Angular services.
- Use strongly typed request and response models.
- Do not hardcode API URLs throughout the application.
- Use the project's environment/configuration mechanism for API configuration.
- Handle loading, success, empty, and error states appropriately.
- Never expose API keys, secrets, or credentials in frontend code.

---

## 5. Styling & UI

TennisDuel has a distinctive medieval tournament/fantasy visual identity.

All user-facing UI should consistently reflect this theme. The design should
feel:

- Medieval and tournament-inspired
- Prestigious and dramatic
- Competitive and immersive
- Polished rather than gimmicky

Draw inspiration from medieval tournaments, heraldry, royal courts, parchment,
stone, metal, wood, and tournament banners.

Avoid generic SaaS/dashboard aesthetics, overly modern styling, or childish
fantasy aesthetics.

For detailed visual rules, colours, typography, spacing, components, and
other design decisions, follow `DESIGN_SYSTEM.md`.

## 6. Dependencies

Do not add npm dependencies without justification.

Before adding a dependency:

1. Check whether Angular or the browser already provides the functionality.
2. Check whether an existing dependency can solve the problem.
3. Consider maintenance, security, and bundle-size implications.

Prefer established, actively maintained packages.

Do not introduce a framework, UI library, or state-management solution for a
problem that can reasonably be solved with existing project functionality.

---

## 7. State & Architecture

- Prefer local component/service state for simple functionality.
- Do not introduce global state management without a concrete requirement.
- Avoid premature abstraction.
- Keep responsibilities separated between components, services, models, and utilities.
- Do not create architectural patterns solely for hypothetical future requirements.

---

## 8. Testing & Validation

After making changes:

1. Format the code.
2. Run linting when available.
3. Run relevant tests.
4. Run the production build.
5. Fix errors introduced by the changes.

New functionality should have appropriate tests where meaningful.

Do not weaken or remove existing tests simply to make a task pass.

---

## 9. Git & File Discipline

- Keep changes limited to the current task.
- Do not modify unrelated files.
- Do not rewrite Git history.
- Do not create commits unless explicitly requested.
- Never commit secrets, credentials, or generated dependencies/build output.
- Do not modify configuration merely because it can be improved; only do so when relevant to the task.

---

## 10. AI Agent Behaviour

AI agents should act as implementation agents, not autonomous product managers.

- Follow the requirements of the current user story.
- Do not expand the scope of a task unnecessarily.
- Do not invent major product or architectural decisions.
- Before making significant changes, inspect the relevant existing implementation.
- Prefer established project conventions over introducing new ones.
- If a requirement is genuinely ambiguous and materially affects the implementation, ask for clarification.
- Do not assume future requirements need to be implemented now.

### Definition of Done

A task is complete when:

- The requested functionality is implemented.
- Existing functionality remains intact.
- The implementation follows project conventions.
- UI follows `DESIGN_SYSTEM.md` where applicable.
- Relevant tests pass.
- Linting passes when configured.
- The production build succeeds.
- No unrelated changes have been introduced.