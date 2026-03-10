# CLAUDE.md — Project Rules for AI Contributors

## Build & Test

```bash
go build ./...          # build all packages
go test ./...           # run all tests
go vet ./...            # static analysis
```

## Project Structure

```
ms-cli/
  cmd/ms-cli/main.go        # CLI entrypoint (thin wrapper)
  internal/
    app/                     # bootstrap, wire, run, commands, demo, train — private to this repo
    project/                 # roadmap and weekly helpers
    train/                   # training types
  agent/
    loop/                    # ReAct execution loop (pure: LLM → tool → LLM → done)
    orchestrator/            # Dispatch: planner decides agent vs workflow mode
    planner/                 # LLM-based execution mode decision and plan generation
    session/                 # Session state and persistence
    memory/                  # Agent memory
    context/                 # Context window management
  workflow/
    executor/                # Workflow executor (stub + demo)
    train/                   # Training workflow + demo scenario
  runtime/
    shell/                   # Shell command execution (stateful: workspace, env, timeout)
  tools/
    fs/                      # LLM-callable file tools (read, write, edit, grep, glob)
    shell/                   # LLM-callable shell tool (thin wrapper over runtime/shell)
  permission/                # Standalone permission engine
  integrations/              # LLM providers, external APIs
  ui/                        # Bubbletea TUI panels, components, training panels
  report/                    # Report generation
  trace/                     # Execution tracing and logs
  configs/                   # Shared configuration types
  demo/scenarios/            # Demo scenario data (JSON)
  docs/                      # Architecture docs, roadmap
```

## External Dependencies

- **`vigo999/mindspore-skills`** — External skill catalog. Contains skill definitions (`skill.yaml`). Skill implementations live there, not in this repo.

## Architecture Boundaries

This project has strict package ownership rules. **Do not move, merge, or restructure packages** without explicit approval from the maintainer.

### Dependency Flow

Dependencies flow **downward only**. Never create upward or circular imports.

```
cmd/ms-cli → internal/app → agent, workflow, ui
                           ↓
              agent → permission, integrations, configs
              workflow → agent/orchestrator, agent/planner, configs
                           ↓
              tools → runtime, configs
                           ↓
              runtime → configs (+ stdlib only)
              permission → configs (+ stdlib only)
              integrations → configs (+ stdlib only)
```

### Call Chain

```
User input
    ↓
internal/app/          # routes commands or tasks
    ↓
agent/orchestrator/    # calls planner, dispatches by mode
    ↓
agent/planner/         # LLM decides: agent or workflow?
    ↓                        ↓
  agent mode             workflow mode
    ↓                        ↓
agent/loop/            workflow/executor/ (stub → falls back to agent)
    ↓
tools/                 # LLM-callable tools (schema + params)
    ↓
runtime/               # actual execution (stateful: workspace, env, timeout)
```

### Package Dependency Rules

```
cmd/ms-cli/       # Calls internal/app only. Nothing else.
internal/app/     # Wiring layer. May import anything. Must NOT be imported by any other package.
agent/            # May import permission/, integrations/, configs/.
                  # Must NOT import internal/app/, ui/, cmd/, runtime/, tools/ directly.
workflow/         # May import agent/orchestrator/, agent/planner/, configs/.
                  # Must NOT import internal/app/, ui/, tools/, runtime/.
tools/            # May import runtime/, integrations/, configs/.
                  # Must NOT import agent/, ui/, workflow/.
runtime/          # May import configs/. Must NOT import any other internal package.
permission/       # May import configs/. Must NOT import any other internal package.
integrations/     # May import configs/. Must NOT import agent/, ui/, runtime/.
ui/               # May import configs/. Must NOT be imported by agent/ or runtime/.
configs/          # Shared types only. No imports from other internal packages.
trace/            # May import configs/. Must NOT import agent/, ui/.
report/           # May import configs/, trace/. Must NOT import agent/, ui/.
```

### Package Ownership & Purpose

| Package | Purpose | Status |
|---|---|---|
| `cmd/ms-cli/` | CLI entrypoint. Thin — just calls `internal/app.Run()`. | Built |
| `internal/app/` | Config init, dependency injection, runtime assembly, CLI/TUI startup, slash commands. Go-private (not importable externally). | Built |
| `agent/loop/` | Pure ReAct execution loop: LLM → tool call → LLM → done. ~295 lines. | Built |
| `agent/orchestrator/` | Calls planner, dispatches by mode. Agent mode → loop. Workflow mode → workflow executor (falls back to agent if stub). | Built |
| `agent/planner/` | LLM decides execution mode (`agent` or `workflow`). Produces `Plan` with mode, goal, optional workflow ID, and `[]Step`. | Built |
| `agent/session/` | Session state and persistence. | Built |
| `agent/memory/` | Agent memory management. | Built |
| `agent/context/` | Context window management (token counting, compaction). | Built |
| `runtime/shell/` | Shell command execution. Stateful: workspace dir, env vars, timeout, allowed/blocked commands, safety checks. | Built |
| `tools/` | LLM-callable tool definitions (schema, params, execute). Stateless wrappers. | Built |
| `tools/fs/` | File tools: read, write, edit, grep, glob. | Built |
| `tools/shell/` | Shell tool: thin LLM wrapper that delegates to `runtime/shell`. | Built |
| `permission/` | Permission engine (levels, decisions, cache). Shared by agent, tools, runtime. | Built |
| `integrations/` | External service clients (LLM providers via OpenAI-compatible API). | Built |
| `ui/` | Bubbletea TUI: chat panel, components, slash commands, training panels. | Built |
| `workflow/executor/` | Workflow executor stub + demo executor (scenario playback). | Built |
| `workflow/train/` | Training workflow definition and demo scenario. | Built |
| `internal/train/` | Training types shared across packages. | Built |
| `demo/scenarios/` | Demo scenario data (JSON). | Built |
| `configs/` | Configuration types and constants. | Built |
| `trace/` | Structured event logs (JSONL trajectories, tool call traces). Internal-facing. | Built |
| `report/` | Report generation from traces/results. | Built |

### Planned (not yet built — do NOT create empty packages)

| Package | Purpose | When to build |
|---|---|---|
| `runtime/workspace/` | Working directory and file management. | When workflow needs isolated workspaces. |
| `runtime/artifacts/` | User-visible output files (generated code, reports). | When workflow produces deliverables. |
| `workflow/executor/` (real impl) | Execute `[]planner.Step` via tools with DAG, retry. Currently a stub that falls back to agent mode. | When stub fallback is insufficient. |
| `agent/router/` | Load skill definitions from `mindspore-skills`, match user intent → skill. | When skill system is needed. |

### Core Types — Do Not Change Without Approval

These types are foundational. Changing their signatures breaks multiple packages:

- `permission.PermissionLevel` and `permission.PermissionDecision` (`permission/types.go`)
- `integrations/llm` provider interfaces and `ToolSchema`
- `agent/loop` engine interfaces and `Event` type
- `agent/planner.Plan` and `planner.Step` — consumed by orchestrator and workflow executor
- `agent/planner.ExecutionMode` (`agent` or `workflow`) — LLM-decided mode
- `agent/orchestrator.RunRequest` and `RunEvent` — orchestrator-level I/O types
- `agent/session` session and snapshot types

If you need to extend a core type, **add new fields/methods** rather than modifying existing ones.

### Interface Boundaries Between Packages

```go
// agent/planner produces this; orchestrator and workflow executor consume it
type Plan struct {
    Mode     ExecutionMode  `json:"mode"`              // "agent" or "workflow"
    Goal     string         `json:"goal"`
    Workflow string         `json:"workflow,omitempty"` // named workflow ID
    Steps    []Step         `json:"steps,omitempty"`
    Params   map[string]any `json:"params,omitempty"`
}

// orchestrator defines this; workflow/executor implements it
type WorkflowExecutor interface {
    Execute(ctx context.Context, req RunRequest, plan planner.Plan) ([]RunEvent, error)
}

// orchestrator defines this; agent/loop (via adapter) implements it
type AgentExecutor interface {
    Execute(ctx context.Context, req RunRequest) ([]RunEvent, error)
}
```

### tools/ vs runtime/ Boundary

| | `tools/` | `runtime/` |
|---|---|---|
| **What** | LLM-callable tool definitions (schema, params, execute) | Execution infrastructure (shell, workspace, artifacts) |
| **State** | Stateless — pure input/output | Stateful — workspace, env, timeout, permissions |
| **Who calls** | Agent loop (via tool registry) | Tools and future workflow steps |
| **Examples** | `tools/shell/shell.go` (LLM schema + dispatch) | `runtime/shell/runner.go` (actual command execution) |

**Rule:** If a function needs workspace path, run_id, artifact dir, permission policy, or job lifecycle → `runtime/`. If it just transforms input → output → `tools/`.

**Dependency:** `tools/ → runtime/` (tools call runtime for execution). `runtime/ ✘ tools/` (runtime never calls tools).

### trace/ vs runtime/artifacts/ Boundary

| | `trace/` | `runtime/artifacts/` (planned) |
|---|---|---|
| **What** | Structured event logs (JSONL), tool call records, session trajectories | Output files: generated code, reports, downloaded files |
| **Who consumes** | Internal — developers, agents, replay/debug systems | External — end users, downstream tools |
| **Lifecycle** | Append-only during execution, read during debug/resume | Created during execution, delivered to user after |

### What NOT to Do

- **Do NOT merge `permission/` into any other package** — it is a shared, standalone service.
- **Do NOT put skill definitions in this repo** — they belong in `vigo999/mindspore-skills`.
- **Do NOT import `ui/` from `agent/`, `runtime/`, or `tools/`** — use interfaces or event buses.
- **Do NOT import `agent/` from `tools/` or `runtime/`** — lower layers never reach up.
- **Do NOT import `runtime/` from `agent/`** — agent uses tools, tools use runtime.
- **Do NOT add direct LLM provider imports outside `integrations/`** — all LLM access goes through `integrations/llm/`.
- **Do NOT bypass permission checks** — all tool/skill execution must go through the permission layer.
- **Do NOT add new top-level packages** without discussing with the maintainer first.
- **Do NOT create packages for unbuilt features** — no empty packages. Build it when you need it.
- **Do NOT mix trace and artifact concerns** — debug/replay logs go in `trace/`, user-deliverable files go in `runtime/artifacts/`.

## Code Style

- Go standard formatting (`gofmt`/`goimports`).
- Error messages: lowercase, no punctuation, wrap with `fmt.Errorf("context: %w", err)`.
- Interfaces belong in the package that *uses* them, not the package that implements them.
- Prefer returning `error` over `panic`. Reserve `panic` for truly unrecoverable states.
- Constructor functions: `NewXxx(...)` pattern.

## Git Conventions

- Branch from `main`.
- PR titles: `feat:`, `fix:`, `refactor:`, `docs:`, `test:` prefixes.
- Keep PRs focused — one concern per PR. Do not mix refactoring with new features.
