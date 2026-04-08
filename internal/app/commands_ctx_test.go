package app

import (
	"strings"
	"testing"

	agentctx "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestCmdCtxShowsProviderBackedSource(t *testing.T) {
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 1000,
		ReserveTokens: 100,
	})
	if err := ctxManager.AddMessage(llm.NewUserMessage("hello")); err != nil {
		t.Fatalf("AddMessage failed: %v", err)
	}
	ctxManager.SetPromptTokenUsage("openai-responses", 120)
	if err := ctxManager.AddMessage(llm.NewAssistantMessage("ok")); err != nil {
		t.Fatalf("AddMessage assistant failed: %v", err)
	}

	app := newModelCommandTestApp()
	app.ctxManager = ctxManager

	app.cmdCtx()

	ev := drainUntilEventType(t, app, model.TokenUpdate)
	reply := drainUntilEventType(t, app, model.AgentReply)

	if got, want := ev.CtxUsed, ctxManager.TokenUsage().Current; got != want {
		t.Fatalf("TokenUpdate.CtxUsed = %d, want %d", got, want)
	}
	if !strings.Contains(reply.Message, "openai-responses API prompt tokens + local delta") {
		t.Fatalf("reply message = %q, want provider source", reply.Message)
	}
	if !strings.Contains(reply.Message, "Provider prompt tokens: 120") {
		t.Fatalf("reply message = %q, want provider token count", reply.Message)
	}
}

func TestCmdCtxShowsProviderBackedTotalSnapshot(t *testing.T) {
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 1000,
		ReserveTokens: 100,
	})
	if err := ctxManager.AddMessage(llm.NewUserMessage("hello")); err != nil {
		t.Fatalf("AddMessage failed: %v", err)
	}
	if err := ctxManager.AddMessage(llm.NewAssistantMessage("ok")); err != nil {
		t.Fatalf("AddMessage assistant failed: %v", err)
	}
	ctxManager.SetTotalTokenUsage("anthropic", 1809)

	app := newModelCommandTestApp()
	app.ctxManager = ctxManager

	app.cmdCtx()

	reply := drainUntilEventType(t, app, model.AgentReply)
	if !strings.Contains(reply.Message, "anthropic API total tokens + local delta") {
		t.Fatalf("reply message = %q, want provider total source", reply.Message)
	}
	if !strings.Contains(reply.Message, "Provider total tokens: 1809") {
		t.Fatalf("reply message = %q, want provider total token count", reply.Message)
	}
}

func TestCmdCtxShowsLocalEstimateFallback(t *testing.T) {
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 1000,
		ReserveTokens: 100,
	})
	if err := ctxManager.AddMessage(llm.NewUserMessage("hello")); err != nil {
		t.Fatalf("AddMessage failed: %v", err)
	}

	app := newModelCommandTestApp()
	app.ctxManager = ctxManager

	app.cmdCtx()

	reply := drainUntilEventType(t, app, model.AgentReply)
	if !strings.Contains(reply.Message, "local heuristic estimate fallback") {
		t.Fatalf("reply message = %q, want local estimate source", reply.Message)
	}
	if !strings.Contains(reply.Message, "Method: utf8 chars / 4 plus message/tool-call overhead") {
		t.Fatalf("reply message = %q, want local estimate method", reply.Message)
	}
}

func TestCmdCtxHidesBootstrapSystemPromptUsageBeforeFirstMessage(t *testing.T) {
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow: 1000,
		ReserveTokens: 100,
	})
	ctxManager.SetSystemPrompt("system prompt")
	if got := ctxManager.TokenUsage().Current; got == 0 {
		t.Fatal("raw context usage unexpectedly zero; test requires non-zero system prompt usage")
	}

	app := newModelCommandTestApp()
	app.ctxManager = ctxManager

	app.cmdCtx()

	ev := drainUntilEventType(t, app, model.TokenUpdate)
	reply := drainUntilEventType(t, app, model.AgentReply)
	if got := ev.CtxUsed; got != 0 {
		t.Fatalf("TokenUpdate.CtxUsed = %d, want 0", got)
	}
	if !strings.Contains(reply.Message, "Current:   0") {
		t.Fatalf("reply message = %q, want hidden bootstrap usage", reply.Message)
	}
	if !strings.Contains(reply.Message, "Pure local estimate now: 0") {
		t.Fatalf("reply message = %q, want zeroed local estimate", reply.Message)
	}
}
