package loop

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"

	ctxmanager "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/tools"
)

type captureProvider struct {
	lastReq *llm.CompletionRequest
}

type summaryThenStreamProvider struct {
	completeCalls int
	streamReq     *llm.CompletionRequest
}

func (p *summaryThenStreamProvider) Name() string { return "summary-then-stream" }

func (p *summaryThenStreamProvider) Complete(context.Context, *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	p.completeCalls++
	return &llm.CompletionResponse{
		Content:      "<analysis>draft</analysis><summary>auto compact summary</summary>",
		FinishReason: llm.FinishStop,
		Usage:        llm.Usage{PromptTokens: 100, CompletionTokens: 20, TotalTokens: 120},
	}, nil
}

func (p *summaryThenStreamProvider) CompleteStream(ctx context.Context, req *llm.CompletionRequest) (llm.StreamIterator, error) {
	copied := *req
	copied.Messages = append([]llm.Message(nil), req.Messages...)
	p.streamReq = &copied
	return (&captureProvider{}).CompleteStream(ctx, req)
}

func (p *summaryThenStreamProvider) SupportsTools() bool { return true }

func (p *summaryThenStreamProvider) AvailableModels() []llm.ModelInfo { return nil }

func (p *captureProvider) Name() string {
	return "capture"
}

func (p *captureProvider) Complete(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	copied := *req
	copied.Messages = append([]llm.Message(nil), req.Messages...)
	copied.Tools = append([]llm.Tool(nil), req.Tools...)
	p.lastReq = &copied

	return &llm.CompletionResponse{
		Content:      "ok",
		FinishReason: llm.FinishStop,
	}, nil
}

func (p *captureProvider) CompleteStream(ctx context.Context, req *llm.CompletionRequest) (llm.StreamIterator, error) {
	copied := *req
	copied.Messages = append([]llm.Message(nil), req.Messages...)
	copied.Tools = append([]llm.Tool(nil), req.Tools...)
	p.lastReq = &copied

	return &captureStreamIterator{
		chunks: []llm.StreamChunk{
			{
				Content:      "ok",
				FinishReason: llm.FinishStop,
			},
		},
	}, nil
}

func (p *captureProvider) SupportsTools() bool {
	return true
}

func (p *captureProvider) AvailableModels() []llm.ModelInfo {
	return nil
}

func newEngineForContextTests(provider llm.Provider) *Engine {
	return NewEngine(EngineConfig{
		MaxIterations: 1,
		ContextWindow: 8000,
	}, provider, tools.NewRegistry())
}

func TestNewEngineDefaultsReserveTokensToTenPercentOfContextWindow(t *testing.T) {
	engine := newEngineForContextTests(&captureProvider{})

	if got, want := engine.ctxManager.TokenUsage().Reserved, 800; got != want {
		t.Fatalf("ctxManager.TokenUsage().Reserved = %d, want %d", got, want)
	}
}

func TestSetContextManagerPreservesSystemPrompt(t *testing.T) {
	engine := newEngineForContextTests(&captureProvider{})

	replacement := ctxmanager.NewManager(ctxmanager.ManagerConfig{
		ContextWindow: 8000,
		ReserveTokens: 4000,
	})
	if replacement.GetSystemPrompt() != nil {
		t.Fatal("replacement context manager should start without system prompt")
	}

	engine.SetContextManager(replacement)

	system := replacement.GetSystemPrompt()
	if system == nil {
		t.Fatal("expected system prompt to be preserved on context manager swap")
	}
	if system.Content != defaultSystemPrompt() {
		t.Fatalf("expected preserved system prompt to match default, got %q", system.Content)
	}
}

func TestSetContextManagerKeepsExistingSystemPrompt(t *testing.T) {
	engine := newEngineForContextTests(&captureProvider{})

	replacement := ctxmanager.NewManager(ctxmanager.ManagerConfig{
		ContextWindow: 8000,
		ReserveTokens: 4000,
	})
	const customPrompt = "custom system prompt"
	replacement.SetSystemPrompt(customPrompt)

	engine.SetContextManager(replacement)

	system := replacement.GetSystemPrompt()
	if system == nil {
		t.Fatal("expected replacement system prompt to remain set")
	}
	if system.Content != customPrompt {
		t.Fatalf("expected custom system prompt %q, got %q", customPrompt, system.Content)
	}
}

func TestRunUsesSystemPromptAfterContextManagerSwap(t *testing.T) {
	provider := &captureProvider{}
	engine := newEngineForContextTests(provider)

	replacement := ctxmanager.NewManager(ctxmanager.ManagerConfig{
		ContextWindow: 8000,
		ReserveTokens: 4000,
	})
	engine.SetContextManager(replacement)

	_, err := engine.Run(Task{
		ID:          "task-context-swap",
		Description: "say hello",
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if provider.lastReq == nil {
		t.Fatal("expected provider to receive completion request")
	}
	if len(provider.lastReq.Messages) < 2 {
		t.Fatalf("expected at least 2 messages (system + user), got %d", len(provider.lastReq.Messages))
	}

	first := provider.lastReq.Messages[0]
	if first.Role != "system" {
		t.Fatalf("expected first message role to be system, got %q", first.Role)
	}
	if first.Content != defaultSystemPrompt() {
		t.Fatalf("expected first message content to be default system prompt, got %q", first.Content)
	}

	second := provider.lastReq.Messages[1]
	if second.Role != "user" {
		t.Fatalf("expected second message role to be user, got %q", second.Role)
	}
	if second.Content != "say hello" {
		t.Fatalf("expected second message content to be user task, got %q", second.Content)
	}
}

func TestRunPassesModelMaxTokensToProvider(t *testing.T) {
	provider := &captureProvider{}
	engine := NewEngine(EngineConfig{
		MaxIterations: 1,
		ContextWindow: 8000,
		MaxTokens:     intPtr(1234),
	}, provider, tools.NewRegistry())

	_, err := engine.Run(Task{
		ID:          "task-max-tokens",
		Description: "say hello",
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if provider.lastReq == nil {
		t.Fatal("expected provider to receive completion request")
	}
	if provider.lastReq.MaxTokens == nil {
		t.Fatal("provider.lastReq.MaxTokens = nil, want value")
	}
	if got, want := *provider.lastReq.MaxTokens, 1234; got != want {
		t.Fatalf("provider.lastReq.MaxTokens = %d, want %d", got, want)
	}
}

func TestRunAutoCompactsWithLLMSummaryBeforeRequest(t *testing.T) {
	provider := &summaryThenStreamProvider{}
	engine := NewEngine(EngineConfig{
		MaxIterations: 1,
		ContextWindow: 220,
	}, provider, tools.NewRegistry())

	cm := ctxmanager.NewManager(ctxmanager.ManagerConfig{
		ContextWindow:            220,
		ReserveTokens:            20,
		CompactMode:              "summary",
		AutoCompactMaxTailTokens: 80,
		AutoCompactMinTailTokens: 40,
		AutoCompactMinMessages:   1,
	})
	cm.SetSystemPrompt("system")
	for i := 0; i < 6; i++ {
		if err := cm.AddMessage(llm.NewUserMessage(strings.Repeat("x", 160))); err != nil {
			t.Fatalf("AddMessage #%d failed: %v", i+1, err)
		}
	}
	engine.SetContextManager(cm)

	events, err := engine.Run(Task{ID: "compact", Description: "continue"})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if provider.completeCalls != 1 {
		t.Fatalf("summary Complete calls = %d, want 1", provider.completeCalls)
	}
	if provider.streamReq == nil || len(provider.streamReq.Messages) == 0 {
		t.Fatal("main stream request missing")
	}
	summaryMessage := ""
	for _, msg := range provider.streamReq.Messages {
		if strings.Contains(msg.Content, "auto compact summary") {
			summaryMessage = msg.Content
			break
		}
	}
	if !strings.Contains(summaryMessage, "Summary:\nauto compact summary") {
		t.Fatalf("main request messages = %#v, want compact summary after system", provider.streamReq.Messages)
	}
	if strings.Contains(summaryMessage, "draft") {
		t.Fatalf("compact summary leaked analysis block: %q", summaryMessage)
	}
	if !hasLoopEvent(events, EventContextCompacted) {
		t.Fatalf("events missing %s: %#v", EventContextCompacted, events)
	}
}

func TestRunCompletesWhenStopOccursAtIterationLimit(t *testing.T) {
	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{{
			Content:      "done",
			FinishReason: llm.FinishStop,
		}},
	}

	engine := NewEngine(EngineConfig{
		MaxIterations: 1,
		ContextWindow: 4096,
	}, provider, tools.NewRegistry())

	events, err := engine.Run(Task{
		ID:          "task-complete-at-limit",
		Description: "say done",
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if len(events) == 0 {
		t.Fatal("expected events, got none")
	}

	last := events[len(events)-1]
	if got, want := last.Type, EventTaskCompleted; got != want {
		t.Fatalf("last event type = %q, want %q", got, want)
	}
}

func hasLoopEvent(events []Event, eventType string) bool {
	for _, ev := range events {
		if ev.Type == eventType {
			return true
		}
	}
	return false
}

func TestRunFailsWhenIterationBudgetExpiresBeforeCompletion(t *testing.T) {
	args, err := json.Marshal(map[string]string{"path": "README.md"})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	provider := &scriptedStreamProvider{
		responses: []*llm.CompletionResponse{
			{
				ToolCalls: []llm.ToolCall{{
					ID:   "call-read-1",
					Type: "function",
					Function: llm.ToolCallFunc{
						Name:      "read",
						Arguments: args,
					},
				}},
				FinishReason: llm.FinishToolCalls,
			},
			{
				Content:      "done",
				FinishReason: llm.FinishStop,
			},
		},
	}

	registry := tools.NewRegistry()
	registry.MustRegister(stubTool{name: "read", content: "file contents", summary: "1 line"})

	engine := NewEngine(EngineConfig{
		MaxIterations: 1,
		ContextWindow: 4096,
	}, provider, registry)

	events, err := engine.Run(Task{
		ID:          "task-exceed-limit",
		Description: "read the file",
	})
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if len(events) == 0 {
		t.Fatal("expected events, got none")
	}

	last := events[len(events)-1]
	if got, want := last.Type, EventTaskFailed; got != want {
		t.Fatalf("last event type = %q, want %q", got, want)
	}
	if got, want := last.Message, "Task exceeded maximum iterations."; got != want {
		t.Fatalf("last event message = %q, want %q", got, want)
	}
}

func intPtr(v int) *int {
	return &v
}

type captureStreamIterator struct {
	chunks []llm.StreamChunk
	index  int
}

func (it *captureStreamIterator) Next() (*llm.StreamChunk, error) {
	if it.index >= len(it.chunks) {
		return nil, io.EOF
	}
	chunk := it.chunks[it.index]
	it.index++
	return &chunk, nil
}

func (it *captureStreamIterator) Close() error {
	return nil
}
