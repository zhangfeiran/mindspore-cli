package context

import (
	stdctx "context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
)

type fakeSummaryProvider struct {
	content string
	err     error
	calls   int
	reqs    []*llm.CompletionRequest
}

func (p *fakeSummaryProvider) Name() string { return "fake-summary" }

func (p *fakeSummaryProvider) Complete(_ stdctx.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	p.calls++
	p.reqs = append(p.reqs, req)
	if p.err != nil {
		return nil, p.err
	}
	return &llm.CompletionResponse{
		Content: p.content,
		Usage: llm.Usage{
			PromptTokens:     100,
			CompletionTokens: 20,
			TotalTokens:      120,
		},
	}, nil
}

func (p *fakeSummaryProvider) CompleteStream(stdctx.Context, *llm.CompletionRequest) (llm.StreamIterator, error) {
	return nil, errors.New("stream not implemented")
}

func (p *fakeSummaryProvider) SupportsTools() bool { return false }

func (p *fakeSummaryProvider) AvailableModels() []llm.ModelInfo { return nil }

func TestCompactWithSummaryGeneratesSummaryAndPreservesTail(t *testing.T) {
	cfg := DefaultManagerConfig()
	cfg.ContextWindow = 220
	cfg.ReserveTokens = 20
	cfg.CompactMode = "summary"
	cfg.CompactSummaryMaxTokens = 64
	cfg.AutoCompactMaxTailTokens = 80
	cfg.AutoCompactMinTailTokens = 40
	cfg.AutoCompactMinMessages = 1
	mgr := NewManager(cfg)

	for i := 0; i < 6; i++ {
		if err := mgr.AddMessage(llm.NewUserMessage(strings.Repeat(string(rune('a'+i)), 160))); err != nil {
			t.Fatalf("AddMessage #%d failed: %v", i+1, err)
		}
	}
	beforeCount := len(mgr.GetNonSystemMessages())
	provider := &fakeSummaryProvider{content: "<analysis>draft only</analysis><summary>important summary</summary>"}

	result, err := mgr.CompactWithSummary(stdctx.Background(), provider, SummaryCompactOptions{MaxSummaryTokens: 64})
	if err != nil {
		t.Fatalf("CompactWithSummary failed: %v", err)
	}
	if !result.LLMCompacted || !result.AutoCompacted {
		t.Fatalf("compact result = %#v, want LLM auto compact", result)
	}
	if provider.calls != 1 {
		t.Fatalf("provider calls = %d, want 1", provider.calls)
	}
	if got := provider.reqs[0].Tools; len(got) != 0 {
		t.Fatalf("summary request tools = %d, want 0", len(got))
	}
	if got := provider.reqs[0].MaxTokens; got == nil || *got != 64 {
		t.Fatalf("summary request max tokens = %v, want 64", got)
	}
	summaryPrompt := provider.reqs[0].Messages[len(provider.reqs[0].Messages)-1].Content
	if !strings.Contains(summaryPrompt, strings.Repeat("a", 160)) {
		t.Fatalf("summary prompt missing oldest context")
	}
	if !strings.Contains(summaryPrompt, strings.Repeat("f", 160)) {
		t.Fatalf("summary prompt missing newest context")
	}

	messages := mgr.GetNonSystemMessages()
	if len(messages) >= beforeCount {
		t.Fatalf("message count after compact = %d, want less than %d", len(messages), beforeCount)
	}
	if len(messages) == 0 || messages[0].Role != "user" {
		t.Fatalf("first post-compact message = %#v, want user summary", messages)
	}
	if !strings.Contains(messages[0].Content, "Summary:\nimportant summary") {
		t.Fatalf("summary message missing formatted summary: %q", messages[0].Content)
	}
	if strings.Contains(messages[0].Content, "draft only") {
		t.Fatalf("summary message leaked analysis block: %q", messages[0].Content)
	}
	if !strings.Contains(messages[0].Content, "Recent messages are preserved verbatim") {
		t.Fatalf("summary message missing preserved-tail notice: %q", messages[0].Content)
	}
}

func TestPrepareForRequestWithSummaryFallsBackToLocalOnFailure(t *testing.T) {
	cfg := DefaultManagerConfig()
	cfg.ContextWindow = 220
	cfg.ReserveTokens = 20
	cfg.CompactMode = "summary"
	cfg.AutoCompactMaxTailTokens = 80
	cfg.AutoCompactMinTailTokens = 40
	cfg.AutoCompactMinMessages = 1
	mgr := NewManager(cfg)

	for i := 0; i < 10; i++ {
		if err := mgr.AddMessage(llm.NewUserMessage(strings.Repeat("x", 120))); err != nil {
			t.Fatalf("AddMessage #%d failed: %v", i+1, err)
		}
	}
	provider := &fakeSummaryProvider{err: errors.New("context_length_exceeded: too many tokens")}

	result, err := mgr.PrepareForRequestWithSummary(stdctx.Background(), provider, SummaryCompactOptions{LocalFallback: true}, zeroTime())
	if err != nil {
		t.Fatalf("PrepareForRequestWithSummary failed: %v", err)
	}
	if !result.LocalFallback {
		t.Fatalf("LocalFallback = false, want true")
	}
	if !result.AutoCompacted {
		t.Fatalf("AutoCompacted = false, want true")
	}
	if provider.calls == 0 {
		t.Fatal("provider was not called before fallback")
	}
}

func zeroTime() time.Time {
	return time.Time{}
}
