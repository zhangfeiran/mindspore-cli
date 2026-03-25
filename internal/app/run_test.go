package app

import (
	"context"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/vigo999/ms-cli/agent/loop"
	"github.com/vigo999/ms-cli/integrations/llm"
	"github.com/vigo999/ms-cli/tools"
	"github.com/vigo999/ms-cli/ui/model"
)

type blockingStreamProvider struct {
	started chan struct{}
}

func (p *blockingStreamProvider) Name() string {
	return "blocking"
}

func (p *blockingStreamProvider) Complete(context.Context, *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return nil, io.EOF
}

func (p *blockingStreamProvider) CompleteStream(ctx context.Context, req *llm.CompletionRequest) (llm.StreamIterator, error) {
	select {
	case <-p.started:
	default:
		close(p.started)
	}
	return &blockingStreamIterator{ctx: ctx}, nil
}

func (p *blockingStreamProvider) SupportsTools() bool {
	return true
}

func (p *blockingStreamProvider) AvailableModels() []llm.ModelInfo {
	return nil
}

type blockingStreamIterator struct {
	ctx context.Context
}

func (it *blockingStreamIterator) Next() (*llm.StreamChunk, error) {
	<-it.ctx.Done()
	return nil, it.ctx.Err()
}

func (it *blockingStreamIterator) Close() error {
	return nil
}

func TestInterruptTokenCancelsActiveTask(t *testing.T) {
	provider := &blockingStreamProvider{started: make(chan struct{})}
	engine := loop.NewEngine(loop.EngineConfig{
		MaxIterations: 1,
		MaxTokens:     4096,
	}, provider, tools.NewRegistry())

	app := &Application{
		Engine:   engine,
		EventCh:  make(chan model.Event, 32),
		llmReady: true,
	}

	done := make(chan struct{})
	go func() {
		app.runTask("hello")
		close(done)
	}()

	select {
	case <-provider.started:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for task to start")
	}

	app.processInput(interruptActiveTaskToken)

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for task cancellation")
	}

	deadline := time.After(200 * time.Millisecond)
	for {
		select {
		case ev := <-app.EventCh:
			if ev.Type == model.ToolError && strings.Contains(strings.ToLower(ev.Message), "canceled") {
				t.Fatalf("expected interrupt cancellation to stay silent, got tool error %q", ev.Message)
			}
		case <-deadline:
			return
		}
	}
}
