package app

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	agentctx "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/agent/session"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestCmdCompactCompactsContextAndEmitsTokenUpdate(t *testing.T) {
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow:       100,
		ReserveTokens:       10,
		CompactionThreshold: 0.9,
		EnableSmartCompact:  false,
	})
	for i := 0; i < 3; i++ {
		if err := ctxManager.AddMessage(llm.NewUserMessage(strings.Repeat("x", 80))); err != nil {
			t.Fatalf("AddMessage #%d failed: %v", i+1, err)
		}
	}

	before := ctxManager.TokenUsage().Current
	app := newModelCommandTestApp()
	app.ctxManager = ctxManager

	app.cmdCompact()

	drainUntilEventType(t, app, model.AgentThinking)
	ev := drainUntilEventType(t, app, model.TokenUpdate)
	reply := drainUntilEventType(t, app, model.AgentReply)

	if got := ctxManager.TokenUsage().Current; got > before/2 {
		t.Fatalf("context usage after cmdCompact = %d, want <= %d", got, before/2)
	}
	if got, want := ev.CtxUsed, ctxManager.TokenUsage().Current; got != want {
		t.Fatalf("TokenUpdate.CtxUsed = %d, want %d", got, want)
	}
	if got, want := ev.CtxMax, ctxManager.TokenUsage().ContextWindow; got != want {
		t.Fatalf("TokenUpdate.CtxMax = %d, want %d", got, want)
	}
	if !strings.Contains(reply.Message, "Context compacted:") {
		t.Fatalf("reply message = %q, want compaction summary", reply.Message)
	}
}

func TestCmdCompactInDebugModePersistsPreCompactSnapshot(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow:       100,
		ReserveTokens:       10,
		CompactionThreshold: 0.9,
		EnableSmartCompact:  false,
	})
	for i := 0; i < 3; i++ {
		if err := ctxManager.AddMessage(llm.NewUserMessage(strings.Repeat("x", 80))); err != nil {
			t.Fatalf("AddMessage #%d failed: %v", i+1, err)
		}
	}

	workDir := t.TempDir()
	runtimeSession, err := session.Create(workDir, "system prompt")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	t.Cleanup(func() { _ = runtimeSession.Close() })

	app := newModelCommandTestApp()
	app.ctxManager = ctxManager
	app.session = runtimeSession
	app.llmDebugDumper = llm.NewDebugDumper(filepath.Dir(runtimeSession.Path()))
	preCompactMessages := ctxManager.GetNonSystemMessages()

	app.cmdCompact()

	drainUntilEventType(t, app, model.AgentThinking)
	drainUntilEventType(t, app, model.TokenUpdate)
	drainUntilEventType(t, app, model.AgentReply)

	matches, err := filepath.Glob(filepath.Join(filepath.Dir(runtimeSession.Path()), "snapshot.compact-pre-manual-*.json"))
	if err != nil {
		t.Fatalf("glob debug snapshots: %v", err)
	}
	if got, want := len(matches), 1; got != want {
		t.Fatalf("debug snapshot count = %d, want %d", got, want)
	}

	data, err := os.ReadFile(matches[0])
	if err != nil {
		t.Fatalf("read debug snapshot: %v", err)
	}
	var snapshot session.Snapshot
	if err := json.Unmarshal(data, &snapshot); err != nil {
		t.Fatalf("unmarshal debug snapshot: %v", err)
	}
	if got, want := len(snapshot.Messages), len(preCompactMessages); got != want {
		t.Fatalf("pre-compact message count = %d, want %d", got, want)
	}
	for i := range preCompactMessages {
		if got, want := snapshot.Messages[i].Content, preCompactMessages[i].Content; got != want {
			t.Fatalf("pre-compact snapshot message[%d] = %q, want %q", i, got, want)
		}
	}
}

func TestCmdCompactUsesLLMSummaryInSummaryMode(t *testing.T) {
	ctxManager := agentctx.NewManager(agentctx.ManagerConfig{
		ContextWindow:            220,
		ReserveTokens:            20,
		CompactionThreshold:      0.9,
		CompactMode:              "summary",
		CompactSummaryMaxTokens:  64,
		AutoCompactMaxTailTokens: 80,
		AutoCompactMinTailTokens: 40,
		AutoCompactMinMessages:   1,
	})
	for i := 0; i < 6; i++ {
		if err := ctxManager.AddMessage(llm.NewUserMessage(strings.Repeat("x", 160))); err != nil {
			t.Fatalf("AddMessage #%d failed: %v", i+1, err)
		}
	}

	app := newModelCommandTestApp()
	app.ctxManager = ctxManager
	app.provider = &singleReplyProvider{content: "<analysis>draft</analysis><summary>compact summary</summary>"}
	app.Config.Context.CompactSummaryMaxTokens = 64

	app.cmdCompact()

	drainUntilEventType(t, app, model.AgentThinking)
	drainUntilEventType(t, app, model.TokenUpdate)
	reply := drainUntilEventType(t, app, model.AgentReply)

	messages := ctxManager.GetNonSystemMessages()
	if len(messages) == 0 {
		t.Fatal("post-compact messages empty")
	}
	if !strings.Contains(messages[0].Content, "Summary:\ncompact summary") {
		t.Fatalf("summary message = %q, want LLM summary", messages[0].Content)
	}
	if strings.Contains(messages[0].Content, "draft") {
		t.Fatalf("summary message leaked analysis block: %q", messages[0].Content)
	}
	if !strings.Contains(reply.Message, "Context compacted:") {
		t.Fatalf("reply message = %q, want compaction summary", reply.Message)
	}
}
