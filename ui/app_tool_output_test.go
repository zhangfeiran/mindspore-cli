package ui

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/mindspore-code/ui/model"
)

func TestTruncateToolContentWithPolicy_HeadTailAndOmittedLineCount(t *testing.T) {
	content := strings.Join([]string{
		"line-1",
		"line-2",
		"line-3",
		"line-4",
		"line-5",
		"line-6",
	}, "\n")

	got := truncateToolContentWithPolicy(content, 2, 1, 1000)

	if !strings.Contains(got, "line-1\nline-2\nline-6") {
		t.Fatalf("expected head/tail preview, got:\n%s", got)
	}
	if !strings.Contains(got, "… +3 lines (ctrl+o to expand)") {
		t.Fatalf("expected omitted line hint, got:\n%s", got)
	}
}

func TestTruncateToolContentForTool_WriteUses3To5LinePreview(t *testing.T) {
	lines := make([]string, 9)
	for i := range lines {
		lines[i] = "line"
	}
	content := strings.Join(lines, "\n")

	got := truncateToolContentForTool("Write", content)

	visibleLines := strings.Count(got, "\n") + 1
	if visibleLines > 6 { // 5 visible + 1 omitted hint
		t.Fatalf("expected compact preview, got %d lines:\n%s", visibleLines, got)
	}
	if !strings.Contains(got, "… +4 lines (ctrl+o to expand)") {
		t.Fatalf("expected omitted hint, got:\n%s", got)
	}
}

func TestReadToolFinalization_HidesContent(t *testing.T) {
	pending := model.Message{
		Kind:     model.MsgTool,
		ToolName: "Read",
		ToolArgs: "configs/skills.yaml",
		Display:  model.DisplayCollapsed,
		Pending:  true,
	}

	resolved := finalizeToolMessage(pending, model.Event{
		Type:    model.ToolRead,
		Message: "skills:\n  repo: x",
		Summary: "6 lines",
	})

	if strings.TrimSpace(resolved.Content) != "" {
		t.Fatalf("expected read tool content hidden, got: %q", resolved.Content)
	}
	if resolved.Summary != "6 lines" {
		t.Fatalf("expected summary preserved, got: %q", resolved.Summary)
	}
}

func TestCtrlO_TogglesToolExpansion(t *testing.T) {
	app := New(make(chan model.Event), nil, "dev", ".", "", "model", 1024)
	app.bootActive = false
	app.state.Messages = []model.Message{{
		Kind:     model.MsgTool,
		ToolName: "Write",
		ToolArgs: "x.md",
		Display:  model.DisplayExpanded,
		Content: strings.Join([]string{
			"a", "b", "c", "d", "e", "f", "g",
		}, "\n"),
	}}

	collapsed := app.viewportRenderState().Messages[0].Content
	if !strings.Contains(collapsed, "ctrl+o to expand") {
		t.Fatalf("expected collapsed content with expansion hint, got:\n%s", collapsed)
	}

	next, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlO})
	updated := next.(App)
	expanded := updated.viewportRenderState().Messages[0].Content
	if strings.Contains(expanded, "ctrl+o to expand") {
		t.Fatalf("expected expanded content after ctrl+o, got:\n%s", expanded)
	}
	if !strings.Contains(expanded, "\nf\ng") {
		t.Fatalf("expected full content after ctrl+o, got:\n%s", expanded)
	}
}

func TestShellCmdOutputRendersLiveInUI(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolCallStart,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
		Message:    "go test ./ui",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdStarted,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdOutput,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
		Message:    "=== RUN   TestShell",
	})
	app = next.(App)

	if app.state.Messages[0].Pending {
		t.Fatal("expected shell message to stop being pending after streamed output")
	}
	if !app.state.Messages[0].Streaming {
		t.Fatal("expected shell message to stay streaming before command finishes")
	}
	if got, want := app.state.Messages[0].Content, "=== RUN   TestShell"; got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}
	if got, want := app.state.WaitKind, model.WaitTool; got != want {
		t.Fatalf("wait kind = %v, want %v", got, want)
	}

	view := app.View()
	if !strings.Contains(view, "running command...") {
		t.Fatalf("expected running shell status in view, got:\n%s", view)
	}
	if !strings.Contains(view, "=== RUN   TestShell") {
		t.Fatalf("expected streamed shell output in view, got:\n%s", view)
	}

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdFinished,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
		Message:    "=== RUN   TestShell\nPASS",
		Summary:    "completed",
	})
	app = next.(App)

	if app.state.Messages[0].Streaming {
		t.Fatal("expected shell message streaming flag cleared after finish")
	}
	if got, want := app.state.WaitKind, model.WaitNone; got != want {
		t.Fatalf("wait kind = %v, want %v", got, want)
	}
	if got, want := app.state.Messages[0].Summary, "completed"; got != want {
		t.Fatalf("summary = %q, want %q", got, want)
	}
	if got, want := app.state.Messages[0].Content, "=== RUN   TestShell\nPASS"; got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}
}

func TestInlineEventCmd_ShellToolCallStartPrintsCommandHeader(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.state = app.startWait(model.WaitTool)

	ev := model.Event{
		Type:       model.ToolCallStart,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
		Message:    "go test ./ui",
	}
	app.state = app.state.WithMessage(app.pendingToolMessage(ev))

	cmd := app.inlineEventCmd(ev, nil)
	if cmd != nil {
		t.Fatal("expected shell start to stay in live UI instead of printing above the inline frame")
	}
	if view := app.View(); !strings.Contains(view, "Shell($ go test ./ui)") || !strings.Contains(view, "running command...") {
		t.Fatalf("expected shell header to render in inline UI, got:\n%s", view)
	}
}

func TestInlineEventCmd_ShellOutputStaysInLiveUIUntilFinish(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolCallStart,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
		Message:    "go test ./ui",
	})
	app = next.(App)
	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdStarted,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
	})
	app = next.(App)
	ev := model.Event{
		Type:       model.CmdOutput,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
		Message:    "=== RUN   TestShell",
	}
	next, _ = app.handleEvent(ev)
	app = next.(App)

	if view := app.View(); !strings.Contains(view, "=== RUN   TestShell") {
		t.Fatalf("expected shell output to stream inside inline UI, got:\n%s", view)
	}
	if cmd := app.inlineEventCmd(ev, nil); cmd != nil {
		t.Fatal("expected shell output chunks to avoid normal-buffer print commands")
	}
}

func TestInlineShellPreviewPreservesCommandHeaderWhenOutputGrows(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)
	app.toolsExpanded = true
	app.state.WaitKind = model.WaitTool

	contentLines := []string{
		"line-01", "line-02", "line-03", "line-04", "line-05",
		"line-06", "line-07", "line-08", "line-09", "line-10",
	}
	app.state.Messages = []model.Message{{
		Kind:       model.MsgTool,
		ToolName:   "Shell",
		ToolCallID: "call-shell-1",
		ToolArgs:   "$ go test ./ui",
		Display:    model.DisplayExpanded,
		Content:    strings.Join(contentLines, "\n"),
		Streaming:  true,
	}}

	view := app.View()
	if !strings.Contains(view, "Shell($ go test ./ui)") {
		t.Fatalf("expected growing shell preview to keep command header, got:\n%s", view)
	}
	if !strings.Contains(view, "line-10") {
		t.Fatalf("expected growing shell preview to keep newest output, got:\n%s", view)
	}
}

func TestInlineShellPreviewRespectsCtrlOExpansion(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)
	app.state.WaitKind = model.WaitTool
	app.state.Messages = []model.Message{{
		Kind:       model.MsgTool,
		ToolName:   "Shell",
		ToolCallID: "call-shell-1",
		ToolArgs:   "$ go test ./ui",
		Display:    model.DisplayExpanded,
		Content: strings.Join([]string{
			"line-01", "line-02", "line-03", "line-04", "line-05", "line-06", "line-07",
		}, "\n"),
		Streaming: true,
	}}

	collapsed := app.View()
	if !strings.Contains(collapsed, "ctrl+o to expand") {
		t.Fatalf("expected collapsed inline shell preview hint, got:\n%s", collapsed)
	}
	if !strings.Contains(collapsed, "Shell($ go test ./ui)") {
		t.Fatalf("expected collapsed inline shell preview to keep command header, got:\n%s", collapsed)
	}

	nextModel, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlO})
	app = nextModel.(App)
	expanded := app.View()
	if strings.Contains(expanded, "ctrl+o to expand") {
		t.Fatalf("expected expanded inline shell preview without collapse hint, got:\n%s", expanded)
	}
	if !strings.Contains(expanded, "line-07") {
		t.Fatalf("expected expanded inline shell preview to show full output, got:\n%s", expanded)
	}
	if !strings.Contains(expanded, "Shell($ go test ./ui)") {
		t.Fatalf("expected expanded inline shell preview to keep command header, got:\n%s", expanded)
	}
}

func TestInlinePrintShellFinished_PrintsResolvedToolBlockOnce(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	before := []model.Message{{
		Kind:       model.MsgTool,
		ToolName:   "Shell",
		ToolCallID: "call-shell-1",
		ToolArgs:   "$ go test ./ui",
		Display:    model.DisplayExpanded,
		Streaming:  true,
	}}
	app.state.Messages = []model.Message{{
		Kind:       model.MsgTool,
		ToolName:   "Shell",
		ToolCallID: "call-shell-1",
		ToolArgs:   "$ go test ./ui",
		Display:    model.DisplayExpanded,
		Content:    "=== RUN   TestShell\nPASS",
		Summary:    "completed",
	}}

	cmd := app.inlinePrintShellFinished(model.Event{
		Type:       model.CmdFinished,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
		Summary:    "completed",
	}, before)
	body := teaPrintlnBody(t, cmd)
	if !strings.Contains(body, "Shell($ go test ./ui)") {
		t.Fatalf("expected shell finish to include command header, got:\n%s", body)
	}
	if !strings.Contains(body, "PASS") {
		t.Fatalf("expected shell finish to include final output, got:\n%s", body)
	}
}

func TestInlinePrintShellFinished_SuccessWithoutOutputPrintsNothing(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.state.Messages = []model.Message{{
		Kind:       model.MsgTool,
		ToolName:   "Shell",
		ToolCallID: "call-shell-1",
		ToolArgs:   "$ true",
		Display:    model.DisplayExpanded,
		Content:    "(No output)",
		Summary:    "completed",
	}}

	cmd := app.inlinePrintShellFinished(model.Event{
		Type:       model.CmdFinished,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
		Summary:    "completed",
	}, nil)
	if cmd != nil {
		t.Fatal("expected successful no-output shell finish to avoid extra print")
	}
}

func TestCombineCmdsUsesSequenceForMultipleCommands(t *testing.T) {
	cmd := combineCmds(
		func() tea.Msg { return nil },
		func() tea.Msg { return nil },
	)
	if cmd == nil {
		t.Fatal("expected combined command")
	}
	if got := fmt.Sprintf("%T", cmd()); got != "tea.sequenceMsg" {
		t.Fatalf("combined cmd type = %s, want tea.sequenceMsg", got)
	}
}

func TestEnsureWaitForEventUsesSequenceWhenCmdPresent(t *testing.T) {
	app := New(make(chan model.Event), nil, "test", ".", "", "demo-model", 4096)
	cmd := app.ensureWaitForEvent(func() tea.Msg { return nil })
	if cmd == nil {
		t.Fatal("expected wrapped command")
	}
	if got := fmt.Sprintf("%T", cmd()); got != "tea.sequenceMsg" {
		t.Fatalf("wrapped cmd type = %s, want tea.sequenceMsg", got)
	}
}

func teaPrintlnBody(t *testing.T, cmd tea.Cmd) string {
	t.Helper()
	if cmd == nil {
		t.Fatal("expected print command")
	}
	msg := cmd()
	if got := fmt.Sprintf("%T", msg); got != "tea.printLineMessage" {
		t.Fatalf("message type = %s, want tea.printLineMessage", got)
	}
	v := reflect.ValueOf(msg)
	field := v.FieldByName("messageBody")
	if !field.IsValid() || field.Kind() != reflect.String {
		t.Fatal("printLineMessage missing messageBody")
	}
	return field.String()
}
