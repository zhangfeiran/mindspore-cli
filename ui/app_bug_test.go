package ui

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/ms-cli/internal/issues"
	"github.com/vigo999/ms-cli/ui/model"
)

func TestBugViewUsesDedicatedSurfaceAndEscReturnsToList(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 24})
	app = next.(App)

	now := time.Now()
	next, _ = app.handleEvent(model.Event{
		Type: model.BugIndexOpen,
		BugView: &model.BugEventData{
			Filter: "open",
			Items: []issues.Bug{
				{ID: 1042, Title: "loss spike after dataloader refactor", Status: "open", Reporter: "travis", UpdatedAt: now},
				{ID: 1041, Title: "npu build fails on cann 8.0.RC3", Status: "doing", Lead: "travis", Reporter: "alice", UpdatedAt: now},
			},
		},
	})
	app = next.(App)

	view := app.View()
	if !strings.Contains(view, "BUGS") {
		t.Fatalf("expected bug view header, got:\n%s", view)
	}
	if strings.Contains(view, "> ") {
		t.Fatalf("expected chat composer to be hidden in bug view, got:\n%s", view)
	}

	app.bugView.Index.Cursor = 1
	next, _ = app.handleEvent(model.Event{
		Type: model.BugDetailOpen,
		BugView: &model.BugEventData{
			ID:        1041,
			Bug:       &issues.Bug{ID: 1041, Title: "npu build fails on cann 8.0.RC3", Status: "doing", Lead: "travis", Reporter: "alice", UpdatedAt: now},
			Activity:  []issues.Activity{{BugID: 1041, Actor: "travis", Text: "travis claimed bug", CreatedAt: now}},
			FromIndex: true,
		},
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	app = next.(App)

	if app.bugView.Mode != model.BugModeIndex {
		t.Fatalf("expected to return to bug index, got mode %v", app.bugView.Mode)
	}
	if app.bugView.Index.Cursor != 1 {
		t.Fatalf("expected cursor to stay on previous row, got %d", app.bugView.Index.Cursor)
	}
}
