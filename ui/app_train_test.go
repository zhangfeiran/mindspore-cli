package ui

import (
	"strings"
	"testing"

	"github.com/vigo999/ms-cli/ui/model"
)

func TestTrainFixActionClearsStaleButtonAndKeepsCompletionMessage(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.TrainModeOpen,
		Train: &model.TrainEventData{
			RunID:    "primary",
			RawInput: "qwen3 lora",
			Model:    "qwen3",
			Method:   "lora",
		},
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:    model.TrainAnalysisReady,
		Message: "Analysis complete. Ready to apply fix.",
		Train: &model.TrainEventData{
			RunID:       "primary",
			IssueType:   "failure",
			ActionID:    "fix-dsa-op",
			ActionKind:  "apply_patch",
			ActionLabel: "apply fix",
		},
	})
	app = next.(App)

	if got := len(app.trainView.GlobalActions.Items); got == 0 || app.trainView.GlobalActions.Items[0].ID != "fix-dsa-op" {
		t.Fatalf("expected fix action before execution, got %#v", app.trainView.GlobalActions.Items)
	}

	next, _ = app.handleEvent(model.Event{
		Type:    model.TrainActionApplied,
		Message: "op-agent: implementing DSA operator and compiling custom torch-npu...",
		Train: &model.TrainEventData{
			RunID:     "primary",
			IssueType: "failure",
			ActionID:  "fix-dsa-op",
		},
	})
	app = next.(App)

	run := app.trainView.RunByID("primary")
	if run == nil {
		t.Fatal("expected primary run")
	}
	if run.Phase != model.TrainPhaseFixing {
		t.Fatalf("expected fixing phase, got %s", run.Phase)
	}
	if len(run.AgentActions) != 0 {
		t.Fatalf("expected stale agent actions to be cleared, got %#v", run.AgentActions)
	}
	if got := len(app.trainView.GlobalActions.Items); got != 1 || app.trainView.GlobalActions.Items[0].ID != "stop" {
		t.Fatalf("expected only stop action while fixing, got %#v", app.trainView.GlobalActions.Items)
	}

	next, _ = app.handleEvent(model.Event{
		Type:    model.TrainFixApplied,
		Message: "op-agent: DSA operator finished. New torch wheel is ready. Please rerun experiment.",
		Train: &model.TrainEventData{
			RunID:      "primary",
			FixSummary: "DSA operator implemented and torch-npu recompiled",
		},
	})
	app = next.(App)

	run = app.trainView.RunByID("primary")
	if run == nil {
		t.Fatal("expected primary run after fix")
	}
	if run.Phase != model.TrainPhaseReady {
		t.Fatalf("expected ready phase after fix, got %s", run.Phase)
	}
	if !run.FixApplied {
		t.Fatal("expected run to be marked as fix-applied")
	}
	if got := len(app.trainView.GlobalActions.Items); got != 1 || app.trainView.GlobalActions.Items[0].Label != "rerun" {
		t.Fatalf("expected rerun action after fix, got %#v", app.trainView.GlobalActions.Items)
	}
	if run.StatusMessage != "op-agent: DSA operator finished. New torch wheel is ready. Please rerun experiment." {
		t.Fatalf("expected status message to keep fix completion text, got %q", run.StatusMessage)
	}
	last := app.state.Messages[len(app.state.Messages)-1]
	if !strings.Contains(last.Content, "DSA operator finished") {
		t.Fatalf("expected final agent message to include fix completion text, got %#v", last)
	}
}
