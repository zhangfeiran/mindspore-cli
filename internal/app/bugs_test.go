package app

import (
	"testing"
	"time"

	"github.com/vigo999/ms-cli/internal/issues"
	"github.com/vigo999/ms-cli/ui/model"
)

func TestCmdBugsDefaultsToAllAndOpensBugIndexView(t *testing.T) {
	store := &fakeIssueStore{
		bugs: []issues.Bug{
			{ID: 1042, Title: "loss spike after dataloader refactor", Status: "open", Reporter: "travis", UpdatedAt: time.Now()},
		},
	}
	app := &Application{
		EventCh:      make(chan model.Event, 4),
		issueService: issues.NewService(store),
	}

	app.cmdBugs(nil)

	ev := <-app.EventCh
	if ev.Type != model.BugIndexOpen {
		t.Fatalf("event type = %s, want %s", ev.Type, model.BugIndexOpen)
	}
	if store.lastListStatus != "" {
		t.Fatalf("list status = %q, want empty for all", store.lastListStatus)
	}
	if ev.BugView == nil || ev.BugView.Filter != "all" {
		t.Fatalf("bug view filter = %#v, want all", ev.BugView)
	}
	if got := len(ev.BugView.Items); got != 1 {
		t.Fatalf("bug count = %d, want 1", got)
	}
}

func TestCmdBugDetailOpensDetailViewEvent(t *testing.T) {
	store := &fakeIssueStore{
		bug: &issues.Bug{ID: 1042, Title: "loss spike after dataloader refactor", Status: "open", Reporter: "travis", UpdatedAt: time.Now()},
		activity: []issues.Activity{
			{BugID: 1042, Actor: "travis", Text: "created bug", CreatedAt: time.Now()},
		},
	}
	app := &Application{
		EventCh:      make(chan model.Event, 4),
		issueService: issues.NewService(store),
	}

	app.cmdBugDetail([]string{"1042"})

	ev := <-app.EventCh
	if ev.Type != model.BugDetailOpen {
		t.Fatalf("event type = %s, want %s", ev.Type, model.BugDetailOpen)
	}
	if ev.BugView == nil || ev.BugView.Bug == nil {
		t.Fatalf("missing bug detail payload: %#v", ev.BugView)
	}
	if ev.BugView.Bug.ID != 1042 {
		t.Fatalf("bug id = %d, want 1042", ev.BugView.Bug.ID)
	}
	if got := len(ev.BugView.Activity); got != 1 {
		t.Fatalf("activity count = %d, want 1", got)
	}
}

type fakeIssueStore struct {
	lastListStatus string
	bugs           []issues.Bug
	bug            *issues.Bug
	activity       []issues.Activity
}

func (f *fakeIssueStore) CreateBug(title, reporter string) (*issues.Bug, error) {
	return nil, nil
}

func (f *fakeIssueStore) ListBugs(status string) ([]issues.Bug, error) {
	f.lastListStatus = status
	return f.bugs, nil
}

func (f *fakeIssueStore) GetBug(id int) (*issues.Bug, error) {
	return f.bug, nil
}

func (f *fakeIssueStore) ClaimBug(id int, lead string) error {
	return nil
}

func (f *fakeIssueStore) CloseBug(id int) error {
	return nil
}

func (f *fakeIssueStore) AddNote(bugID int, author, content string) (*issues.Note, error) {
	return nil, nil
}

func (f *fakeIssueStore) ListActivity(bugID int) ([]issues.Activity, error) {
	return f.activity, nil
}

func (f *fakeIssueStore) DockSummary() (*issues.DockData, error) {
	return nil, nil
}
