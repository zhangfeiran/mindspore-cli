package model

import "github.com/vigo999/ms-cli/internal/issues"

// Bug-specific UI event types.
const (
	BugIndexOpen  EventType = "BugIndexOpen"
	BugDetailOpen EventType = "BugDetailOpen"
)

type BugMode int

const (
	BugModeNone BugMode = iota
	BugModeIndex
	BugModeDetail
)

type BugIndexState struct {
	Items  []issues.Bug
	Cursor int
	Filter string
	Err    string
}

type BugDetailState struct {
	ID        int
	Bug       *issues.Bug
	Activity  []issues.Activity
	Err       string
	FromIndex bool
}

type BugViewState struct {
	Mode   BugMode
	Index  BugIndexState
	Detail BugDetailState
}

func (s BugViewState) Active() bool {
	return s.Mode != BugModeNone
}

// BugEventData carries bug-specific payloads on Event.
type BugEventData struct {
	Filter    string
	Items     []issues.Bug
	ID        int
	Bug       *issues.Bug
	Activity  []issues.Activity
	FromIndex bool
	Err       error
}
