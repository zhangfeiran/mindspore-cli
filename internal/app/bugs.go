package app

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/vigo999/ms-cli/ui/model"
	"github.com/vigo999/ms-cli/ui/render"
)

func (a *Application) cmdReport(args []string) {
	if !a.ensureIssueService() {
		return
	}
	if len(args) == 0 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Usage: /report <bug title>"}
		return
	}
	title := strings.Join(args, " ")
	bug, err := a.issueService.ReportBug(title, a.issueUser)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("report failed: %v", err)}
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("created bug #%d: %s", bug.ID, bug.Title),
	}
}

func (a *Application) cmdBugs(args []string) {
	if !a.ensureIssueService() {
		return
	}
	status := "all"
	if len(args) > 0 {
		status = args[0]
	}
	listStatus := status
	if status == "all" {
		listStatus = ""
	}
	bugs, err := a.issueService.ListBugs(listStatus)
	if err != nil {
		a.EventCh <- model.Event{
			Type: model.BugIndexOpen,
			BugView: &model.BugEventData{
				Filter: status,
				Err:    err,
			},
		}
		return
	}
	a.EventCh <- model.Event{
		Type: model.BugIndexOpen,
		BugView: &model.BugEventData{
			Filter: status,
			Items:  bugs,
		},
	}
}

func (a *Application) cmdBugDetail(args []string) {
	if !a.ensureIssueService() {
		return
	}
	if len(args) == 0 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Usage: /__bug_detail <bug-id>"}
		return
	}
	id, err := strconv.Atoi(args[0])
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "invalid bug id"}
		return
	}
	bug, err := a.issueService.GetBug(id)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("get bug failed: %v", err)}
		return
	}
	acts, err := a.issueService.GetActivity(id)
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("list activity failed: %v", err)}
		return
	}
	a.EventCh <- model.Event{
		Type: model.BugDetailOpen,
		BugView: &model.BugEventData{
			ID:        id,
			Bug:       bug,
			Activity:  acts,
			FromIndex: true,
		},
	}
}

func (a *Application) cmdClaim(args []string) {
	if !a.ensureIssueService() {
		return
	}
	if len(args) == 0 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Usage: /claim <bug-id>"}
		return
	}
	id, err := strconv.Atoi(args[0])
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "invalid bug id"}
		return
	}
	if err := a.issueService.ClaimBug(id, a.issueUser); err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("claim failed: %v", err)}
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("you claimed bug #%d", id),
	}
}

func (a *Application) cmdClose(args []string) {
	if !a.ensureIssueService() {
		return
	}
	if len(args) == 0 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Usage: /close <bug-id>"}
		return
	}
	id, err := strconv.Atoi(args[0])
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "invalid bug id"}
		return
	}
	if err := a.issueService.CloseBug(id); err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("close failed: %v", err)}
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("closed bug #%d", id),
	}
}

func (a *Application) cmdDock() {
	if !a.ensureIssueService() {
		return
	}
	data, err := a.issueService.DockSummary()
	if err != nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: fmt.Sprintf("dock failed: %v", err)}
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: render.Dock(data),
	}
}
