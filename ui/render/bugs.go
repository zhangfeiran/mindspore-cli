package render

import (
	"fmt"

	"github.com/vigo999/ms-cli/internal/issues"
)

func BugList(bugs []issues.Bug) string {
	idW, titleW, statusW, leadW, reporterW := 2, 12, 6, 4, 8
	for _, b := range bugs {
		if l := len(fmt.Sprintf("%d", b.ID)); l > idW {
			idW = l
		}
		if len(b.Title) > titleW {
			titleW = len(b.Title)
		}
		if len(b.Status) > statusW {
			statusW = len(b.Status)
		}
		lead := b.Lead
		if lead == "" {
			lead = "-"
		}
		if len(lead) > leadW {
			leadW = len(lead)
		}
		if len(b.Reporter) > reporterW {
			reporterW = len(b.Reporter)
		}
	}
	if titleW > 50 {
		titleW = 50
	}

	centerPad := func(s string, w int) string {
		gap := w - len(s)
		if gap <= 0 {
			return s
		}
		left := gap / 2
		right := gap - left
		return fmt.Sprintf("%*s%s%*s", left, "", s, right, "")
	}

	var lines []string
	header := fmt.Sprintf("  %-*s  %-*s  %-*s  %-*s  %-*s",
		idW, "id", titleW, "title", statusW, "status", leadW, "lead", reporterW, "reporter")
	lines = append(lines, TitleStyle.Render("BUG LIST"))
	lines = append(lines, TitleStyle.Render(header))

	for _, b := range bugs {
		idStr := centerPad(fmt.Sprintf("%d", b.ID), idW)
		title := b.Title
		if len(title) > titleW {
			title = title[:titleW-3] + "..."
		}
		lead := b.Lead
		if lead == "" {
			lead = "-"
		}
		statusStyle := StatusOpenStyle
		if b.Status == "doing" {
			statusStyle = StatusDoingStyle
		}
		line := fmt.Sprintf("  %s  %-*s  %s  %s  %s",
			idStr, titleW, title,
			statusStyle.Render(centerPad(b.Status, statusW)),
			centerPad(lead, leadW),
			centerPad(b.Reporter, reporterW))
		lines = append(lines, line)
	}
	return Box(lines)
}

func Dock(data *issues.DockData) string {
	lines := []string{
		TitleStyle.Render("dock"),
		"",
		fmt.Sprintf("  %s %s",
			LabelStyle.Render("open bugs"),
			ValueStyle.Render(fmt.Sprintf("%d", data.OpenCount)),
		),
	}

	if len(data.ReadyBugs) > 0 {
		lines = append(lines, "", LabelStyle.Render("  ready (unassigned)"))
		for _, b := range data.ReadyBugs {
			lines = append(lines, fmt.Sprintf("    %d  %s  %s",
				b.ID, b.Title, StatusOpenStyle.Render(b.Status)))
		}
	}

	if len(data.RecentFeed) > 0 {
		lines = append(lines, "", LabelStyle.Render("  recent activity"))
		for _, a := range data.RecentFeed {
			ts := a.CreatedAt.Format("01-02 15:04")
			lines = append(lines, ActivityStyle.Render(fmt.Sprintf("    %s  %s  %s", ts, a.Actor, a.Text)))
		}
	}

	return Box(lines)
}
