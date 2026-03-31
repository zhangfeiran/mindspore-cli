package render

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// Style vars are populated by InitStyles() in styles.go.
var (
	BoxBorderStyle   lipgloss.Style
	TitleStyle       lipgloss.Style
	LabelStyle       lipgloss.Style
	ValueStyle       lipgloss.Style
	StatusOpenStyle  lipgloss.Style
	StatusDoingStyle lipgloss.Style
	ActivityStyle    lipgloss.Style
)

func Box(lines []string) string {
	width := 0
	for _, line := range lines {
		if w := lipgloss.Width(line); w > width {
			width = w
		}
	}
	if width < 24 {
		width = 24
	}

	border := BoxBorderStyle.Render
	boxed := make([]string, 0, len(lines)+2)
	boxed = append(boxed, border("╭"+strings.Repeat("─", width+2)+"╮"))
	for _, line := range lines {
		visible := lipgloss.Width(line)
		pad := width - visible
		if pad < 0 {
			pad = 0
		}
		boxed = append(boxed, border("│")+" "+line+strings.Repeat(" ", pad)+" "+border("│"))
	}
	boxed = append(boxed, border("╰"+strings.Repeat("─", width+2)+"╯"))
	return strings.Join(boxed, "\n")
}
