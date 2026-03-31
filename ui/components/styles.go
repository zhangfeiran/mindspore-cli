package components

import (
	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/mindspore-code/ui/theme"
)

// InitStyles rebuilds all package-level style vars from theme.Current.
func InitStyles() {
	t := theme.Current

	// spinner.go
	spinnerStyle = lipgloss.NewStyle().Foreground(t.Thinking)

	// thinking.go
	thinkingStyle = lipgloss.NewStyle().Foreground(t.Thinking).Italic(true)
	thinkingSpinnerStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("212"))

	// textinput.go
	sugCmdStyle = lipgloss.NewStyle().Foreground(t.TextPrimary)
	sugDescStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	sugSelCmdStyle = lipgloss.NewStyle().Foreground(t.AccentAlt).Bold(true)
	sugSelDescStyle = lipgloss.NewStyle().Foreground(t.AccentAlt)
	separatorStyle = lipgloss.NewStyle().Foreground(t.Border)
}
