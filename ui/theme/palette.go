package theme

import "github.com/charmbracelet/lipgloss"

// Palette holds all semantic color tokens used across the UI.
type Palette struct {
	TextPrimary   lipgloss.Color
	TextSecondary lipgloss.Color
	TextMuted     lipgloss.Color
	Accent        lipgloss.Color
	AccentAlt     lipgloss.Color
	Success       lipgloss.Color
	Warning       lipgloss.Color
	Error         lipgloss.Color
	ErrorLight    lipgloss.Color
	Border        lipgloss.Color
	SelectionBG   lipgloss.Color
	SurfaceDim    lipgloss.Color
	Thinking      lipgloss.Color
	BadgeFG       lipgloss.Color
	BadgeFGBright lipgloss.Color
}
