package panels

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// Style vars are populated by InitStyles() in styles.go.
var (
	bootMessageBaseStyle lipgloss.Style
	bootMessageGlowStyle lipgloss.Style
	bootMessageHotStyle  lipgloss.Style
)

// RenderBootScreen renders a centered splash screen shown before the TUI opens.
func RenderBootScreen(width, height, highlight int) string {
	if width <= 0 || height <= 0 {
		return ""
	}

	content := renderBootShimmer("MindSpore", highlight)
	return lipgloss.Place(width, height, lipgloss.Center, lipgloss.Center, content)
}

func renderBootShimmer(text string, highlight int) string {
	runes := []rune(text)
	if len(runes) == 0 {
		return ""
	}

	parts := make([]string, len(runes))
	bandCenter := highlight % (len(runes) + 6)
	bandCenter -= 3

	for i, r := range runes {
		style := bootMessageBaseStyle
		if r != ' ' {
			dist := absInt(i - bandCenter)
			switch {
			case dist == 0:
				style = bootMessageHotStyle
			case dist <= 2:
				style = bootMessageGlowStyle
			}
		}
		parts[i] = style.Render(string(r))
	}

	return strings.Join(parts, "")
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
