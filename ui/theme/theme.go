package theme

import "fmt"

// Dark is the default palette optimised for dark terminal backgrounds.
var Dark = Palette{
	TextPrimary:   "252",
	TextSecondary: "244",
	TextMuted:     "240",
	Accent:        "39",
	AccentAlt:     "117",
	Success:       "114",
	Warning:       "214",
	Error:         "196",
	ErrorLight:    "203",
	Border:        "238",
	SelectionBG:   "237",
	SurfaceDim:    "236",
	Thinking:      "205",
	BadgeFG:       "16",
	BadgeFGBright: "255",
}

// Light is a palette optimised for light terminal backgrounds.
var Light = Palette{
	TextPrimary:   "235",
	TextSecondary: "242",
	TextMuted:     "245",
	Accent:        "33",
	AccentAlt:     "33",
	Success:       "28",
	Warning:       "166",
	Error:         "160",
	ErrorLight:    "167",
	Border:        "250",
	SelectionBG:   "254",
	SurfaceDim:    "255",
	Thinking:      "163",
	BadgeFG:       "15",
	BadgeFGBright: "15",
}

// HighContrast is a palette for maximum readability.
var HighContrast = Palette{
	TextPrimary:   "15",
	TextSecondary: "7",
	TextMuted:     "8",
	Accent:        "14",
	AccentAlt:     "14",
	Success:       "10",
	Warning:       "11",
	Error:         "9",
	ErrorLight:    "9",
	Border:        "15",
	SelectionBG:   "0",
	SurfaceDim:    "0",
	Thinking:      "13",
	BadgeFG:       "0",
	BadgeFGBright: "15",
}

// Current is the active palette. It defaults to Dark so that
// unmigrated code keeps working without any init call.
var Current = Dark

// Apply sets Current to the palette matching name.
// Accepted names: "dark", "light", "high-contrast".
// An empty string or "default" selects Dark.
func Apply(name string) error {
	switch name {
	case "", "default", "dark":
		Current = Dark
	case "light":
		Current = Light
	case "high-contrast":
		Current = HighContrast
	default:
		return fmt.Errorf("unknown theme %q (valid: dark, light, high-contrast)", name)
	}
	return nil
}
