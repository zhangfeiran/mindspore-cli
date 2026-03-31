package theme

import "testing"

func TestApplyDark(t *testing.T) {
	if err := Apply("dark"); err != nil {
		t.Fatal(err)
	}
	if Current.TextPrimary != Dark.TextPrimary {
		t.Fatalf("expected Dark.TextPrimary %q, got %q", Dark.TextPrimary, Current.TextPrimary)
	}
}

func TestApplyLight(t *testing.T) {
	if err := Apply("light"); err != nil {
		t.Fatal(err)
	}
	if Current.TextPrimary != Light.TextPrimary {
		t.Fatalf("expected Light.TextPrimary %q, got %q", Light.TextPrimary, Current.TextPrimary)
	}
}

func TestApplyHighContrast(t *testing.T) {
	if err := Apply("high-contrast"); err != nil {
		t.Fatal(err)
	}
	if Current.TextPrimary != HighContrast.TextPrimary {
		t.Fatalf("expected HighContrast.TextPrimary %q, got %q", HighContrast.TextPrimary, Current.TextPrimary)
	}
}

func TestApplyDefault(t *testing.T) {
	// "default" and "" should both resolve to Dark.
	for _, name := range []string{"", "default"} {
		if err := Apply(name); err != nil {
			t.Fatalf("Apply(%q): %v", name, err)
		}
		if Current.TextPrimary != Dark.TextPrimary {
			t.Fatalf("Apply(%q): expected Dark palette", name)
		}
	}
}

func TestApplyUnknownErrors(t *testing.T) {
	if err := Apply("neon"); err == nil {
		t.Fatal("expected error for unknown theme")
	}
}
