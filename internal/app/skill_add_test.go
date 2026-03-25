package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vigo999/ms-cli/integrations/skills"
	"github.com/vigo999/ms-cli/ui/model"
)

func TestCmdSkillAddInputCopiesLocalSkillAndListsAvailableSkills(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	sourceRoot := t.TempDir()
	sourceDir := filepath.Join(sourceRoot, "demo-skill")
	if err := os.MkdirAll(sourceDir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	skillBody := "---\nname: demo-skill\ndescription: Demo skill\n---\n\nbody"
	if err := os.WriteFile(filepath.Join(sourceDir, "SKILL.md"), []byte(skillBody), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	app := &Application{
		EventCh:       make(chan model.Event, 16),
		WorkDir:       sourceRoot,
		skillsHomeDir: home,
		skillLoader:   skills.NewLoader(filepath.Join(home, ".ms-cli", "skills")),
	}

	app.cmdSkillAddInput(sourceDir)

	destSkillFile := filepath.Join(home, ".ms-cli", "skills", "demo-skill", "SKILL.md")
	if _, err := os.Stat(destSkillFile); err != nil {
		t.Fatalf("expected copied skill at %s: %v", destSkillFile, err)
	}

	if _, err := app.skillLoader.Load("demo-skill"); err != nil {
		t.Fatalf("skillLoader.Load() error = %v", err)
	}

	adding := drainUntilEventType(t, app, model.ToolSkill)
	if got, want := adding.ToolName, "Skill add"; got != want {
		t.Fatalf("tool name = %q, want %q", got, want)
	}
	if got, want := adding.Summary, "adding demo-skill to ~/.ms-cli/skills/"; got != want {
		t.Fatalf("summary = %q, want %q", got, want)
	}

	ready := drainUntilEventType(t, app, model.ToolSkill)
	if got, want := ready.ToolName, "Skill ready: 1 available"; got != want {
		t.Fatalf("tool name = %q, want %q", got, want)
	}
	if !strings.Contains(ready.Summary, "demo-skill") {
		t.Fatalf("expected skill summary to include demo-skill, got %q", ready.Summary)
	}
}
