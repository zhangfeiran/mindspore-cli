package app

import (
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/vigo999/ms-cli/integrations/skills"
	"github.com/vigo999/ms-cli/ui/model"
)

func (a *Application) cmdSkillAddInput(raw string) {
	if a.skillLoader == nil {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "Skills not available."}
		return
	}

	sourceDir, err := resolveSkillAddSource(strings.TrimSpace(raw), a.WorkDir, a.skillsHomeDir)
	if err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "skill-add",
			Message:  fmt.Sprintf("Failed to add local skill: %v", err),
		}
		return
	}

	destRoot, err := a.localSkillsDir()
	if err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "skill-add",
			Message:  fmt.Sprintf("Failed to add local skill: %v", err),
		}
		return
	}
	if err := os.MkdirAll(destRoot, 0o755); err != nil {
		a.EventCh <- model.Event{
			Type:     model.ToolError,
			ToolName: "skill-add",
			Message:  fmt.Sprintf("Failed to add local skill: create destination: %v", err),
		}
		return
	}

	destDir := filepath.Join(destRoot, filepath.Base(sourceDir))
	if !samePath(sourceDir, destDir) {
		if err := copySkillDir(sourceDir, destDir); err != nil {
			a.EventCh <- model.Event{
				Type:     model.ToolError,
				ToolName: "skill-add",
				Message:  fmt.Sprintf("Failed to add local skill: %v", err),
			}
			return
		}
	}

	a.refreshSkillCatalog()
	a.emitAvailableSkills(true)
}

func (a *Application) emitAvailableSkills(includeUsage bool) {
	if a == nil || a.skillLoader == nil || a.EventCh == nil {
		return
	}

	summaries := a.skillLoader.List()
	if len(summaries) == 0 {
		a.EventCh <- model.Event{Type: model.AgentReply, Message: "No skills available."}
		return
	}

	msg := "Available skills:\n\n" + skills.FormatSummaries(summaries)
	if includeUsage {
		msg += "\nUsage: /skill <name> [request...] (omit request to start the skill immediately)"
	}
	a.EventCh <- model.Event{Type: model.AgentReply, Message: msg}
}

func (a *Application) localSkillsDir() (string, error) {
	home := strings.TrimSpace(a.skillsHomeDir)
	if home == "" {
		var err error
		home, err = os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve home directory: %w", err)
		}
	}
	if strings.TrimSpace(home) == "" {
		return "", fmt.Errorf("home directory is required")
	}
	return filepath.Join(home, ".ms-cli", "skills"), nil
}

func resolveSkillAddSource(rawPath, workDir, homeDir string) (string, error) {
	path := trimQuotedPath(rawPath)
	if path == "" {
		return "", fmt.Errorf("usage: /skill-add <local-path>")
	}

	if strings.HasPrefix(path, "~"+string(os.PathSeparator)) {
		home := strings.TrimSpace(homeDir)
		if home == "" {
			var err error
			home, err = os.UserHomeDir()
			if err != nil {
				return "", fmt.Errorf("resolve home directory: %w", err)
			}
		}
		path = filepath.Join(home, strings.TrimPrefix(path, "~"+string(os.PathSeparator)))
	}

	if !filepath.IsAbs(path) {
		base := strings.TrimSpace(workDir)
		if base == "" {
			base = "."
		}
		path = filepath.Join(base, path)
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("resolve skill path: %w", err)
	}

	info, err := os.Stat(absPath)
	if err != nil {
		return "", fmt.Errorf("stat skill path: %w", err)
	}

	sourceDir := absPath
	if !info.IsDir() {
		if !strings.EqualFold(info.Name(), "SKILL.md") {
			return "", fmt.Errorf("skill path must point to a skill directory or SKILL.md")
		}
		sourceDir = filepath.Dir(absPath)
	}

	if _, err := os.Stat(filepath.Join(sourceDir, "SKILL.md")); err != nil {
		return "", fmt.Errorf("SKILL.md not found under %s", sourceDir)
	}

	return sourceDir, nil
}

func trimQuotedPath(path string) string {
	path = strings.TrimSpace(path)
	if len(path) >= 2 {
		if (path[0] == '"' && path[len(path)-1] == '"') || (path[0] == '\'' && path[len(path)-1] == '\'') {
			return strings.TrimSpace(path[1 : len(path)-1])
		}
	}
	return path
}

func samePath(left, right string) bool {
	leftAbs, err := filepath.Abs(left)
	if err != nil {
		leftAbs = left
	}
	rightAbs, err := filepath.Abs(right)
	if err != nil {
		rightAbs = right
	}
	return filepath.Clean(leftAbs) == filepath.Clean(rightAbs)
}

func copySkillDir(sourceDir, destDir string) error {
	if err := os.RemoveAll(destDir); err != nil {
		return fmt.Errorf("reset destination %s: %w", destDir, err)
	}

	return filepath.WalkDir(sourceDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		rel, err := filepath.Rel(sourceDir, path)
		if err != nil {
			return err
		}
		targetPath := filepath.Join(destDir, rel)

		if d.IsDir() {
			info, err := d.Info()
			if err != nil {
				return err
			}
			return os.MkdirAll(targetPath, info.Mode().Perm())
		}
		if d.Type()&fs.ModeSymlink != 0 {
			return fmt.Errorf("symlinked files are not supported: %s", path)
		}
		if !d.Type().IsRegular() {
			return fmt.Errorf("unsupported file type: %s", path)
		}
		return copySkillFile(path, targetPath, d)
	})
}

func copySkillFile(sourcePath, destPath string, d fs.DirEntry) error {
	info, err := d.Info()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return err
	}

	sourceFile, err := os.Open(sourcePath)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.OpenFile(destPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, info.Mode().Perm())
	if err != nil {
		return err
	}
	defer destFile.Close()

	if _, err := io.Copy(destFile, sourceFile); err != nil {
		return err
	}
	return nil
}
