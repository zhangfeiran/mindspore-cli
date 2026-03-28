package skills

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// SkillMeta holds the frontmatter metadata from a SKILL.md file.
type SkillMeta struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
}

// SkillSummary is a skill's identity used for system prompt listing.
type SkillSummary struct {
	Name        string
	Description string
}

// Loader discovers and loads skills from one or more filesystem paths.
// Paths are ordered from lowest to highest priority; later paths override
// earlier ones when the same skill name appears in multiple locations.
type Loader struct {
	paths []string
}

// NewLoader creates a Loader that searches the given paths in order.
// Empty or non-existent paths are silently skipped.
func NewLoader(paths ...string) *Loader {
	var valid []string
	for _, p := range paths {
		if p != "" {
			valid = append(valid, p)
		}
	}
	return &Loader{paths: valid}
}

// List returns all available skills, deduplicated by name.
// If the same skill name appears in multiple paths, the higher-priority
// (later) path wins. Results are sorted alphabetically by name.
func (l *Loader) List() []SkillSummary {
	seen := make(map[string]SkillSummary)
	for _, p := range l.paths {
		for _, s := range scanDir(p) {
			seen[s.Name] = s
		}
	}
	result := make([]SkillSummary, 0, len(seen))
	for _, s := range seen {
		result = append(result, s)
	}
	sort.Slice(result, func(i, j int) bool {
		return result[i].Name < result[j].Name
	})
	return result
}

// Names returns the names of all available skills, sorted alphabetically.
func (l *Loader) Names() []string {
	summaries := l.List()
	names := make([]string, len(summaries))
	for i, s := range summaries {
		names[i] = s.Name
	}
	return names
}

// Load finds the highest-priority occurrence of the named skill,
// reads its SKILL.md, strips the YAML frontmatter, and returns the
// body wrapped in <skill ...>...</skill> tags with absolute source
// location metadata.
func (l *Loader) Load(name string) (string, error) {
	lookupName := strings.TrimSpace(name)
	lookupKey := normalizedSkillLookupKey(lookupName)
	// Iterate in reverse so highest-priority path wins.
	for i := len(l.paths) - 1; i >= 0; i-- {
		dirName, ok := findSkillDir(l.paths[i], lookupName, lookupKey)
		if !ok {
			continue
		}
		path := filepath.Join(l.paths[i], dirName, "SKILL.md")
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		meta, body, err := parseFrontmatter(string(data))
		if err != nil {
			return "", fmt.Errorf("parse skill %q: %w", name, err)
		}
		// Use directory name as skill name if frontmatter name is empty.
		skillName := meta.Name
		if skillName == "" {
			skillName = dirName
		}
		return wrapContent(skillName, absolutePath(path), body), nil
	}
	return "", fmt.Errorf("skill %q not found", name)
}

// FormatSummaries formats a slice of SkillSummary for inclusion in the
// system prompt. Each skill is listed as "- name: description".
func FormatSummaries(summaries []SkillSummary) string {
	if len(summaries) == 0 {
		return ""
	}
	var sb strings.Builder
	for _, s := range summaries {
		fmt.Fprintf(&sb, "- %s: %s\n", s.Name, s.Description)
	}
	return sb.String()
}

// scanDir reads all skill subdirectories under dir and returns their
// metadata. Directories without a readable SKILL.md are skipped.
func scanDir(dir string) []SkillSummary {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var result []SkillSummary
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		skillFile := filepath.Join(dir, e.Name(), "SKILL.md")
		data, err := os.ReadFile(skillFile)
		if err != nil {
			continue
		}
		meta, _, err := parseFrontmatter(string(data))
		if err != nil {
			continue
		}
		name := meta.Name
		if name == "" {
			name = e.Name()
		}
		desc := meta.Description
		result = append(result, SkillSummary{Name: name, Description: desc})
	}
	return result
}

func findSkillDir(rootDir, lookupName, lookupKey string) (string, bool) {
	entries, err := os.ReadDir(rootDir)
	if err != nil {
		return "", false
	}

	var aliasMatch string
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		dirName := entry.Name()
		if dirName == lookupName {
			return dirName, true
		}
		if aliasMatch == "" && normalizedSkillLookupKey(dirName) == lookupKey {
			aliasMatch = dirName
		}
	}
	if aliasMatch != "" {
		return aliasMatch, true
	}
	return "", false
}

// parseFrontmatter splits a SKILL.md into YAML frontmatter and body.
// The frontmatter is delimited by lines containing only "---".
// If no frontmatter is found the entire content is returned as body.
func parseFrontmatter(content string) (SkillMeta, string, error) {
	normalized := strings.ReplaceAll(content, "\r\n", "\n")
	normalized = strings.TrimPrefix(normalized, "\ufeff")
	lines := strings.Split(normalized, "\n")

	if len(lines) < 3 || strings.TrimSpace(lines[0]) != "---" {
		return SkillMeta{}, strings.TrimSpace(content), nil
	}

	end := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			end = i
			break
		}
	}
	if end == -1 {
		return SkillMeta{}, strings.TrimSpace(content), nil
	}

	var meta SkillMeta
	frontMatter := strings.Join(lines[1:end], "\n")
	if err := yaml.Unmarshal([]byte(frontMatter), &meta); err != nil {
		return SkillMeta{}, "", fmt.Errorf("unmarshal frontmatter: %w", err)
	}

	body := ""
	if end+1 < len(lines) {
		body = strings.TrimSpace(strings.Join(lines[end+1:], "\n"))
	}
	return meta, body, nil
}

func absolutePath(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		return path
	}
	return abs
}

func normalizedSkillLookupKey(name string) string {
	return strings.ReplaceAll(strings.TrimSpace(name), "_", "-")
}

// wrapContent wraps skill body in <skill ...>...</skill> tags and includes
// the absolute skill location so relative references next to SKILL.md are
// discoverable to the model.
func wrapContent(name, skillFile, body string) string {
	skillDir := filepath.Dir(skillFile)
	return fmt.Sprintf(
		"<skill name=%q location=%q source=%q>\nThe skill directory is %s. Resolve files mentioned next to SKILL.md from that directory.\n\n%s\n</skill>",
		name,
		skillDir,
		skillFile,
		skillDir,
		body,
	)
}
