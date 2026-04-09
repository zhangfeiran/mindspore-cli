package fs

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestReadToolAllowsSessionArtifactPath(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	workDir := t.TempDir()
	artifactDir := filepath.Join(home, ".mscli", "sessions", "bucket", "sess_1", "tool-results")
	if err := os.MkdirAll(artifactDir, 0755); err != nil {
		t.Fatalf("MkdirAll failed: %v", err)
	}
	artifactPath := filepath.Join(artifactDir, "call_1.txt")
	if err := os.WriteFile(artifactPath, []byte("persisted output"), 0600); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	params, err := json.Marshal(map[string]string{"path": artifactPath})
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	result, err := NewReadTool(workDir).Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("tool error = %v", result.Error)
	}
	if got, want := result.Content, "persisted output"; got != want {
		t.Fatalf("result.Content = %q, want %q", got, want)
	}
}
