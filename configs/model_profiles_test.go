package configs

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadWithEnv_AutoTokenLimitsByModel(t *testing.T) {
	clearEnv(t)

	home := t.TempDir()
	t.Setenv("HOME", home)
	projectDir := t.TempDir()
	t.Chdir(projectDir)

	projectPath := filepath.Join(projectDir, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(projectPath), 0755); err != nil {
		t.Fatalf("mkdir project dir: %v", err)
	}
	if err := os.WriteFile(projectPath, []byte("model:\n  model: gpt-5\n"), 0600); err != nil {
		t.Fatalf("write project config: %v", err)
	}

	cfg, err := LoadWithEnv()
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}

	if got, want := cfg.Model.MaxTokens, 128000; got != want {
		t.Fatalf("model.max_tokens = %d, want %d", got, want)
	}
	if got, want := cfg.Context.Window, 400000; got != want {
		t.Fatalf("context.window = %d, want %d", got, want)
	}
}

func TestLoadWithEnv_EnvOverridesAutoTokenLimits(t *testing.T) {
	clearEnv(t)

	home := t.TempDir()
	t.Setenv("HOME", home)
	projectDir := t.TempDir()
	t.Chdir(projectDir)

	projectPath := filepath.Join(projectDir, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(projectPath), 0755); err != nil {
		t.Fatalf("mkdir project dir: %v", err)
	}
	if err := os.WriteFile(projectPath, []byte("model:\n  model: gpt-5\n"), 0600); err != nil {
		t.Fatalf("write project config: %v", err)
	}

	t.Setenv("MSCLI_MAX_TOKENS", "8192")
	t.Setenv("MSCLI_CONTEXT_WINDOW", "16000")

	cfg, err := LoadWithEnv()
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}

	if got, want := cfg.Model.MaxTokens, 8192; got != want {
		t.Fatalf("model.max_tokens = %d, want %d", got, want)
	}
	if got, want := cfg.Context.Window, 16000; got != want {
		t.Fatalf("context.window = %d, want %d", got, want)
	}
}

func TestLoadWithEnv_ConfigOverridesAutoTokenLimits(t *testing.T) {
	clearEnv(t)

	home := t.TempDir()
	t.Setenv("HOME", home)
	projectDir := t.TempDir()
	t.Chdir(projectDir)

	projectPath := filepath.Join(projectDir, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(projectPath), 0755); err != nil {
		t.Fatalf("mkdir project dir: %v", err)
	}
	if err := os.WriteFile(projectPath, []byte(`model:
  model: gpt-5
  max_tokens: 2048
context:
  window: 12000
`), 0600); err != nil {
		t.Fatalf("write project config: %v", err)
	}

	cfg, err := LoadWithEnv()
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}

	if got, want := cfg.Model.MaxTokens, 2048; got != want {
		t.Fatalf("model.max_tokens = %d, want %d", got, want)
	}
	if got, want := cfg.Context.Window, 12000; got != want {
		t.Fatalf("context.window = %d, want %d", got, want)
	}
}

func TestLoadWithEnv_CustomModelProfiles(t *testing.T) {
	clearEnv(t)

	home := t.TempDir()
	t.Setenv("HOME", home)
	projectDir := t.TempDir()
	t.Chdir(projectDir)

	projectPath := filepath.Join(projectDir, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(projectPath), 0755); err != nil {
		t.Fatalf("mkdir project dir: %v", err)
	}
	if err := os.WriteFile(projectPath, []byte(`model:
  model: my-inhouse-model-v2
model_profiles:
  my-inhouse-model:
    model_max_tokens: 7777
    context_window: 55555
`), 0600); err != nil {
		t.Fatalf("write project config: %v", err)
	}

	cfg, err := LoadWithEnv()
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}

	if got, want := cfg.Model.MaxTokens, 7777; got != want {
		t.Fatalf("model.max_tokens = %d, want %d", got, want)
	}
	if got, want := cfg.Context.Window, 55555; got != want {
		t.Fatalf("context.window = %d, want %d", got, want)
	}
}

func TestContextConfig_AcceptsLegacyMaxTokens(t *testing.T) {
	clearEnv(t)

	home := t.TempDir()
	t.Setenv("HOME", home)
	projectDir := t.TempDir()
	t.Chdir(projectDir)

	projectPath := filepath.Join(projectDir, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(projectPath), 0755); err != nil {
		t.Fatalf("mkdir project dir: %v", err)
	}
	if err := os.WriteFile(projectPath, []byte(`model:
  model: gpt-4o-mini
context:
  max_tokens: 18000
`), 0600); err != nil {
		t.Fatalf("write project config: %v", err)
	}

	cfg, err := LoadWithEnv()
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}

	if got, want := cfg.Context.Window, 18000; got != want {
		t.Fatalf("context.window = %d, want %d", got, want)
	}
}
