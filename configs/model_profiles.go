package configs

import "strings"

// ModelTokenProfile defines default token limits for a model family.
type ModelTokenProfile struct {
	ModelMaxTokens int `yaml:"model_max_tokens"`
	ContextWindow  int `yaml:"context_window"`
}

var builtinModelTokenProfiles = map[string]ModelTokenProfile{
	"gpt-5": {
		ModelMaxTokens: 128000,
		ContextWindow:  400000,
	},
	"claude-4": {
		ModelMaxTokens: 64000,
		ContextWindow:  200000,
	},
	"glm-4.7": {
		ModelMaxTokens: 131072,
		ContextWindow:  200000,
	},
	"glm-5": {
		ModelMaxTokens: 131072,
		ContextWindow:  200000,
	},
	"kimi-k2.5": {
		ModelMaxTokens: 32768,
		ContextWindow:  256000,
	},
	"minimax-m2.5": {
		ModelMaxTokens: 204800,
		ContextWindow:  204800,
	},
	"minimax-m2.7": {
		ModelMaxTokens: 204800,
		ContextWindow:  204800,
	},
}

func applyModelTokenDefaults(cfg *Config, defaultModelMaxTokens, defaultContextWindow int) {
	profile, ok := matchModelTokenProfile(cfg.Model.Model, cfg.ModelProfiles)
	if !ok {
		return
	}

	if cfg.Model.MaxTokens == defaultModelMaxTokens && profile.ModelMaxTokens > 0 {
		cfg.Model.MaxTokens = profile.ModelMaxTokens
	}
	if cfg.Context.Window == defaultContextWindow && profile.ContextWindow > 0 {
		cfg.Context.Window = profile.ContextWindow
	}
}

func matchModelTokenProfile(modelName string, custom map[string]ModelTokenProfile) (ModelTokenProfile, bool) {
	normalizedModel := strings.ToLower(strings.TrimSpace(modelName))
	if normalizedModel == "" {
		return ModelTokenProfile{}, false
	}

	for prefix, profile := range custom {
		normalizedPrefix := strings.ToLower(strings.TrimSpace(prefix))
		if normalizedPrefix != "" && strings.HasPrefix(normalizedModel, normalizedPrefix) {
			return profile, true
		}
	}

	for prefix, profile := range builtinModelTokenProfiles {
		if strings.HasPrefix(normalizedModel, prefix) {
			return profile, true
		}
	}

	return ModelTokenProfile{}, false
}
