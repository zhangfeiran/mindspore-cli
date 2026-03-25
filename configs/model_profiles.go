package configs

import (
	"sort"
	"strings"
)

// ModelTokenProfile defines default token limits for a model family.
type ModelTokenProfile struct {
	ModelMaxTokens int `yaml:"model_max_tokens"`
	ContextWindow  int `yaml:"context_window"`
}

var builtinModelTokenProfiles = map[string]ModelTokenProfile{
	// OpenAI GPT-5 family.
	"gpt-5.4": {ModelMaxTokens: 128000, ContextWindow: 1050000},
	"gpt-5.3": {ModelMaxTokens: 128000, ContextWindow: 400000},
	"gpt-5.2": {ModelMaxTokens: 128000, ContextWindow: 400000},
	"gpt-5.1": {ModelMaxTokens: 128000, ContextWindow: 400000},
	"gpt-5":   {ModelMaxTokens: 128000, ContextWindow: 400000},

	// Anthropic Claude 4.5-4.6 family.
	"claude-opus-4.6":   {ModelMaxTokens: 128000, ContextWindow: 1000000},
	"claude-opus-4-6":   {ModelMaxTokens: 128000, ContextWindow: 1000000},
	"claude-sonnet-4.6": {ModelMaxTokens: 64000, ContextWindow: 1000000},
	"claude-sonnet-4-6": {ModelMaxTokens: 64000, ContextWindow: 1000000},
	"claude-haiku-4.5":  {ModelMaxTokens: 64000, ContextWindow: 200000},
	"claude-haiku-4-5":  {ModelMaxTokens: 64000, ContextWindow: 200000},

	// GLM family.
	"glm-4.7": {ModelMaxTokens: 131072, ContextWindow: 200000},
	"glm-5":   {ModelMaxTokens: 131072, ContextWindow: 200000},

	// Moonshot Kimi family.
	"kimi-k2.5": {ModelMaxTokens: 32768, ContextWindow: 256000},
	"kimi-k2":   {ModelMaxTokens: 32000, ContextWindow: 128000},

	// MiniMax family.
	"minimax-m2.7": {ModelMaxTokens: 204800, ContextWindow: 204800},
	"minimax-m2.5": {ModelMaxTokens: 204800, ContextWindow: 204800},

	// DeepSeek family (API model names).
	"deepseek-reasoner": {ModelMaxTokens: 64000, ContextWindow: 128000},
	"deepseek-chat":     {ModelMaxTokens: 8000, ContextWindow: 128000},
	"deepseek":          {ModelMaxTokens: 8000, ContextWindow: 128000},

	// Qwen families.
	"qwen3.5": {ModelMaxTokens: 65536, ContextWindow: 1000000},
	"qwen3":   {ModelMaxTokens: 65536, ContextWindow: 262144},
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

// RefreshModelTokenDefaults reapplies auto token defaults after a model change.
// Explicit config overrides are preserved; only values still on the default or
// previous auto-profile values are updated.
func RefreshModelTokenDefaults(cfg *Config, previousModel string) {
	if cfg == nil {
		return
	}

	defaults := DefaultConfig()
	previousProfile, previousProfileMatched := matchModelTokenProfile(previousModel, cfg.ModelProfiles)
	nextProfile, nextProfileMatched := matchModelTokenProfile(cfg.Model.Model, cfg.ModelProfiles)

	if shouldRefreshAutoTokenValue(cfg.Model.MaxTokens, defaults.Model.MaxTokens, previousProfile.ModelMaxTokens, previousProfileMatched) {
		cfg.Model.MaxTokens = defaults.Model.MaxTokens
		if nextProfileMatched && nextProfile.ModelMaxTokens > 0 {
			cfg.Model.MaxTokens = nextProfile.ModelMaxTokens
		}
	}

	if shouldRefreshAutoTokenValue(cfg.Context.Window, defaults.Context.Window, previousProfile.ContextWindow, previousProfileMatched) {
		cfg.Context.Window = defaults.Context.Window
		if nextProfileMatched && nextProfile.ContextWindow > 0 {
			cfg.Context.Window = nextProfile.ContextWindow
		}
	}
}

func shouldRefreshAutoTokenValue(currentValue, defaultValue, previousProfileValue int, previousProfileMatched bool) bool {
	if currentValue == defaultValue {
		return true
	}
	return previousProfileMatched && previousProfileValue > 0 && currentValue == previousProfileValue
}

func matchModelTokenProfile(modelName string, custom map[string]ModelTokenProfile) (ModelTokenProfile, bool) {
	for _, candidate := range modelMatchCandidates(modelName) {
		if profile, ok := matchByLongestPrefix(candidate, custom); ok {
			return profile, true
		}
		if profile, ok := matchByLongestPrefix(candidate, builtinModelTokenProfiles); ok {
			return profile, true
		}
	}

	return ModelTokenProfile{}, false
}

func modelMatchCandidates(modelName string) []string {
	normalizedModel := strings.ToLower(strings.TrimSpace(modelName))
	if normalizedModel == "" {
		return nil
	}

	candidates := []string{normalizedModel}
	if i := strings.LastIndex(normalizedModel, "/"); i >= 0 && i+1 < len(normalizedModel) {
		candidates = append(candidates, normalizedModel[i+1:])
	}
	return candidates
}

func matchByLongestPrefix(model string, profiles map[string]ModelTokenProfile) (ModelTokenProfile, bool) {
	if len(profiles) == 0 {
		return ModelTokenProfile{}, false
	}

	type candidate struct {
		prefix  string
		profile ModelTokenProfile
	}

	candidates := make([]candidate, 0, len(profiles))
	for prefix, profile := range profiles {
		normalized := strings.ToLower(strings.TrimSpace(prefix))
		if normalized != "" {
			candidates = append(candidates, candidate{prefix: normalized, profile: profile})
		}
	}
	sort.Slice(candidates, func(i, j int) bool {
		return len(candidates[i].prefix) > len(candidates[j].prefix)
	})

	for _, candidate := range candidates {
		if strings.HasPrefix(model, candidate.prefix) {
			return candidate.profile, true
		}
	}

	return ModelTokenProfile{}, false
}
