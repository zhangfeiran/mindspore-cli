package app

import (
	"strings"
	"time"

	agentctx "github.com/mindspore-lab/mindspore-cli/agent/context"
	"github.com/mindspore-lab/mindspore-cli/agent/session"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
)

func providerUsageSnapshotFromDetails(details agentctx.TokenUsageDetails) *session.UsageSnapshot {
	if details.Source != agentctx.TokenUsageSourceProvider || details.ProviderSnapshotTokens <= 0 {
		return nil
	}

	return &session.UsageSnapshot{
		Provider:   details.Provider,
		TokenScope: string(details.ProviderTokenScope),
		Tokens:     details.ProviderSnapshotTokens,
		LocalDelta: details.LocalDelta,
		Usage:      ptrProviderUsage(details.ProviderUsage),
	}
}

func compressionSnapshotFromManager(cm *agentctx.Manager) *session.CompressionState {
	if cm == nil {
		return nil
	}
	return compressionSnapshotFromState(cm.ExportCompressionState())
}

func compressionSnapshotFromState(state *agentctx.CompressionState) *session.CompressionState {
	if state == nil {
		return nil
	}

	result := &session.CompressionState{
		LastAssistantAt: cloneTimePtr(state.LastAssistantAt),
		ToolArtifacts:   make([]session.ToolArtifact, 0, len(state.ToolArtifacts)),
	}
	for _, artifact := range state.ToolArtifacts {
		result.ToolArtifacts = append(result.ToolArtifacts, session.ToolArtifact{
			ToolCallID:   artifact.ToolCallID,
			ToolName:     artifact.ToolName,
			Path:         artifact.Path,
			OriginalSize: artifact.OriginalSize,
			State:        artifact.State,
			CreatedAt:    artifact.CreatedAt,
		})
	}
	return result
}

func persistDebugCompactionSnapshot(s *session.Session, label, systemPrompt string, messages []llm.Message, details agentctx.TokenUsageDetails, compression *agentctx.CompressionState) error {
	if s == nil {
		return nil
	}
	_, err := s.SaveDebugSnapshotWithCompression(
		label,
		systemPrompt,
		messages,
		providerUsageSnapshotFromDetails(details),
		compressionSnapshotFromState(compression),
	)
	return err
}

func restoreCompressionSnapshot(cm *agentctx.Manager, state *session.CompressionState) {
	if cm == nil {
		return
	}
	if state == nil {
		cm.RestoreCompressionState(nil)
		return
	}

	result := &agentctx.CompressionState{
		LastAssistantAt: cloneTimePtr(state.LastAssistantAt),
		ToolArtifacts:   make([]agentctx.ToolArtifact, 0, len(state.ToolArtifacts)),
	}
	for _, artifact := range state.ToolArtifacts {
		result.ToolArtifacts = append(result.ToolArtifacts, agentctx.ToolArtifact{
			ToolCallID:   artifact.ToolCallID,
			ToolName:     artifact.ToolName,
			Path:         artifact.Path,
			OriginalSize: artifact.OriginalSize,
			State:        artifact.State,
			CreatedAt:    artifact.CreatedAt,
		})
	}
	cm.RestoreCompressionState(result)
}

func restoreProviderUsageSnapshot(cm *agentctx.Manager, usage *session.UsageSnapshot) {
	if cm == nil || usage == nil || usage.Tokens <= 0 {
		return
	}

	scope := agentctx.ProviderTokenScope(strings.TrimSpace(usage.TokenScope))
	if scope != agentctx.ProviderTokenScopeTotal {
		scope = agentctx.ProviderTokenScopePrompt
	}

	cm.RestoreProviderUsageSnapshot(agentctx.ProviderUsageSnapshot{
		Provider:   usage.Provider,
		TokenScope: scope,
		Tokens:     usage.Tokens,
		LocalDelta: usage.LocalDelta,
		Usage:      derefProviderUsage(usage.Usage),
	})
}

func ptrProviderUsage(usage llm.Usage) *llm.Usage {
	if usage.IsZero() {
		return nil
	}
	copy := usage.Clone()
	return &copy
}

func derefProviderUsage(usage *llm.Usage) llm.Usage {
	if usage == nil {
		return llm.Usage{}
	}
	return usage.Clone()
}

func cloneTimePtr(v *time.Time) *time.Time {
	if v == nil {
		return nil
	}
	copy := *v
	return &copy
}
