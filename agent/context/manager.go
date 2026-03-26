package context

import (
	"fmt"
	"sync"
	"time"

	"github.com/vigo999/ms-cli/integrations/llm"
)

// ManagerConfig holds the manager configuration.
type ManagerConfig struct {
	ContextWindow       int
	ReserveTokens       int
	CompactionThreshold float64
	MaxHistoryRounds    int

	// 新增配置
	EnableSmartCompact bool            // 启用智能压缩
	CompactStrategy    CompactStrategy // 压缩策略
	EnablePriority     bool            // 启用优先级系统
}

// DefaultManagerConfig 返回默认配置
func DefaultManagerConfig() ManagerConfig {
	return ManagerConfig{
		ContextWindow:       24000,
		ReserveTokens:       4000,
		CompactionThreshold: 0.85,
		MaxHistoryRounds:    10,
		EnableSmartCompact:  true,
		CompactStrategy:     CompactStrategyHybrid,
		EnablePriority:      true,
	}
}

// Manager manages conversation context.
type Manager struct {
	config   ManagerConfig
	mu       sync.RWMutex
	messages []llm.Message
	system   *llm.Message
	usage    TokenUsage

	// 增强组件
	tokenizer *Tokenizer
	compactor *Compactor
	scorer    *PriorityScorer

	// 统计
	stats Stats
}

// TokenUsage represents token usage statistics.
type TokenUsage struct {
	Current       int
	ContextWindow int
	Reserved      int
	Available     int
}

// Stats 上下文统计
type Stats struct {
	MessageCount    int
	ToolCallCount   int
	CompactCount    int
	LastCompactAt   *time.Time
	TotalTokensUsed int
}

// NewManager creates a new context manager.
func NewManager(cfg ManagerConfig) *Manager {
	if cfg.ContextWindow == 0 {
		cfg.ContextWindow = 24000
	}
	if cfg.ReserveTokens == 0 {
		cfg.ReserveTokens = 4000
	}
	if cfg.CompactionThreshold == 0 {
		cfg.CompactionThreshold = 0.85
	}
	if cfg.MaxHistoryRounds == 0 {
		cfg.MaxHistoryRounds = 10
	}

	// 创建压缩器
	compactor := NewCompactor(CompactorConfig{
		Strategy:        cfg.CompactStrategy,
		MaxKeepMessages: cfg.MaxHistoryRounds * 2,
	})

	m := &Manager{
		config:    cfg,
		messages:  make([]llm.Message, 0),
		tokenizer: NewTokenizer(),
		compactor: compactor,
		scorer:    NewPriorityScorer(),
		usage: TokenUsage{
			ContextWindow: cfg.ContextWindow,
			Reserved:      cfg.ReserveTokens,
			Available:     cfg.ContextWindow - cfg.ReserveTokens,
		},
	}

	return m
}

// SetSystemPrompt sets the system prompt.
func (m *Manager) SetSystemPrompt(content string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	msg := llm.NewSystemMessage(content)
	m.system = &msg

	m.recalculateUsage()
}

// GetSystemPrompt returns the system prompt.
func (m *Manager) GetSystemPrompt() *llm.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.system == nil {
		return nil
	}
	msg := *m.system
	return &msg
}

// AddMessage adds a message to the context.
func (m *Manager) AddMessage(msg llm.Message) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// 估算新消息的 Token
	msgTokens := m.tokenizer.EstimateMessage(msg)
	maxUsable := m.config.ContextWindow - m.config.ReserveTokens
	if msgTokens > maxUsable {
		return fmt.Errorf("single message too large for context budget: %d tokens > %d", msgTokens, maxUsable)
	}

	// 先追加，再按真实占用触发后置压缩
	m.messages = append(m.messages, msg)

	// 后置压缩：基于最新上下文做决策，避免仅靠预估触发
	if m.shouldCompactLocked(0) {
		if err := m.compactLocked(); err != nil {
			return fmt.Errorf("compact context: %w", err)
		}
	}

	// 紧急压缩：后置压缩后仍超上下文窗口时启用更激进策略
	if m.totalTokensLocked() > m.maxUsableTokensLocked() {
		if err := m.emergencyCompactLocked(); err != nil {
			return fmt.Errorf("emergency compact: %w", err)
		}
	}

	m.recalculateUsage()
	m.stats.MessageCount++
	if msg.Role == "tool" {
		m.stats.ToolCallCount++
	}

	return nil
}

// AddToolResult adds a tool result to the context.
func (m *Manager) AddToolResult(callID, content string) error {
	return m.AddMessage(llm.NewToolMessage(callID, content))
}

// GetMessages returns all messages including system prompt.
func (m *Manager) GetMessages() []llm.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]llm.Message, 0, len(m.messages)+1)
	if m.system != nil {
		result = append(result, *m.system)
	}
	result = append(result, m.messages...)
	return result
}

// GetNonSystemMessages returns only non-system messages.
func (m *Manager) GetNonSystemMessages() []llm.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]llm.Message, len(m.messages))
	copy(result, m.messages)
	return result
}

// SetNonSystemMessages replaces all non-system messages.
func (m *Manager) SetNonSystemMessages(msgs []llm.Message) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.messages = make([]llm.Message, len(msgs))
	copy(m.messages, msgs)

	m.stats.MessageCount = len(m.messages)
	m.stats.ToolCallCount = 0
	for _, msg := range m.messages {
		if msg.Role == "tool" {
			m.stats.ToolCallCount++
		}
	}

	m.recalculateUsage()
}

// Clear clears all messages except system prompt.
func (m *Manager) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.messages = make([]llm.Message, 0)
	m.recalculateUsage()
}

// Compact manually triggers context compaction.
func (m *Manager) Compact() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.compactLocked()
}

// TokenUsage returns current token usage.
func (m *Manager) TokenUsage() TokenUsage {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.usage
}

// SetContextWindowLimits updates the runtime context window limits.
func (m *Manager) SetContextWindowLimits(contextWindow, reserveTokens int) error {
	if contextWindow <= 0 {
		return fmt.Errorf("context window must be positive")
	}
	if reserveTokens < 0 {
		return fmt.Errorf("reserve tokens must be non-negative")
	}
	if reserveTokens >= contextWindow {
		return fmt.Errorf("reserve tokens must be less than context window")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.config.ContextWindow = contextWindow
	m.config.ReserveTokens = reserveTokens
	m.recalculateUsage()

	return nil
}

// EstimateTokens estimates token count for messages.
func (m *Manager) EstimateTokens(msgs []llm.Message) int {
	return m.tokenizer.EstimateMessages(msgs)
}

// IsWithinBudget checks if adding a message would exceed the context window.
func (m *Manager) IsWithinBudget(msg llm.Message) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	estimated := m.totalTokensLocked() + m.tokenizer.EstimateMessage(msg)
	return estimated <= m.maxUsableTokensLocked()
}

// GetStats returns context statistics.
func (m *Manager) GetStats() map[string]any {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return map[string]any{
		"total_messages":    len(m.messages),
		"has_system_prompt": m.system != nil,
		"token_usage":       m.usage,
		"context_window":    m.config.ContextWindow,
		"compact_count":     m.stats.CompactCount,
		"tool_call_count":   m.stats.ToolCallCount,
		"last_compact_at":   m.stats.LastCompactAt,
	}
}

// GetDetailedStats returns detailed statistics.
func (m *Manager) GetDetailedStats() map[string]any {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := map[string]any{
		"messages": map[string]any{
			"total":     len(m.messages),
			"user":      m.countByRole("user"),
			"assistant": m.countByRole("assistant"),
			"tool":      m.countByRole("tool"),
		},
		"tokens": map[string]any{
			"current":        m.usage.Current,
			"context_window": m.usage.ContextWindow,
			"reserved":       m.usage.Reserved,
			"available":      m.usage.Available,
		},
		"stats": m.stats,
	}

	return stats
}

// shouldCompactLocked checks if compaction is needed (must hold lock).
func (m *Manager) shouldCompactLocked(additionalTokens int) bool {
	threshold := m.compactionThresholdPercentLocked()
	estimatedTokens := m.totalTokensLocked() + additionalTokens
	return float64(estimatedTokens) >= float64(m.config.ContextWindow)*(threshold/100.0)
}

// compactLocked compacts the context (must hold lock).
func (m *Manager) compactLocked() error {
	if len(m.messages) <= m.config.MaxHistoryRounds {
		return nil
	}

	// 使用智能压缩
	if m.config.EnableSmartCompact && m.compactor != nil {
		compacted, result := m.compactor.Compact(m.messages, m.system)
		m.messages = compacted
		m.stats.CompactCount++
		now := time.Now()
		m.stats.LastCompactAt = &now
		_ = result // 可以在日志中记录
	} else {
		// 简单压缩
		keepCount := m.config.MaxHistoryRounds * 2
		if keepCount < len(m.messages) {
			kept := keepRecentMessages(m.messages, keepCount)
			removed := len(m.messages) - len(kept)
			summary := fmt.Sprintf("[Earlier conversation: %d messages summarized]", removed)
			summaryMsg := llm.NewSystemMessage(summary)
			m.messages = append([]llm.Message{summaryMsg}, kept...)
			m.stats.CompactCount++
			now := time.Now()
			m.stats.LastCompactAt = &now
		}
	}

	m.recalculateUsage()
	return nil
}

// emergencyCompactLocked performs emergency compaction when budget is exceeded.
func (m *Manager) emergencyCompactLocked() error {
	// 紧急压缩：保留更少消息
	keepCount := m.config.MaxHistoryRounds
	if keepCount < 4 {
		keepCount = 4
	}

	if len(m.messages) > keepCount {
		if m.config.EnableSmartCompact {
			priorityCompactor := NewCompactor(CompactorConfig{
				Strategy:        CompactStrategyPriority,
				MaxKeepMessages: keepCount,
			})
			compacted, _ := priorityCompactor.Compact(m.messages, m.system)
			m.messages = compacted
		} else {
			m.messages = keepRecentMessages(m.messages, keepCount)
		}
		m.stats.CompactCount++
		now := time.Now()
		m.stats.LastCompactAt = &now
	}

	m.recalculateUsage()
	return nil
}

func (m *Manager) compactionThresholdPercentLocked() float64 {
	threshold := m.config.CompactionThreshold
	switch {
	case threshold <= 0:
		return 85.0
	case threshold <= 1:
		return threshold * 100
	default:
		// 兼容旧配置：允许直接填写百分比（0-100）
		if threshold > 100 {
			return 100
		}
		return threshold
	}
}

// recalculateUsage recalculates token usage (must hold lock).
func (m *Manager) recalculateUsage() {
	total := m.totalTokensLocked()

	m.usage = TokenUsage{
		Current:       total,
		ContextWindow: m.config.ContextWindow,
		Reserved:      m.config.ReserveTokens,
		Available:     m.config.ContextWindow - total - m.config.ReserveTokens,
	}

	m.stats.TotalTokensUsed = total
}

func (m *Manager) totalTokensLocked() int {
	total := m.tokenizer.EstimateMessages(m.messages)
	if m.system != nil {
		total += m.tokenizer.EstimateMessage(*m.system)
	}
	return total
}

func (m *Manager) maxUsableTokensLocked() int {
	return m.config.ContextWindow - m.config.ReserveTokens
}

// countByRole counts messages by role (must hold lock).
func (m *Manager) countByRole(role string) int {
	count := 0
	for _, msg := range m.messages {
		if msg.Role == role {
			count++
		}
	}
	return count
}

// SetCompactStrategy sets the compaction strategy.
func (m *Manager) SetCompactStrategy(s CompactStrategy) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.config.CompactStrategy = s
	if m.compactor != nil {
		m.compactor.SetStrategy(s)
	}
}

// GetMessagePriority returns the priority of a message.
func (m *Manager) GetMessagePriority(index int) Priority {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if index < 0 || index >= len(m.messages) {
		return PriorityLow
	}

	return m.scorer.ScoreMessage(m.messages[index], index, len(m.messages))
}

// TruncateTo truncates messages to the specified count (keeping the most recent).
func (m *Manager) TruncateTo(count int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if count < 0 {
		count = 0
	}
	if count >= len(m.messages) {
		return
	}

	m.messages = keepRecentMessages(m.messages, count)
	m.recalculateUsage()
}
