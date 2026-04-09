package context

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
)

const (
	persistedToolResultOpenTag  = "<persisted-tool-result>"
	persistedToolResultCloseTag = "</persisted-tool-result>"
	clearedToolResultMessage    = "[Old tool result content cleared]"
	sessionNotesHeader          = "[Session Notes]"

	toolArtifactStatePreviewed = "previewed"
	toolArtifactStateCleared   = "cleared"
)

var microcompactToolNames = map[string]struct{}{
	"read":  {},
	"shell": {},
	"grep":  {},
	"glob":  {},
}

type ToolArtifact struct {
	ToolCallID   string
	ToolName     string
	Path         string
	OriginalSize int
	State        string
	CreatedAt    time.Time
}

type SessionNotes struct {
	Content          string
	UpdatedAt        time.Time
	SourceTokenCount int
}

type CompressionState struct {
	LastAssistantAt *time.Time
	ToolArtifacts   []ToolArtifact
	SessionNotes    *SessionNotes
}

type PrepareResult struct {
	Changed              bool
	AutoCompacted        bool
	ToolResultsPersisted int
	ToolResultsCleared   int
	NotesUpdated         bool
	BeforeTokens         int
	AfterTokens          int
}

func (m *Manager) SetToolResultArtifactDir(dir string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.toolResultDir = strings.TrimSpace(dir)
}

func (m *Manager) ExportCompressionState() *CompressionState {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return cloneCompressionStateLocked(m)
}

func (m *Manager) RestoreCompressionState(state *CompressionState) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.toolArtifacts = make(map[string]ToolArtifact)
	m.sessionNotes = nil
	m.lastAssistantAt = nil
	if state == nil {
		return
	}

	m.lastAssistantAt = cloneTimePtrLocked(state.LastAssistantAt)
	for _, artifact := range state.ToolArtifacts {
		if id := strings.TrimSpace(artifact.ToolCallID); id != "" {
			m.toolArtifacts[id] = artifact
			if name := strings.TrimSpace(artifact.ToolName); name != "" {
				m.toolCallNames[id] = name
			}
		}
	}
	m.sessionNotes = cloneSessionNotesLocked(state.SessionNotes)
}

func (m *Manager) PrepareForRequest(now time.Time) (PrepareResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.prepareLocked(now, false)
}

func (m *Manager) prepareLocked(now time.Time, forceCompact bool) (PrepareResult, error) {
	if now.IsZero() {
		now = time.Now()
	}

	beforeTokens := m.currentTokensLocked()
	baseMessages, _ := stripSessionNotesMessages(m.messages)
	working := cloneMessages(baseMessages)

	persisted, next, err := m.applyAggregateToolResultBudgetLocked(working, now)
	if err != nil {
		return PrepareResult{}, err
	}
	working = next

	cleared := 0
	if next, count := m.applyTimeBasedMicrocompactLocked(working, now); count > 0 {
		working = next
		cleared = count
	}

	preparedTokens := estimateMessagesWithSystem(m.assembleMessagesWithNotesLocked(working), m.system, m.tokenizer)
	notesUpdated := m.maybeRefreshSessionNotesLocked(working, preparedTokens, now)

	autoCompacted := false
	if forceCompact || m.shouldAutoCompactPreparedLocked(working) {
		compacted, changed := m.autoCompactPreparedLocked(working, now, forceCompact)
		if changed {
			working = compacted
			autoCompacted = true
		}
	}

	finalMessages := m.assembleMessagesWithNotesLocked(working)
	changed := !messagesEqual(finalMessages, m.messages) || notesUpdated || persisted > 0 || cleared > 0
	if changed {
		m.messages = finalMessages
		m.stats.MessageCount = len(m.messages)
		m.stats.ToolCallCount = 0
		for _, msg := range m.messages {
			if msg.Role == "tool" {
				m.stats.ToolCallCount++
			}
		}
		m.clearProviderTokenUsageLocked()
		m.recalculateUsage()
	}

	return PrepareResult{
		Changed:              changed,
		AutoCompacted:        autoCompacted,
		ToolResultsPersisted: persisted,
		ToolResultsCleared:   cleared,
		NotesUpdated:         notesUpdated,
		BeforeTokens:         beforeTokens,
		AfterTokens:          m.currentTokensLocked(),
	}, nil
}

func (m *Manager) prepareToolResultMessageLocked(toolName, callID, content string) (llm.Message, error) {
	msg := llm.NewToolMessage(callID, content)
	maxUsable := m.maxUsableTokensLocked()
	if maxUsable <= 0 {
		return msg, fmt.Errorf("invalid context budget")
	}

	if strings.TrimSpace(toolName) == "" {
		toolName = m.toolCallNames[strings.TrimSpace(callID)]
	}
	msgTokens := m.tokenizer.EstimateMessage(msg)
	if msgTokens <= maxUsable && (m.config.ToolResultMaxChars <= 0 || len(content) <= m.config.ToolResultMaxChars) {
		if id := strings.TrimSpace(callID); id != "" && strings.TrimSpace(toolName) != "" {
			m.toolCallNames[id] = strings.TrimSpace(toolName)
		}
		return msg, nil
	}

	artifact, err := m.persistToolResultLocked(strings.TrimSpace(toolName), callID, content, time.Now())
	if err != nil {
		if msgTokens <= maxUsable {
			return msg, nil
		}
		return llm.Message{}, err
	}
	preview, err := m.buildPersistedToolResultPreviewLocked(callID, artifact.Path, content, maxUsable)
	if err != nil {
		return llm.Message{}, err
	}

	if artifact.ToolCallID != "" {
		m.toolArtifacts[artifact.ToolCallID] = artifact
	}
	return llm.NewToolMessage(callID, preview), nil
}

func (m *Manager) addPreparedMessageLocked(msg llm.Message) error {
	msgTokens := m.tokenizer.EstimateMessage(msg)
	maxUsable := m.config.ContextWindow - m.config.ReserveTokens
	if msgTokens > maxUsable {
		return fmt.Errorf("single message too large for context budget: %d tokens > %d", msgTokens, maxUsable)
	}

	m.messages = append(m.messages, msg)

	if m.shouldCompactLocked(0) {
		if err := m.compactLocked(); err != nil {
			return fmt.Errorf("compact context: %w", err)
		}
	}

	m.recalculateUsage()
	m.stats.MessageCount++
	if msg.Role == "assistant" {
		now := time.Now()
		m.lastAssistantAt = &now
		for _, tc := range msg.ToolCalls {
			if id := strings.TrimSpace(tc.ID); id != "" {
				m.toolCallNames[id] = strings.TrimSpace(tc.Function.Name)
			}
		}
	}
	if msg.Role == "tool" {
		m.stats.ToolCallCount++
	}
	return nil
}

func (m *Manager) shouldAutoCompactPreparedLocked(messages []llm.Message) bool {
	threshold := m.maxUsableTokensLocked() - m.config.AutoCompactBufferTokens
	if threshold <= 0 {
		threshold = m.maxUsableTokensLocked()
	}
	current := estimateMessagesWithSystem(m.assembleMessagesWithNotesLocked(messages), m.system, m.tokenizer)
	return current >= threshold
}

func (m *Manager) autoCompactPreparedLocked(messages []llm.Message, now time.Time, force bool) ([]llm.Message, bool) {
	if len(messages) == 0 {
		return m.assembleMessagesWithNotesLocked(messages), false
	}

	groups := groupMessages(messages)
	pinned := pinnedMessageGroups(groups)
	pinnedSet := messageGroupSet(pinned)
	selected := append([]messageGroup{}, pinned...)
	usedTokens := countTokensInGroups(selected, m.tokenizer)
	textMessages := countTextMessagesInGroups(selected)

	tailMax := m.config.NotesMaxTailTokens
	if tailMax <= 0 {
		tailMax = m.maxUsableTokensLocked()
	}
	defaultTarget := m.maxUsableTokensLocked() / 2
	if defaultTarget > 0 && defaultTarget < tailMax {
		tailMax = defaultTarget
	}
	if force {
		manualTarget := m.maxUsableTokensLocked() / 2
		if manualTarget > 0 && manualTarget < tailMax {
			tailMax = manualTarget
		}
	}
	minTail := m.config.NotesMinTailTokens
	if minTail <= 0 {
		minTail = tailMax
	}
	if minTail > tailMax {
		minTail = tailMax
	}
	minMessages := m.config.NotesMinMessages
	if minMessages <= 0 {
		minMessages = 1
	}

	for i := len(groups) - 1; i >= 0; i-- {
		group := groups[i]
		if _, ok := pinnedSet[group.Start]; ok {
			continue
		}
		groupTokens := estimateGroupTokens(group, m.tokenizer)
		nextTextMessages := textMessages + countTextMessages(group.Messages)
		if usedTokens >= tailMax && textMessages >= minMessages && usedTokens >= minTail {
			break
		}
		if usedTokens+groupTokens > tailMax && textMessages >= minMessages && usedTokens >= minTail {
			continue
		}
		selected = append(selected, group)
		usedTokens += groupTokens
		textMessages = nextTextMessages
	}

	sortMessageGroupsByStart(selected)
	compacted := flattenMessageGroups(selected)
	finalMessages := m.assembleMessagesWithNotesLocked(compacted)
	maxUsable := m.maxUsableTokensLocked()
	if defaultTarget > 0 && defaultTarget < maxUsable {
		maxUsable = defaultTarget
	}
	if force {
		manualTarget := maxUsable / 2
		if manualTarget > 0 {
			maxUsable = manualTarget
		}
	}
	if estimateMessagesWithSystem(finalMessages, m.system, m.tokenizer) > maxUsable {
		budget := maxUsable
		if m.sessionNotes != nil {
			budget -= m.tokenizer.EstimateMessage(llm.NewSystemMessage(m.sessionNotes.Content))
			if budget < 0 {
				budget = 0
			}
		}
		compacted = keepRecentMessagesByTokens(compacted, budget, m.tokenizer)
		finalMessages = m.assembleMessagesWithNotesLocked(compacted)
	}

	if estimateMessagesWithSystem(finalMessages, m.system, m.tokenizer) >= estimateMessagesWithSystem(m.assembleMessagesWithNotesLocked(messages), m.system, m.tokenizer) {
		fallback := keepRecentMessagesWithinTotalBudget(messages, m.system, maxUsable, m.tokenizer)
		finalMessages = m.assembleMessagesWithNotesLocked(fallback)
		if estimateMessagesWithSystem(finalMessages, m.system, m.tokenizer) >= estimateMessagesWithSystem(m.assembleMessagesWithNotesLocked(messages), m.system, m.tokenizer) {
			return messages, false
		}
		compacted = fallback
	}

	m.stats.CompactCount++
	compactAt := now
	m.stats.LastCompactAt = &compactAt
	return compacted, true
}

func (m *Manager) applyAggregateToolResultBudgetLocked(messages []llm.Message, now time.Time) (int, []llm.Message, error) {
	limit := m.config.ToolResultBatchChars
	if limit <= 0 {
		return 0, messages, nil
	}

	type candidate struct {
		index    int
		callID   string
		toolName string
		size     int
		content  string
	}

	replaced := 0
	result := cloneMessages(messages)
	for i := 0; i < len(result); i++ {
		msg := result[i]
		if msg.Role != "assistant" || len(msg.ToolCalls) == 0 {
			continue
		}

		candidates := make([]candidate, 0)
		totalSize := 0
		for j := i + 1; j < len(result) && result[j].Role == "tool"; j++ {
			toolMsg := result[j]
			content := toolMsg.Content
			if content == "" || content == clearedToolResultMessage || isPersistedToolResultMessage(content) {
				continue
			}
			callID := strings.TrimSpace(toolMsg.ToolCallID)
			toolName := m.toolCallNames[callID]
			if toolName == "load_skill" {
				continue
			}
			size := len(content)
			totalSize += size
			candidates = append(candidates, candidate{
				index:    j,
				callID:   callID,
				toolName: toolName,
				size:     size,
				content:  content,
			})
		}

		selected := make([]candidate, 0)
		for _, c := range candidates {
			if m.config.ToolResultMaxChars > 0 && c.size > m.config.ToolResultMaxChars {
				selected = append(selected, c)
			}
		}
		if totalSize > limit {
			sorted := append([]candidate(nil), candidates...)
			sort.Slice(sorted, func(a, b int) bool {
				return sorted[a].size > sorted[b].size
			})
			currentSize := totalSize
			selectedSet := make(map[int]struct{}, len(selected))
			for _, c := range selected {
				selectedSet[c.index] = struct{}{}
				currentSize -= c.size
			}
			for _, c := range sorted {
				if currentSize <= limit {
					break
				}
				if _, ok := selectedSet[c.index]; ok {
					continue
				}
				selected = append(selected, c)
				selectedSet[c.index] = struct{}{}
				currentSize -= c.size
			}
		}

		for _, c := range selected {
			artifact, err := m.persistToolResultLocked(c.toolName, c.callID, c.content, now)
			if err != nil {
				return replaced, messages, err
			}
			preview, err := m.buildPersistedToolResultPreviewLocked(c.callID, artifact.Path, c.content, m.maxUsableTokensLocked())
			if err != nil {
				return replaced, messages, err
			}
			result[c.index].Content = preview
			if artifact.ToolCallID != "" {
				m.toolArtifacts[artifact.ToolCallID] = artifact
			}
			replaced++
		}
	}

	return replaced, result, nil
}

func (m *Manager) applyTimeBasedMicrocompactLocked(messages []llm.Message, now time.Time) ([]llm.Message, int) {
	if m.lastAssistantAt == nil || m.config.MicrocompactIdleMinutes <= 0 {
		return messages, 0
	}
	if now.Sub(*m.lastAssistantAt) < time.Duration(m.config.MicrocompactIdleMinutes)*time.Minute {
		return messages, 0
	}

	type toolMessage struct {
		index    int
		callID   string
		toolName string
	}

	compactable := make([]toolMessage, 0)
	for i, msg := range messages {
		if msg.Role != "tool" || msg.Content == clearedToolResultMessage {
			continue
		}
		callID := strings.TrimSpace(msg.ToolCallID)
		toolName := m.toolCallNames[callID]
		if artifact, ok := m.toolArtifacts[callID]; ok && strings.TrimSpace(toolName) == "" {
			toolName = artifact.ToolName
		}
		if _, ok := microcompactToolNames[toolName]; !ok {
			continue
		}
		compactable = append(compactable, toolMessage{
			index:    i,
			callID:   callID,
			toolName: toolName,
		})
	}

	keepRecent := m.config.MicrocompactKeepRecent
	if keepRecent <= 0 {
		keepRecent = 1
	}
	if len(compactable) <= keepRecent {
		return messages, 0
	}

	keepSet := make(map[int]struct{}, keepRecent)
	for _, entry := range compactable[len(compactable)-keepRecent:] {
		keepSet[entry.index] = struct{}{}
	}

	result := cloneMessages(messages)
	cleared := 0
	for _, entry := range compactable {
		if _, ok := keepSet[entry.index]; ok {
			continue
		}
		if result[entry.index].Content == clearedToolResultMessage {
			continue
		}
		result[entry.index].Content = clearedToolResultMessage
		if artifact, ok := m.toolArtifacts[entry.callID]; ok {
			artifact.State = toolArtifactStateCleared
			m.toolArtifacts[entry.callID] = artifact
		}
		cleared++
	}

	return result, cleared
}

func (m *Manager) maybeRefreshSessionNotesLocked(messages []llm.Message, sourceTokens int, now time.Time) bool {
	if !m.config.NotesEnabled || sourceTokens < m.config.NotesInitTokens {
		return false
	}
	if m.sessionNotes != nil && sourceTokens-m.sessionNotes.SourceTokenCount < m.config.NotesUpdateTokens {
		return false
	}

	notes := buildSessionNotes(m.sessionNotes, messages, now, sourceTokens)
	if m.sessionNotes != nil && m.sessionNotes.Content == notes.Content && m.sessionNotes.SourceTokenCount == notes.SourceTokenCount {
		return false
	}
	m.sessionNotes = notes
	return true
}

func (m *Manager) assembleMessagesWithNotesLocked(messages []llm.Message) []llm.Message {
	if m.sessionNotes == nil || strings.TrimSpace(m.sessionNotes.Content) == "" {
		return cloneMessages(messages)
	}

	result := make([]llm.Message, 0, len(messages)+1)
	result = append(result, llm.NewSystemMessage(m.sessionNotes.Content))
	result = append(result, messages...)
	return result
}

func (m *Manager) persistToolResultLocked(toolName, callID, content string, now time.Time) (ToolArtifact, error) {
	dir := strings.TrimSpace(m.toolResultDir)
	if dir == "" {
		return ToolArtifact{}, fmt.Errorf("tool result artifact directory is not configured")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		return ToolArtifact{}, fmt.Errorf("create tool result dir: %w", err)
	}

	filename := sanitizeToolResultFilename(callID)
	path := filepath.Join(dir, filename+".txt")
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		return ToolArtifact{}, fmt.Errorf("write tool result artifact: %w", err)
	}

	artifact := ToolArtifact{
		ToolCallID:   strings.TrimSpace(callID),
		ToolName:     strings.TrimSpace(toolName),
		Path:         path,
		OriginalSize: len(content),
		State:        toolArtifactStatePreviewed,
		CreatedAt:    now,
	}
	return artifact, nil
}

func buildToolResultPreview(path, content string, previewBytes int) string {
	var sb strings.Builder
	sb.WriteString(persistedToolResultOpenTag)
	sb.WriteString("\n")
	sb.WriteString(fmt.Sprintf("Output too large (%d bytes). Full output saved to: %s\n", len(content), path))
	if previewBytes > 0 {
		preview, hasMore := truncateAtLineBoundary(content, previewBytes)
		sb.WriteString("\n")
		sb.WriteString(fmt.Sprintf("Preview (first %d bytes):\n", previewBytes))
		sb.WriteString(preview)
		if hasMore {
			sb.WriteString("\n...\n")
		} else {
			sb.WriteString("\n")
		}
	} else {
		sb.WriteString("\n")
	}
	sb.WriteString(persistedToolResultCloseTag)
	return sb.String()
}

func (m *Manager) buildPersistedToolResultPreviewLocked(callID, path, content string, maxTokens int) (string, error) {
	if maxTokens <= 0 {
		maxTokens = 1
	}

	candidates := make([]int, 0, 8)
	seen := make(map[int]struct{})
	addCandidate := func(size int) {
		if size < 0 {
			size = 0
		}
		if _, ok := seen[size]; ok {
			return
		}
		seen[size] = struct{}{}
		candidates = append(candidates, size)
	}

	previewBytes := m.config.ToolResultPreviewBytes
	if previewBytes < 0 {
		previewBytes = 0
	}
	addCandidate(previewBytes)
	for size := previewBytes; size > 1; size /= 2 {
		addCandidate(size / 2)
	}
	addCandidate(0)

	lastTokens := 0
	for _, size := range candidates {
		preview := buildToolResultPreview(path, content, size)
		tokens := m.tokenizer.EstimateMessage(llm.NewToolMessage(callID, preview))
		lastTokens = tokens
		if tokens <= maxTokens {
			return preview, nil
		}
	}

	return "", fmt.Errorf("persisted tool result preview exceeds context budget: %d tokens > %d", lastTokens, maxTokens)
}

func buildSessionNotes(previous *SessionNotes, messages []llm.Message, now time.Time, sourceTokens int) *SessionNotes {
	worklog := make([]string, 0, 8)
	openProblems := make([]string, 0, 6)
	corrections := make([]string, 0, 6)
	currentState := ""

	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		text := summarizeMessageForNotes(msg)
		if text == "" {
			continue
		}
		if currentState == "" && (msg.Role == "assistant" || msg.Role == "user") {
			currentState = text
		}
		if len(worklog) < 8 {
			worklog = append([]string{text}, worklog...)
		}
		lower := strings.ToLower(text)
		if looksLikeProblem(lower) && len(openProblems) < 6 {
			openProblems = appendUniqueLine(openProblems, text)
		}
		if looksLikeCorrection(lower) && len(corrections) < 6 {
			corrections = appendUniqueLine(corrections, text)
		}
	}

	if currentState == "" {
		currentState = "(no recent user or assistant state)"
	}
	if len(worklog) == 0 {
		worklog = []string{"(no recent worklog)"}
	}
	if len(openProblems) == 0 {
		openProblems = []string{"(no open problems captured)"}
	}
	if len(corrections) == 0 {
		corrections = []string{"(no corrections captured)"}
	}

	priorSummary := "(none)"
	if previous != nil && strings.TrimSpace(previous.Content) != "" {
		priorSummary = trimForNotes(previous.Content, 1200)
	}

	var sb strings.Builder
	sb.WriteString(sessionNotesHeader)
	sb.WriteString("\n\nPrior Summary:\n")
	sb.WriteString(priorSummary)
	sb.WriteString("\n\nCurrent State:\n")
	sb.WriteString(currentState)
	sb.WriteString("\n\nOpen Problems:\n")
	for _, line := range openProblems {
		sb.WriteString("- ")
		sb.WriteString(line)
		sb.WriteString("\n")
	}
	sb.WriteString("\nErrors & Corrections:\n")
	for _, line := range corrections {
		sb.WriteString("- ")
		sb.WriteString(line)
		sb.WriteString("\n")
	}
	sb.WriteString("\nWorklog:\n")
	for _, line := range worklog {
		sb.WriteString("- ")
		sb.WriteString(line)
		sb.WriteString("\n")
	}

	return &SessionNotes{
		Content:          strings.TrimSpace(sb.String()),
		UpdatedAt:        now,
		SourceTokenCount: sourceTokens,
	}
}

func summarizeMessageForNotes(msg llm.Message) string {
	switch msg.Role {
	case "user":
		return "user: " + trimForNotes(msg.Content, 160)
	case "assistant":
		parts := make([]string, 0, 1+len(msg.ToolCalls))
		if text := strings.TrimSpace(msg.Content); text != "" {
			parts = append(parts, trimForNotes(text, 160))
		}
		for _, tc := range msg.ToolCalls {
			if name := strings.TrimSpace(tc.Function.Name); name != "" {
				parts = append(parts, "called "+name)
			}
		}
		if len(parts) == 0 {
			return ""
		}
		return "assistant: " + strings.Join(parts, "; ")
	case "tool":
		content := strings.TrimSpace(msg.Content)
		if content == "" {
			return ""
		}
		if content == clearedToolResultMessage {
			return "tool: old tool result content cleared"
		}
		return "tool: " + trimForNotes(content, 160)
	default:
		return ""
	}
}

func trimForNotes(text string, maxRunes int) string {
	text = strings.Join(strings.Fields(strings.TrimSpace(text)), " ")
	if text == "" {
		return ""
	}
	runes := []rune(text)
	if maxRunes <= 0 || len(runes) <= maxRunes {
		return text
	}
	if maxRunes <= 3 {
		return string(runes[:maxRunes])
	}
	return string(runes[:maxRunes-3]) + "..."
}

func appendUniqueLine(lines []string, line string) []string {
	for _, existing := range lines {
		if existing == line {
			return lines
		}
	}
	return append(lines, line)
}

func looksLikeProblem(lower string) bool {
	return strings.Contains(lower, "error") ||
		strings.Contains(lower, "fail") ||
		strings.Contains(lower, "denied") ||
		strings.Contains(lower, "missing") ||
		strings.Contains(lower, "not found")
}

func looksLikeCorrection(lower string) bool {
	return strings.Contains(lower, "fixed") ||
		strings.Contains(lower, "updated") ||
		strings.Contains(lower, "retry") ||
		strings.Contains(lower, "resolved")
}

func countTextMessages(group []llm.Message) int {
	total := 0
	for _, msg := range group {
		if msg.Role == "user" || msg.Role == "assistant" {
			if strings.TrimSpace(msg.Content) != "" {
				total++
			}
		}
	}
	return total
}

func countTextMessagesInGroups(groups []messageGroup) int {
	total := 0
	for _, group := range groups {
		total += countTextMessages(group.Messages)
	}
	return total
}

func stripSessionNotesMessages(messages []llm.Message) ([]llm.Message, bool) {
	result := make([]llm.Message, 0, len(messages))
	removed := false
	for _, msg := range messages {
		if isSessionNotesMessage(msg) {
			removed = true
			continue
		}
		result = append(result, msg)
	}
	return result, removed
}

func isSessionNotesMessage(msg llm.Message) bool {
	return msg.Role == "system" && strings.HasPrefix(strings.TrimSpace(msg.Content), sessionNotesHeader)
}

func isPersistedToolResultMessage(content string) bool {
	return strings.HasPrefix(content, persistedToolResultOpenTag)
}

func sanitizeToolResultFilename(callID string) string {
	callID = strings.TrimSpace(callID)
	if callID == "" {
		callID = "tool_result"
	}
	var b strings.Builder
	for _, r := range callID {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '-' || r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}

func truncateAtLineBoundary(content string, maxBytes int) (string, bool) {
	if len(content) <= maxBytes {
		return content, false
	}
	cut := maxBytes
	if idx := strings.LastIndex(content[:maxBytes], "\n"); idx >= maxBytes/2 {
		cut = idx
	}
	return content[:cut], true
}

func cloneMessages(messages []llm.Message) []llm.Message {
	if len(messages) == 0 {
		return nil
	}
	result := make([]llm.Message, len(messages))
	copy(result, messages)
	return result
}

func messagesEqual(a, b []llm.Message) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Role != b[i].Role || a[i].Content != b[i].Content || a[i].ToolCallID != b[i].ToolCallID {
			return false
		}
		if len(a[i].ToolCalls) != len(b[i].ToolCalls) {
			return false
		}
		for j := range a[i].ToolCalls {
			if a[i].ToolCalls[j].ID != b[i].ToolCalls[j].ID ||
				a[i].ToolCalls[j].Function.Name != b[i].ToolCalls[j].Function.Name ||
				string(a[i].ToolCalls[j].Function.Arguments) != string(b[i].ToolCalls[j].Function.Arguments) {
				return false
			}
		}
	}
	return true
}

func cloneCompressionStateLocked(m *Manager) *CompressionState {
	state := &CompressionState{
		LastAssistantAt: cloneTimePtrLocked(m.lastAssistantAt),
		ToolArtifacts:   make([]ToolArtifact, 0, len(m.toolArtifacts)),
		SessionNotes:    cloneSessionNotesLocked(m.sessionNotes),
	}
	ids := make([]string, 0, len(m.toolArtifacts))
	for id := range m.toolArtifacts {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		state.ToolArtifacts = append(state.ToolArtifacts, m.toolArtifacts[id])
	}
	return state
}

func cloneTimePtrLocked(v *time.Time) *time.Time {
	if v == nil {
		return nil
	}
	copy := *v
	return &copy
}

func cloneSessionNotesLocked(v *SessionNotes) *SessionNotes {
	if v == nil {
		return nil
	}
	copy := *v
	return &copy
}
