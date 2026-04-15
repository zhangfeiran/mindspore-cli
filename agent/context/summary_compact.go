package context

import (
	stdctx "context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/mindspore-lab/mindspore-cli/configs"
	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
)

const (
	summaryCompactTriggerAuto   = "auto"
	summaryCompactTriggerManual = "manual"
	summaryCompactMaxRetries    = 3
)

var (
	errSummaryCompactStale          = errors.New("context changed during summary compaction")
	errSummaryCompactNoSummarizable = errors.New("no compactable history to summarize")

	compactAnalysisPattern = regexp.MustCompile(`(?is)<analysis>[\s\S]*?</analysis>`)
	compactSummaryPattern  = regexp.MustCompile(`(?is)<summary>([\s\S]*?)</summary>`)
)

type SummaryCompactOptions struct {
	Trigger          string
	MaxSummaryTokens int
	LocalFallback    bool
}

type summaryCompactPlan struct {
	expected             []llm.Message
	working              []llm.Message
	kept                 []llm.Message
	toSummarize          []messageGroup
	beforeTokens         int
	targetTokens         int
	persistedToolResults int
	clearedToolResults   int
}

// PrepareForRequestWithSummary prepares context for an LLM request using LLM-backed summarization when needed.
func (m *Manager) PrepareForRequestWithSummary(ctx stdctx.Context, provider llm.Provider, opts SummaryCompactOptions, now time.Time) (PrepareResult, error) {
	opts.Trigger = summaryCompactTriggerAuto
	result, err := m.prepareWithSummaryCompact(ctx, provider, opts, now, false)
	if err != nil && opts.LocalFallback && !errors.Is(err, errSummaryCompactStale) {
		fallback, fallbackErr := m.PrepareForRequest(now)
		fallback.LocalFallback = true
		return fallback, fallbackErr
	}
	return result, err
}

// CompactWithSummary manually compacts context using an LLM-generated summary.
func (m *Manager) CompactWithSummary(ctx stdctx.Context, provider llm.Provider, opts SummaryCompactOptions) (PrepareResult, error) {
	opts.Trigger = summaryCompactTriggerManual
	return m.prepareWithSummaryCompact(ctx, provider, opts, time.Now(), true)
}

func (m *Manager) prepareWithSummaryCompact(ctx stdctx.Context, provider llm.Provider, opts SummaryCompactOptions, now time.Time, force bool) (PrepareResult, error) {
	if ctx == nil {
		ctx = stdctx.Background()
	}
	if now.IsZero() {
		now = time.Now()
	}

	plan, done, err := m.buildSummaryCompactPlan(now, force, opts.LocalFallback)
	if err != nil {
		return PrepareResult{}, err
	}
	if done != nil {
		return *done, nil
	}
	if provider == nil {
		return PrepareResult{BeforeTokens: plan.beforeTokens, AfterTokens: plan.beforeTokens}, fmt.Errorf("summary compact provider is not configured")
	}

	summary, usage, err := m.generateCompactSummary(ctx, provider, plan.toSummarize, opts)
	if err != nil {
		return PrepareResult{BeforeTokens: plan.beforeTokens, AfterTokens: plan.beforeTokens, CompactionUsage: usage}, err
	}

	return m.applySummaryCompactPlan(plan, summary, usage, now, opts.Trigger)
}

func (m *Manager) buildSummaryCompactPlan(now time.Time, force bool, localFallback bool) (*summaryCompactPlan, *PrepareResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	beforeTokens := m.currentTokensLocked()
	expected := cloneMessages(m.messages)
	working := cloneMessages(m.messages)

	persisted, next, err := m.applyAggregateToolResultBudgetLocked(working, now)
	if err != nil {
		return nil, nil, err
	}
	working = next

	cleared := 0
	if next, count := m.applyTimeBasedMicrocompactLocked(working, now); count > 0 {
		working = next
		cleared = count
	}

	shouldCompact := force || m.shouldAutoCompactPreparedLocked(working)
	if !shouldCompact {
		result := m.commitPreparedMessagesLocked(working, beforeTokens, false, false, false, persisted, cleared, llm.Usage{})
		return nil, &result, nil
	}

	keptGroups, _, targetTokens := m.selectSummaryCompactGroupsLocked(working)
	summarizeGroups := groupMessages(working)
	if len(summarizeGroups) == 0 {
		if localFallback {
			return nil, nil, errSummaryCompactNoSummarizable
		}
		result := m.commitPreparedMessagesLocked(working, beforeTokens, false, false, false, persisted, cleared, llm.Usage{})
		return nil, &result, nil
	}

	return &summaryCompactPlan{
		expected:             expected,
		working:              working,
		kept:                 flattenMessageGroups(keptGroups),
		toSummarize:          summarizeGroups,
		beforeTokens:         beforeTokens,
		targetTokens:         targetTokens,
		persistedToolResults: persisted,
		clearedToolResults:   cleared,
	}, nil, nil
}

func (m *Manager) applySummaryCompactPlan(plan *summaryCompactPlan, rawSummary string, usage llm.Usage, now time.Time, trigger string) (PrepareResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if plan == nil {
		return PrepareResult{}, fmt.Errorf("summary compact plan is nil")
	}
	if !messagesEqual(m.messages, plan.expected) {
		return PrepareResult{}, errSummaryCompactStale
	}

	summary := formatCompactSummary(rawSummary)
	if summary == "" {
		return PrepareResult{}, fmt.Errorf("summary compact returned empty summary")
	}
	if startsWithAPIError(summary) {
		return PrepareResult{}, errors.New(summary)
	}

	suppressFollowUp := strings.EqualFold(strings.TrimSpace(trigger), summaryCompactTriggerAuto)
	summaryMsg := llm.NewUserMessage(buildPostCompactSummaryContent(summary, trigger, plan.beforeTokens, suppressFollowUp))
	summaryMsg = m.fitSummaryMessageLocked(summaryMsg, plan.targetTokens)
	finalMessages := m.fitPostSummaryMessagesLocked(summaryMsg, plan.kept, plan.targetTokens)
	if len(finalMessages) == 0 {
		return PrepareResult{}, fmt.Errorf("summary compact produced no messages")
	}

	finalTokens := estimateMessagesWithSystem(finalMessages, m.system, m.tokenizer)
	maxUsable := m.maxUsableTokensLocked()
	if finalTokens > maxUsable {
		return PrepareResult{}, fmt.Errorf("summary compact exceeds context budget: %d tokens > %d", finalTokens, maxUsable)
	}
	if finalTokens >= estimateMessagesWithSystem(plan.working, m.system, m.tokenizer) {
		return PrepareResult{}, fmt.Errorf("summary compact did not reduce context")
	}

	m.messages = finalMessages
	m.refreshMessageStateLocked()
	m.clearProviderTokenUsageLocked()
	m.recalculateUsage()
	m.stats.CompactCount++
	compactAt := now
	m.stats.LastCompactAt = &compactAt

	return PrepareResult{
		Changed:              true,
		AutoCompacted:        true,
		LLMCompacted:         true,
		ToolResultsPersisted: plan.persistedToolResults,
		ToolResultsCleared:   plan.clearedToolResults,
		BeforeTokens:         plan.beforeTokens,
		AfterTokens:          m.currentTokensLocked(),
		CompactionUsage:      usage,
	}, nil
}

func (m *Manager) commitPreparedMessagesLocked(messages []llm.Message, beforeTokens int, autoCompacted, llmCompacted, localFallback bool, persisted, cleared int, usage llm.Usage) PrepareResult {
	changed := !messagesEqual(messages, m.messages) || persisted > 0 || cleared > 0
	if changed {
		m.messages = messages
		m.refreshMessageStateLocked()
		m.clearProviderTokenUsageLocked()
		m.recalculateUsage()
	}

	return PrepareResult{
		Changed:              changed,
		AutoCompacted:        autoCompacted,
		LLMCompacted:         llmCompacted,
		LocalFallback:        localFallback,
		ToolResultsPersisted: persisted,
		ToolResultsCleared:   cleared,
		BeforeTokens:         beforeTokens,
		AfterTokens:          m.currentTokensLocked(),
		CompactionUsage:      usage,
	}
}

func (m *Manager) selectSummaryCompactGroupsLocked(messages []llm.Message) ([]messageGroup, []messageGroup, int) {
	groups := groupMessages(messages)
	if len(groups) == 0 {
		return nil, nil, m.summaryCompactTargetTokensLocked()
	}

	pinned := pinnedMessageGroups(groups)
	pinnedSet := messageGroupSet(pinned)
	selected := append([]messageGroup{}, pinned...)
	usedTokens := countTokensInGroups(selected, m.tokenizer)
	textMessages := countTextMessagesInGroups(selected)

	targetTokens := m.summaryCompactTargetTokensLocked()
	tailMax := m.config.AutoCompactMaxTailTokens
	if tailMax <= 0 || tailMax > targetTokens {
		tailMax = targetTokens
	}
	minTail := m.config.AutoCompactMinTailTokens
	if minTail <= 0 {
		minTail = tailMax
	}
	if minTail > tailMax {
		minTail = tailMax
	}
	minMessages := m.config.AutoCompactMinMessages
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
	return selected, excludeMessageGroups(groups, selected), targetTokens
}

func (m *Manager) summaryCompactTargetTokensLocked() int {
	target := m.maxUsableTokensLocked() / 2
	if target <= 0 {
		target = m.maxUsableTokensLocked()
	}
	if target <= 0 {
		target = 1
	}
	return target
}

func (m *Manager) generateCompactSummary(ctx stdctx.Context, provider llm.Provider, groups []messageGroup, opts SummaryCompactOptions) (string, llm.Usage, error) {
	if len(groups) == 0 {
		return "", llm.Usage{}, errSummaryCompactNoSummarizable
	}

	maxTokens := opts.MaxSummaryTokens
	if maxTokens <= 0 {
		maxTokens = m.config.CompactSummaryMaxTokens
	}
	if maxTokens <= 0 {
		maxTokens = configs.DefaultCompactSummaryMaxTokens
	}

	summaryGroups := append([]messageGroup(nil), groups...)
	truncated := false
	var lastUsage llm.Usage
	for attempt := 0; ; attempt++ {
		resp, err := provider.Complete(ctx, &llm.CompletionRequest{
			Messages: []llm.Message{
				llm.NewSystemMessage("You are a helpful AI assistant tasked with summarizing conversations. Respond with text only and do not call tools."),
				llm.NewUserMessage(buildCompactSummaryRequest(summaryGroups, truncated)),
			},
			MaxTokens: &maxTokens,
		})
		if resp != nil {
			lastUsage = resp.Usage.Clone()
		}
		if err == nil {
			if resp == nil {
				return "", lastUsage, fmt.Errorf("summary compact returned nil response")
			}
			if strings.TrimSpace(resp.Content) == "" {
				return "", lastUsage, fmt.Errorf("summary compact returned empty response")
			}
			return resp.Content, lastUsage, nil
		}

		if !isPromptTooLongError(err) || attempt >= summaryCompactMaxRetries || len(summaryGroups) <= 1 {
			return "", lastUsage, err
		}
		drop := len(summaryGroups) / 5
		if drop < 1 {
			drop = 1
		}
		if drop >= len(summaryGroups) {
			drop = len(summaryGroups) - 1
		}
		summaryGroups = summaryGroups[drop:]
		truncated = true
	}
}

func buildCompactSummaryRequest(groups []messageGroup, truncated bool) string {
	var sb strings.Builder
	sb.WriteString("CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\n")
	sb.WriteString("Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and previous assistant actions. ")
	sb.WriteString("This summary must preserve technical details, code patterns, files touched, errors and fixes, pending tasks, and the exact current work needed to continue.\n\n")
	sb.WriteString("Before the final summary, write an <analysis> block. Then write the final result in a <summary> block with these sections:\n")
	sb.WriteString("1. Primary Request and Intent\n")
	sb.WriteString("2. Key Technical Concepts\n")
	sb.WriteString("3. Files and Code Sections\n")
	sb.WriteString("4. Errors and Fixes\n")
	sb.WriteString("5. Problem Solving\n")
	sb.WriteString("6. All User Messages\n")
	sb.WriteString("7. Pending Tasks\n")
	sb.WriteString("8. Current Work\n")
	sb.WriteString("9. Optional Next Step\n\n")
	if truncated {
		sb.WriteString("[Earlier conversation was truncated before summarization because the compact request exceeded the provider context limit.]\n\n")
	}
	sb.WriteString("<conversation>\n")
	for _, group := range groups {
		for _, msg := range group.Messages {
			sb.WriteString(formatMessageForCompactTranscript(msg))
		}
	}
	sb.WriteString("</conversation>\n\n")
	sb.WriteString("REMINDER: Do NOT call tools. Respond with plain text only: an <analysis> block followed by a <summary> block.")
	return sb.String()
}

func formatMessageForCompactTranscript(msg llm.Message) string {
	var sb strings.Builder
	role := strings.TrimSpace(msg.Role)
	if role == "" {
		role = "message"
	}
	sb.WriteString("\n--- ")
	sb.WriteString(strings.ToUpper(role))
	if msg.ToolCallID != "" {
		sb.WriteString(" tool_call_id=")
		sb.WriteString(msg.ToolCallID)
	}
	sb.WriteString(" ---\n")
	if text := strings.TrimSpace(msg.Content); text != "" {
		sb.WriteString(truncateCompactTranscriptText(text, 20000))
		sb.WriteString("\n")
	}
	for _, tc := range msg.ToolCalls {
		sb.WriteString("Tool call: ")
		sb.WriteString(strings.TrimSpace(tc.Function.Name))
		if id := strings.TrimSpace(tc.ID); id != "" {
			sb.WriteString(" id=")
			sb.WriteString(id)
		}
		if len(tc.Function.Arguments) > 0 {
			sb.WriteString(" args=")
			sb.WriteString(truncateCompactTranscriptText(string(tc.Function.Arguments), 4000))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func truncateCompactTranscriptText(text string, maxRunes int) string {
	if maxRunes <= 0 {
		return ""
	}
	runes := []rune(text)
	if len(runes) <= maxRunes {
		return text
	}
	return string(runes[:maxRunes]) + "\n[truncated for compact summary]"
}

func formatCompactSummary(summary string) string {
	formatted := compactAnalysisPattern.ReplaceAllString(summary, "")
	if match := compactSummaryPattern.FindStringSubmatch(formatted); len(match) > 1 {
		formatted = "Summary:\n" + strings.TrimSpace(match[1])
	}
	formatted = strings.ReplaceAll(formatted, "\r\n", "\n")
	for strings.Contains(formatted, "\n\n\n") {
		formatted = strings.ReplaceAll(formatted, "\n\n\n", "\n\n")
	}
	return strings.TrimSpace(formatted)
}

func buildPostCompactSummaryContent(summary, trigger string, preCompactTokens int, suppressFollowUp bool) string {
	trigger = strings.TrimSpace(trigger)
	if trigger == "" {
		trigger = summaryCompactTriggerAuto
	}
	var sb strings.Builder
	sb.WriteString("<compact-boundary trigger=\"")
	sb.WriteString(trigger)
	sb.WriteString(fmt.Sprintf("\" pre_tokens=\"%d\">\n", preCompactTokens))
	sb.WriteString("This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.\n\n")
	sb.WriteString(summary)
	sb.WriteString("\n\nRecent messages are preserved verbatim after this summary.")
	if suppressFollowUp {
		sb.WriteString("\n\nContinue the conversation from where it left off without asking the user any further questions. Resume directly and do not acknowledge this summary.")
	}
	sb.WriteString("\n</compact-boundary>")
	return sb.String()
}

func (m *Manager) fitPostSummaryMessagesLocked(summaryMsg llm.Message, kept []llm.Message, targetTokens int) []llm.Message {
	if targetTokens <= 0 {
		targetTokens = m.maxUsableTokensLocked()
	}
	remaining := targetTokens - m.tokenizer.EstimateMessage(summaryMsg)
	if m.system != nil {
		remaining -= m.tokenizer.EstimateMessage(*m.system)
	}
	var keptTail []llm.Message
	if remaining > 0 {
		keptTail = keepRecentMessagesByTokens(kept, remaining, m.tokenizer)
	}
	result := make([]llm.Message, 0, 1+len(keptTail))
	result = append(result, summaryMsg)
	result = append(result, keptTail...)
	return result
}

func (m *Manager) fitSummaryMessageLocked(msg llm.Message, targetTokens int) llm.Message {
	systemTokens := 0
	if m.system != nil {
		systemTokens = m.tokenizer.EstimateMessage(*m.system)
	}
	budget := targetTokens - systemTokens
	if budget <= 0 {
		budget = targetTokens
	}
	if budget <= 0 {
		budget = 1
	}
	for m.tokenizer.EstimateMessage(msg) > budget {
		runes := []rune(msg.Content)
		if len(runes) <= 128 {
			break
		}
		nextLen := len(runes) * 3 / 4
		if nextLen < 128 {
			nextLen = 128
		}
		msg.Content = string(runes[:nextLen]) + "\n\n[Compact summary truncated to fit context budget.]"
	}
	return msg
}

func isPromptTooLongError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	needles := []string{
		"prompt is too long",
		"context length",
		"context_length_exceeded",
		"maximum context",
		"too many tokens",
		"request too large",
		"413",
		"tokens >",
	}
	for _, needle := range needles {
		if strings.Contains(msg, needle) {
			return true
		}
	}
	return false
}

func startsWithAPIError(text string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(text)), "api error")
}

func (m *Manager) refreshMessageStateLocked() {
	m.stats.MessageCount = len(m.messages)
	m.stats.ToolCallCount = 0
	m.toolCallNames = make(map[string]string)
	for _, msg := range m.messages {
		if msg.Role == "tool" {
			m.stats.ToolCallCount++
		}
		if msg.Role != "assistant" {
			continue
		}
		for _, tc := range msg.ToolCalls {
			if id := strings.TrimSpace(tc.ID); id != "" {
				m.toolCallNames[id] = strings.TrimSpace(tc.Function.Name)
			}
		}
	}
}
