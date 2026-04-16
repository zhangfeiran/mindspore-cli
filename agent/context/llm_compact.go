package context

import (
	stdctx "context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
)

const (
	envCompactMode       = "MSCLI_COMPACT_MODE"
	envDisableLLMCompact = "MSCLI_DISABLE_LLM_COMPACT"

	compactModeLLM      = "llm"
	compactModeLegacy   = "legacy"
	compactModePriority = "priority"
)

const compactSummarySystemPrompt = "You are a helpful AI assistant tasked with summarizing conversations."

const compactSummaryPrompt = `CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and the assistant's previous actions.
This summary should preserve technical details, code patterns, file paths, command results, decisions, and unresolved work that are essential for continuing development without losing context.

Before providing your final summary, wrap your drafting analysis in <analysis> tags. Then provide the final summary in <summary> tags.

In your analysis, chronologically inspect the conversation and identify:
- The user's explicit requests and intent.
- The assistant's approach and actions.
- Important technical concepts, architecture, and code patterns.
- Specific file names, functions, command outputs, and edits.
- Errors encountered and how they were fixed.
- User feedback or corrections.

Your <summary> section must include:
1. Primary Request and Intent: Capture all explicit user requests and intent.
2. Key Technical Concepts: List important technologies, APIs, packages, and architecture details.
3. Files and Code Sections: List specific files and code sections examined, modified, or created, and why they matter.
4. Errors and Fixes: List errors encountered and how they were handled.
5. Problem Solving: Document solved problems and ongoing troubleshooting.
6. All User Messages: List all non-tool user messages.
7. Pending Tasks: List work the user explicitly asked for that remains pending.
8. Current Work: Describe exactly what was being worked on immediately before compaction.
9. Optional Next Step: If there is a direct next step, state it and anchor it to the most recent user request.

Do not invent facts. Preserve concrete paths, commands, function names, and decisions whenever they appear in the conversation.`

var (
	compactAnalysisBlockRE = regexp.MustCompile(`(?is)<analysis>.*?</analysis>`)
	compactSummaryBlockRE  = regexp.MustCompile(`(?is)<summary>(.*?)</summary>`)
)

func compactModeFromEnv() string {
	if isTruthyEnv(os.Getenv(envDisableLLMCompact)) {
		return compactModeLegacy
	}

	mode := strings.ToLower(strings.TrimSpace(os.Getenv(envCompactMode)))
	switch mode {
	case "", compactModeLLM:
		return compactModeLLM
	case compactModeLegacy, "fallback", "heuristic":
		return compactModeLegacy
	case compactModePriority:
		return compactModePriority
	default:
		return compactModeLLM
	}
}

func isTruthyEnv(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "y", "on":
		return true
	default:
		return false
	}
}

func (m *Manager) compactWithLLMLocked(ctx stdctx.Context, targetTokens int) ([]llm.Message, CompactResult, error) {
	if m.provider == nil {
		return nil, CompactResult{Strategy: CompactStrategyLLM}, fmt.Errorf("llm compact provider is not configured")
	}
	if err := ctx.Err(); err != nil {
		return nil, CompactResult{Strategy: CompactStrategyLLM}, err
	}
	if len(m.messages) == 0 {
		return nil, CompactResult{Strategy: CompactStrategyLLM}, fmt.Errorf("not enough messages to compact")
	}

	maxTokens := compactSummaryMaxTokens(targetTokens)
	req := &llm.CompletionRequest{
		Messages: []llm.Message{
			llm.NewSystemMessage(compactSummarySystemPrompt),
			llm.NewUserMessage(compactSummaryPrompt + "\n\nConversation to summarize:\n\n" + renderConversationForCompact(m.messages)),
		},
		MaxTokens: &maxTokens,
	}

	if m.dumper != nil {
		ctx = llm.WithDebugDumper(ctx, m.dumper)
	}
	resp, err := m.provider.Complete(ctx, req)
	if err != nil {
		return nil, CompactResult{Strategy: CompactStrategyLLM}, fmt.Errorf("generate compact summary: %w", err)
	}
	summary := strings.TrimSpace(resp.Content)
	if summary == "" {
		return nil, CompactResult{Strategy: CompactStrategyLLM}, fmt.Errorf("generate compact summary: empty response")
	}
	if len(resp.ToolCalls) > 0 {
		return nil, CompactResult{Strategy: CompactStrategyLLM}, fmt.Errorf("generate compact summary: model attempted tool use")
	}

	summaryMsg := llm.NewUserMessage(compactContinuationMessage(summary, m.trajectoryPath))
	compacted := []llm.Message{summaryMsg}
	if estimateMessagesWithSystem(compacted, m.system, m.tokenizer) > targetTokens {
		return nil, CompactResult{Strategy: CompactStrategyLLM}, fmt.Errorf("generated compact summary exceeds target budget")
	}

	return compacted, CompactResult{
		Kept:     len(compacted),
		Removed:  len(m.messages) - len(compacted),
		Strategy: CompactStrategyLLM,
		Summary:  formatCompactSummary(summary),
	}, nil
}

func compactSummaryMaxTokens(targetTokens int) int {
	maxTokens := targetTokens / 2
	if maxTokens < 512 {
		maxTokens = 512
	}
	if maxTokens > 20000 {
		maxTokens = 20000
	}
	return maxTokens
}

func renderConversationForCompact(messages []llm.Message) string {
	var b strings.Builder
	for i, msg := range messages {
		fmt.Fprintf(&b, "## Message %d\nrole: %s\n", i+1, strings.TrimSpace(msg.Role))
		if msg.ToolCallID != "" {
			fmt.Fprintf(&b, "tool_call_id: %s\n", msg.ToolCallID)
		}
		if len(msg.ToolCalls) > 0 {
			if data, err := json.Marshal(msg.ToolCalls); err == nil {
				fmt.Fprintf(&b, "tool_calls: %s\n", string(data))
			}
		}
		content := strings.TrimSpace(msg.Content)
		if content == "" {
			content = "(empty)"
		}
		b.WriteString("content:\n")
		b.WriteString(content)
		b.WriteString("\n\n")
	}
	return b.String()
}

func formatCompactSummary(summary string) string {
	formatted := compactAnalysisBlockRE.ReplaceAllString(summary, "")
	if match := compactSummaryBlockRE.FindStringSubmatch(formatted); len(match) >= 2 {
		formatted = compactSummaryBlockRE.ReplaceAllString(formatted, "Summary:\n"+strings.TrimSpace(match[1]))
	}
	formatted = strings.ReplaceAll(formatted, "\r\n", "\n")
	for strings.Contains(formatted, "\n\n\n") {
		formatted = strings.ReplaceAll(formatted, "\n\n\n", "\n\n")
	}
	return strings.TrimSpace(formatted)
}

func compactContinuationMessage(summary, trajectoryPath string) string {
	var b strings.Builder
	b.WriteString("This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.\n\n")
	b.WriteString(formatCompactSummary(summary))
	if path := strings.TrimSpace(trajectoryPath); path != "" {
		b.WriteString("\n\nReference: the full trajectory is available at: ")
		b.WriteString(path)
	}
	b.WriteString("\n\nContinue from where the conversation left off. Do not acknowledge this summary unless the user asks about it.")
	return b.String()
}
