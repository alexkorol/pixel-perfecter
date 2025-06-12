# Failures & Lessons Learned

- Early attempts using autocorrelation yielded false grid periods
- AI assistants and coders tend to make wrong assumptions a lot, and fall back on erroneous approaches even when corrected. 
- Averaging cell color instead of mode led to edge blur
- GPT-generated scripts rarely verify grid size correctly
- ChatGPT (o3 and 4o) are either unable or forget to visually check the outputs of their scripts and are more often than not "confidently wrong" in their responses when trying to tackle this problem. 
- Claude Sonnet 4 in github copilot gets carried away easily creating unnecessary and sometimes hilarious files (like a 'Mission Accomplished' PROJECT_SUMMARY.py that outputs self-congratulatory emojislop)

### Uncontrolled Script Proliferation
-   **Failure:** Allowing the AI assistant to create a new script for every minor algorithmic idea or debug attempt resulted in a completely unmanageable codebase. Instead of refining a central algorithm, it produced over half a dozen conflicting, half-finished approaches.
-   **Root Cause:** The AI optimizes for "completing the immediate prompt," which it often interprets as "create a new runnable file." It does not have the strategic oversight to recognize when this approach is counterproductive.
-   **Lesson Learned:** The AI's tendency to equate "running without syntax errors" with "a successful solution" is a persistent flaw. It requires a strong, explicit directive to **modify existing code** within a defined structure, rather than creating new files. Without this constraint, it will default to generating clutter.