# Development Insights

- Most of the work is verifying grid assumptions visually
- AI tools don't reason in terms of implied pixels unless forced
- A debug grid overlay tool would help rapidly iterate on alignment

### A Unified Pipeline Beats a Medley of Tools

The most effective and maintainable approach is not to have numerous small, specialized scripts, but to build a **single, robust, and well-defined pipeline** (e.g., a `reconstructor.run()` method).

This central pipeline should then be driven by a single, focused validation script that is responsible for generating all necessary debug artifacts (grid overlays, difference maps, etc.). This architecture:

1.  Enforces a consistent, repeatable process.
2.  Makes it trivial to evaluate the impact of any single change.
3.  Keeps the core logic (`src`) clean from testing and validation code.

### AI tools don't reason in terms of implied pixels unless forced. 

AI tends to be very lazy in its assumptions and will assume "best case scenarios" when tackling complex problems, resulting in incorrect output. 

### Agentic AI assistants will be "confidently mistaken" and will manipulate test results to avoid confronting difficult problems. 

It will assume success simply because the scripts it created ran, but will be oblivious to obviously bad output of these scripts. Then it will prematurely celebrate success by making outputs in the console or md files. Sometimes it will knowingly hardcode the test instead of using test as a tool to debug. 

