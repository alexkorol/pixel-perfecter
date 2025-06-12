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