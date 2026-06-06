# llm-tool-registry

A small, dependency-light Rust library for building a registry of tools (functions)
that you expose to an LLM agent, and for serializing those tools into the JSON
schema format that LLM APIs expect.

If you are building an agent that calls tools, you typically need to:

1. Describe each tool — its name, what it does, and its parameters.
2. Send those descriptions to the model as a list of JSON schemas.
3. Keep them organized and filterable as the set of tools grows.

`llm-tool-registry` gives you a tiny, ergonomic builder API for exactly that.

## Features

- Fluent `ToolDef` builder for defining tools and their parameters.
- Per-parameter typing, required/optional flags, and optional descriptions.
- Tag-based grouping and filtering of tools.
- `to_schema()` emits an Anthropic-style tool schema
  (`name`, `description`, `input_schema`) ready to hand to an LLM.
- `ToolRegistry` for registering, looking up, removing, and listing tools,
  with deterministic (sorted) ordering for stable output.
- Only one runtime dependency: [`serde_json`](https://crates.io/crates/serde_json).

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
llm-tool-registry = "0.1"
serde_json = "1"
```

## Usage

### Define a tool

```rust
use llm_tool_registry::ToolDef;

let search = ToolDef::new("search", "Search the web")
    .param_desc("query", "string", true, "The search query")
    .param("limit", "integer", false)
    .tag("read");

// Serialize to an LLM-ready JSON schema.
let schema = search.to_schema();
println!("{}", serde_json::to_string_pretty(&schema).unwrap());
```

This produces a schema shaped like:

```json
{
  "name": "search",
  "description": "Search the web",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "The search query" },
      "limit": { "type": "integer" }
    },
    "required": ["query"]
  }
}
```

### Build a registry

```rust
use llm_tool_registry::{ToolRegistry, ToolDef};

let mut reg = ToolRegistry::new();

reg.register(ToolDef::new("search", "Search the web")
    .param("query", "string", true)
    .tag("read"));

reg.register(ToolDef::new("write_file", "Write a file")
    .param("path", "string", true)
    .param("contents", "string", true)
    .tag("write"));

// Look up a single tool.
assert!(reg.get("search").is_some());

// All tool names, sorted.
assert_eq!(reg.names(), vec!["search", "write_file"]);

// All schemas to send to the model (sorted by name).
let schemas = reg.schemas();
assert_eq!(schemas.len(), 2);

// Filter by tag.
let read_only = reg.by_tag("read");
assert_eq!(read_only.len(), 1);
```

## API overview

### `ToolDef`

| Method | Description |
| --- | --- |
| `ToolDef::new(name, description)` | Create a new tool definition. |
| `.param(name, type, required)` | Add a parameter. |
| `.param_desc(name, type, required, desc)` | Add a parameter with a description. |
| `.tag(tag)` | Add a tag for grouping/filtering. |
| `.to_schema()` | Serialize to an Anthropic-style tool schema (`serde_json::Value`). |

### `ToolRegistry`

| Method | Description |
| --- | --- |
| `ToolRegistry::new()` | Create an empty registry. |
| `.register(tool)` | Register a tool (replaces any tool with the same name). |
| `.get(name)` | Borrow a tool by name. |
| `.remove(name)` | Remove and return a tool by name. |
| `.len()` / `.is_empty()` | Number of registered tools. |
| `.names()` | All tool names, sorted. |
| `.schemas()` | All tool schemas, sorted by name. |
| `.by_tag(tag)` | All tools carrying a given tag, sorted by name. |

## Building and testing

```sh
cargo build
cargo test
```

## Tech stack

- **Language:** Rust (edition 2021)
- **Dependencies:** `serde_json`

## License

Licensed under the MIT License. See the `license` field in `Cargo.toml`.
