/*!
`llm-tool-registry`: a small, dependency-light registry of tool definitions with
Anthropic-style JSON schemas for LLM agents.

Define tools with [`ToolDef`], collect them in a [`ToolRegistry`], and emit the
JSON schemas an LLM expects via [`ToolRegistry::schemas`] / [`ToolDef::to_schema`].

# Example

```rust
use llm_tool_registry::{ToolRegistry, ToolDef};

let mut reg = ToolRegistry::new();
reg.register(
    ToolDef::new("search", "Search the web")
        .param("query", "string", true),
);
assert_eq!(reg.len(), 1);

// Serialize every registered tool to the schema shape an LLM consumes.
let schemas = reg.schemas();
assert_eq!(schemas[0]["name"], "search");
assert_eq!(schemas[0]["input_schema"]["required"][0], "query");
```
*/

use serde_json::{json, Value};
use std::collections::HashMap;

/// A single parameter of a [`ToolDef`].
///
/// Parameters map onto the `properties` of the generated JSON schema; a
/// parameter whose [`required`](ParamDef::required) flag is set is also listed
/// in the schema's `required` array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamDef {
    /// Parameter name, used as the property key in the generated schema.
    pub name: String,
    /// JSON Schema type, e.g. `"string"`, `"integer"`, `"boolean"`.
    pub param_type: String,
    /// Optional human-readable description surfaced to the LLM.
    pub description: Option<String>,
    /// Whether the parameter is required (added to the schema's `required` list).
    pub required: bool,
}

/// A tool definition: a name, a description, an ordered list of parameters, and
/// optional tags for filtering.
///
/// Build one fluently with [`ToolDef::new`] followed by chained
/// [`param`](ToolDef::param) / [`param_desc`](ToolDef::param_desc) /
/// [`tag`](ToolDef::tag) calls, then serialize with
/// [`to_schema`](ToolDef::to_schema).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDef {
    /// Unique tool name (also the key used inside a [`ToolRegistry`]).
    pub name: String,
    /// Human-readable description of what the tool does.
    pub description: String,
    /// Ordered list of parameters.
    pub params: Vec<ParamDef>,
    /// Tags used by [`ToolRegistry::by_tag`] for filtering.
    pub tags: Vec<String>,
}

impl ToolDef {
    /// Create a new tool definition with the given name and description and no
    /// parameters or tags.
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            params: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Add a parameter without a description.
    ///
    /// Returns `self` for fluent chaining.
    pub fn param(mut self, name: &str, param_type: &str, required: bool) -> Self {
        self.params.push(ParamDef {
            name: name.to_string(),
            param_type: param_type.to_string(),
            description: None,
            required,
        });
        self
    }

    /// Add a parameter with a description.
    ///
    /// Returns `self` for fluent chaining.
    pub fn param_desc(mut self, name: &str, param_type: &str, required: bool, desc: &str) -> Self {
        self.params.push(ParamDef {
            name: name.to_string(),
            param_type: param_type.to_string(),
            description: Some(desc.to_string()),
            required,
        });
        self
    }

    /// Add a tag used for filtering via [`ToolRegistry::by_tag`].
    ///
    /// Returns `self` for fluent chaining.
    pub fn tag(mut self, t: &str) -> Self {
        self.tags.push(t.to_string());
        self
    }

    /// Serialize this tool into the Anthropic-style tool schema:
    ///
    /// ```json
    /// {
    ///   "name": "...",
    ///   "description": "...",
    ///   "input_schema": {
    ///     "type": "object",
    ///     "properties": { /* one entry per parameter */ },
    ///     "required": [ /* names of required parameters */ ]
    ///   }
    /// }
    /// ```
    pub fn to_schema(&self) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required_fields: Vec<Value> = Vec::new();

        for p in &self.params {
            let mut prop = json!({"type": p.param_type});
            if let Some(desc) = &p.description {
                prop["description"] = json!(desc);
            }
            properties.insert(p.name.clone(), prop);
            if p.required {
                required_fields.push(json!(p.name));
            }
        }

        json!({
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required_fields
            }
        })
    }
}

/// A registry of tools keyed by tool name.
///
/// Tool names are unique: registering a tool whose name already exists replaces
/// the previous definition (see [`register`](ToolRegistry::register)).
#[derive(Debug, Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, ToolDef>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool, keyed by its [`name`](ToolDef::name).
    ///
    /// If a tool with the same name is already registered it is **replaced**,
    /// and the previous definition is returned as `Some(old)`. Returns `None`
    /// when the name was not previously present.
    pub fn register(&mut self, tool: ToolDef) -> Option<ToolDef> {
        self.tools.insert(tool.name.clone(), tool)
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<&ToolDef> {
        self.tools.get(name)
    }

    /// Returns `true` if a tool with the given name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Remove a tool by name, returning it if it was present.
    pub fn remove(&mut self, name: &str) -> Option<ToolDef> {
        self.tools.remove(name)
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns `true` if no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// All tool names, sorted lexicographically.
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.tools.keys().map(|s| s.as_str()).collect();
        v.sort();
        v
    }

    /// Iterate over the registered tools in arbitrary order.
    ///
    /// Use [`names`](ToolRegistry::names) or [`schemas`](ToolRegistry::schemas)
    /// when you need a deterministic, sorted order.
    pub fn iter(&self) -> impl Iterator<Item = &ToolDef> {
        self.tools.values()
    }

    /// All tool schemas, sorted by tool name, ready to send to an LLM.
    pub fn schemas(&self) -> Vec<Value> {
        let mut v: Vec<Value> = self.tools.values().map(|t| t.to_schema()).collect();
        v.sort_by_key(|v| v["name"].as_str().unwrap_or("").to_string());
        v
    }

    /// All tools carrying the given tag, sorted by tool name.
    pub fn by_tag(&self, tag: &str) -> Vec<&ToolDef> {
        let mut v: Vec<&ToolDef> = self
            .tools
            .values()
            .filter(|t| t.tags.iter().any(|x| x == tag))
            .collect();
        v.sort_by_key(|t| &t.name);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_get() {
        let mut r = ToolRegistry::new();
        r.register(ToolDef::new("search", "Search the web"));
        assert!(r.get("search").is_some());
    }

    #[test]
    fn register_returns_none_for_new_name() {
        let mut r = ToolRegistry::new();
        assert!(r.register(ToolDef::new("a", "A")).is_none());
    }

    #[test]
    fn register_replaces_and_returns_previous() {
        let mut r = ToolRegistry::new();
        r.register(ToolDef::new("a", "first"));
        let prev = r.register(ToolDef::new("a", "second"));
        assert_eq!(prev.unwrap().description, "first");
        assert_eq!(r.get("a").unwrap().description, "second");
        // Re-registering the same name must not grow the registry.
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn contains() {
        let mut r = ToolRegistry::new();
        assert!(!r.contains("x"));
        r.register(ToolDef::new("x", "X"));
        assert!(r.contains("x"));
    }

    #[test]
    fn iter_yields_all_tools() {
        let mut r = ToolRegistry::new();
        r.register(ToolDef::new("a", "A"));
        r.register(ToolDef::new("b", "B"));
        let mut names: Vec<&str> = r.iter().map(|t| t.name.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn names_sorted() {
        let mut r = ToolRegistry::new();
        r.register(ToolDef::new("z_tool", "Z"));
        r.register(ToolDef::new("a_tool", "A"));
        assert_eq!(r.names()[0], "a_tool");
    }

    #[test]
    fn schema_has_name_and_description() {
        let t = ToolDef::new("search", "Web search");
        let s = t.to_schema();
        assert_eq!(s["name"], "search");
        assert_eq!(s["description"], "Web search");
    }

    #[test]
    fn schema_has_object_input_schema() {
        let t = ToolDef::new("search", "Web search");
        let s = t.to_schema();
        assert_eq!(s["input_schema"]["type"], "object");
    }

    #[test]
    fn schema_has_required_params() {
        let t = ToolDef::new("search", "Search").param("query", "string", true);
        let s = t.to_schema();
        assert!(s["input_schema"]["required"]
            .as_array()
            .unwrap()
            .contains(&json!("query")));
    }

    #[test]
    fn optional_param_not_in_required() {
        let t = ToolDef::new("search", "Search").param("limit", "integer", false);
        let s = t.to_schema();
        let req = s["input_schema"]["required"].as_array().unwrap();
        assert!(!req.contains(&json!("limit")));
    }

    #[test]
    fn param_type_is_preserved_in_schema() {
        let t = ToolDef::new("fn", "Fn").param("count", "integer", false);
        let s = t.to_schema();
        assert_eq!(s["input_schema"]["properties"]["count"]["type"], "integer");
    }

    #[test]
    fn param_with_description() {
        let t = ToolDef::new("fn", "Fn").param_desc("q", "string", true, "search query");
        let s = t.to_schema();
        assert_eq!(
            s["input_schema"]["properties"]["q"]["description"],
            "search query"
        );
    }

    #[test]
    fn param_without_description_omits_description_key() {
        let t = ToolDef::new("fn", "Fn").param("q", "string", true);
        let s = t.to_schema();
        assert!(s["input_schema"]["properties"]["q"]
            .get("description")
            .is_none());
    }

    #[test]
    fn tool_with_no_params_has_empty_required() {
        let t = ToolDef::new("ping", "Ping");
        let s = t.to_schema();
        assert!(s["input_schema"]["required"].as_array().unwrap().is_empty());
        assert!(s["input_schema"]["properties"]
            .as_object()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn schemas_returns_all() {
        let mut r = ToolRegistry::new();
        r.register(ToolDef::new("a", "A"));
        r.register(ToolDef::new("b", "B"));
        assert_eq!(r.schemas().len(), 2);
    }

    #[test]
    fn schemas_are_sorted_by_name() {
        let mut r = ToolRegistry::new();
        r.register(ToolDef::new("zebra", "Z"));
        r.register(ToolDef::new("apple", "A"));
        let s = r.schemas();
        assert_eq!(s[0]["name"], "apple");
        assert_eq!(s[1]["name"], "zebra");
    }

    #[test]
    fn by_tag() {
        let mut r = ToolRegistry::new();
        r.register(ToolDef::new("search", "Search").tag("read"));
        r.register(ToolDef::new("write", "Write").tag("write"));
        r.register(ToolDef::new("fetch", "Fetch").tag("read"));
        assert_eq!(r.by_tag("read").len(), 2);
        assert_eq!(r.by_tag("write").len(), 1);
    }

    #[test]
    fn by_tag_results_are_sorted_by_name() {
        let mut r = ToolRegistry::new();
        r.register(ToolDef::new("zebra", "Z").tag("read"));
        r.register(ToolDef::new("apple", "A").tag("read"));
        let read = r.by_tag("read");
        assert_eq!(read[0].name, "apple");
        assert_eq!(read[1].name, "zebra");
    }

    #[test]
    fn by_tag_unknown_tag_is_empty() {
        let mut r = ToolRegistry::new();
        r.register(ToolDef::new("search", "Search").tag("read"));
        assert!(r.by_tag("nope").is_empty());
    }

    #[test]
    fn remove() {
        let mut r = ToolRegistry::new();
        r.register(ToolDef::new("x", "X"));
        assert!(r.remove("x").is_some());
        assert!(r.get("x").is_none());
    }

    #[test]
    fn remove_absent_returns_none() {
        let mut r = ToolRegistry::new();
        assert!(r.remove("missing").is_none());
    }

    #[test]
    fn len_and_empty() {
        let mut r = ToolRegistry::new();
        assert!(r.is_empty());
        r.register(ToolDef::new("x", "X"));
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn multiple_params() {
        let t = ToolDef::new("fn", "Fn")
            .param("q", "string", true)
            .param("limit", "integer", false);
        let s = t.to_schema();
        let props = s["input_schema"]["properties"].as_object().unwrap();
        assert!(props.contains_key("q"));
        assert!(props.contains_key("limit"));
    }

    #[test]
    fn tag_chaining() {
        let t = ToolDef::new("x", "X").tag("a").tag("b");
        assert_eq!(t.tags.len(), 2);
    }
}
