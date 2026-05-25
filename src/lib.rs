/*!
llm-tool-registry: registry of available tools with JSON schemas.

```rust
use llm_tool_registry::{ToolRegistry, ToolDef};
use serde_json::json;

let mut reg = ToolRegistry::new();
reg.register(ToolDef::new("search", "Search the web")
    .param("query", "string", true));
assert_eq!(reg.len(), 1);
```
*/

use serde_json::{Value, json};
use std::collections::HashMap;

/// A parameter definition for a tool.
#[derive(Debug, Clone)]
pub struct ParamDef {
    pub name: String,
    pub param_type: String,
    pub description: Option<String>,
    pub required: bool,
}

/// A tool definition with name, description, and parameters.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub params: Vec<ParamDef>,
    pub tags: Vec<String>,
}

impl ToolDef {
    pub fn new(name: &str, description: &str) -> Self {
        Self { name: name.to_string(), description: description.to_string(), params: Vec::new(), tags: Vec::new() }
    }

    /// Add a parameter.
    pub fn param(mut self, name: &str, param_type: &str, required: bool) -> Self {
        self.params.push(ParamDef { name: name.to_string(), param_type: param_type.to_string(), description: None, required });
        self
    }

    /// Add a parameter with description.
    pub fn param_desc(mut self, name: &str, param_type: &str, required: bool, desc: &str) -> Self {
        self.params.push(ParamDef { name: name.to_string(), param_type: param_type.to_string(), description: Some(desc.to_string()), required });
        self
    }

    /// Add a tag for filtering.
    pub fn tag(mut self, t: &str) -> Self {
        self.tags.push(t.to_string());
        self
    }

    /// Serialize to Anthropic-style tool schema.
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

/// Registry of available tools.
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, ToolDef>,
}

impl ToolRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register(&mut self, tool: ToolDef) {
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&ToolDef> { self.tools.get(name) }

    pub fn remove(&mut self, name: &str) -> Option<ToolDef> { self.tools.remove(name) }

    pub fn len(&self) -> usize { self.tools.len() }
    pub fn is_empty(&self) -> bool { self.tools.is_empty() }

    /// All tool names (sorted).
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.tools.keys().map(|s| s.as_str()).collect();
        v.sort();
        v
    }

    /// All schemas for sending to an LLM.
    pub fn schemas(&self) -> Vec<Value> {
        let mut v: Vec<Value> = self.tools.values().map(|t| t.to_schema()).collect();
        v.sort_by_key(|v| v["name"].as_str().unwrap_or("").to_string());
        v
    }

    /// Filter tools by tag.
    pub fn by_tag(&self, tag: &str) -> Vec<&ToolDef> {
        let mut v: Vec<&ToolDef> = self.tools.values().filter(|t| t.tags.iter().any(|x| x == tag)).collect();
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
    fn schema_has_required_params() {
        let t = ToolDef::new("search", "Search").param("query", "string", true);
        let s = t.to_schema();
        assert!(s["input_schema"]["required"].as_array().unwrap().contains(&json!("query")));
    }

    #[test]
    fn optional_param_not_in_required() {
        let t = ToolDef::new("search", "Search").param("limit", "integer", false);
        let s = t.to_schema();
        let req = s["input_schema"]["required"].as_array().unwrap();
        assert!(!req.contains(&json!("limit")));
    }

    #[test]
    fn param_with_description() {
        let t = ToolDef::new("fn", "Fn").param_desc("q", "string", true, "search query");
        let s = t.to_schema();
        assert_eq!(s["input_schema"]["properties"]["q"]["description"], "search query");
    }

    #[test]
    fn schemas_returns_all() {
        let mut r = ToolRegistry::new();
        r.register(ToolDef::new("a", "A"));
        r.register(ToolDef::new("b", "B"));
        assert_eq!(r.schemas().len(), 2);
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
    fn remove() {
        let mut r = ToolRegistry::new();
        r.register(ToolDef::new("x", "X"));
        assert!(r.remove("x").is_some());
        assert!(r.get("x").is_none());
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
