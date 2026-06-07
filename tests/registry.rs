//! Integration tests exercising the public API as an external crate would.

use llm_tool_registry::{ParamDef, ToolDef, ToolRegistry};
use serde_json::json;

#[test]
fn end_to_end_build_register_and_serialize() {
    let mut reg = ToolRegistry::new();

    reg.register(
        ToolDef::new("search", "Search the web")
            .param_desc("query", "string", true, "The search query")
            .param("limit", "integer", false)
            .tag("read"),
    );
    reg.register(
        ToolDef::new("write_file", "Write a file to disk")
            .param("path", "string", true)
            .param("contents", "string", true)
            .tag("write"),
    );

    assert_eq!(reg.len(), 2);
    assert!(reg.contains("search"));
    assert!(!reg.contains("delete"));

    // names() is sorted.
    assert_eq!(reg.names(), vec!["search", "write_file"]);

    // schemas() is sorted by name and shaped for an LLM.
    let schemas = reg.schemas();
    assert_eq!(schemas.len(), 2);
    assert_eq!(schemas[0]["name"], "search");
    assert_eq!(schemas[0]["input_schema"]["type"], "object");
    assert_eq!(
        schemas[0]["input_schema"]["properties"]["query"]["description"],
        "The search query"
    );

    // Only `query` is required for `search`; `limit` is optional.
    let required = schemas[0]["input_schema"]["required"].as_array().unwrap();
    assert!(required.contains(&json!("query")));
    assert!(!required.contains(&json!("limit")));
}

#[test]
fn by_tag_filters_across_registry() {
    let mut reg = ToolRegistry::new();
    reg.register(ToolDef::new("a", "A").tag("read"));
    reg.register(ToolDef::new("b", "B").tag("read"));
    reg.register(ToolDef::new("c", "C").tag("write"));

    let read = reg.by_tag("read");
    assert_eq!(read.len(), 2);
    assert_eq!(read[0].name, "a");
    assert_eq!(read[1].name, "b");

    assert_eq!(reg.by_tag("write").len(), 1);
    assert!(reg.by_tag("admin").is_empty());
}

#[test]
fn register_replaces_existing_tool() {
    let mut reg = ToolRegistry::new();
    assert!(reg.register(ToolDef::new("x", "first")).is_none());

    let previous = reg.register(ToolDef::new("x", "second"));
    assert_eq!(previous.unwrap().description, "first");
    assert_eq!(reg.len(), 1);
    assert_eq!(reg.get("x").unwrap().description, "second");
}

#[test]
fn remove_returns_the_removed_tool() {
    let mut reg = ToolRegistry::new();
    reg.register(ToolDef::new("x", "X").param("a", "string", true));

    let removed = reg.remove("x").expect("tool should exist");
    assert_eq!(removed.name, "x");
    assert_eq!(removed.params.len(), 1);
    assert!(reg.is_empty());
    assert!(reg.remove("x").is_none());
}

#[test]
fn tool_def_is_clone_and_eq() {
    let original = ToolDef::new("echo", "Echo input").param("text", "string", true);
    let copy = original.clone();
    assert_eq!(original, copy);

    let expected_param = ParamDef {
        name: "text".to_string(),
        param_type: "string".to_string(),
        description: None,
        required: true,
    };
    assert_eq!(copy.params[0], expected_param);
}

#[test]
fn iter_visits_every_tool() {
    let mut reg = ToolRegistry::new();
    reg.register(ToolDef::new("a", "A"));
    reg.register(ToolDef::new("b", "B"));
    reg.register(ToolDef::new("c", "C"));

    let count = reg.iter().count();
    assert_eq!(count, 3);

    let mut names: Vec<&str> = reg.iter().map(|t| t.name.as_str()).collect();
    names.sort();
    assert_eq!(names, vec!["a", "b", "c"]);
}
