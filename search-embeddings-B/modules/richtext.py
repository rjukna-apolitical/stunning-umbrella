def extract_values(nodes: list) -> list[str]:
    """Recursively extract text values from Contentful rich text nodes."""
    texts = []
    for node in nodes:
        if node.get("nodeType") == "text":
            value = node.get("value", "").strip()
            if value:
                texts.append(value)
        if "content" in node:
            texts.extend(extract_values(node["content"]))
    return texts
