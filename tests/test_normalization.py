from nesy_reasoning_mcp.normalization import (
    normalize_relation_edges,
    normalized_implication_preview,
)
from nesy_reasoning_mcp.schemas import RelationInput, RelationRecord


def test_normalized_implication_preview_matches_relation_semantics() -> None:
    assert normalized_implication_preview("A", "B", "sufficient") == [
        {"antecedent": "A", "consequent": "B"}
    ]
    assert normalized_implication_preview("A", "B", "necessary") == [
        {"antecedent": "B", "consequent": "A"}
    ]
    assert normalized_implication_preview("A", "B", "equivalent") == [
        {"antecedent": "A", "consequent": "B"},
        {"antecedent": "B", "consequent": "A"},
    ]


def test_normalize_relation_edges_keeps_existing_direction_semantics() -> None:
    relation = RelationRecord.from_input(
        RelationInput(source="A", target="B", relation_type="necessary")
    )

    edges = normalize_relation_edges(relation)

    assert [(edge.antecedent, edge.consequent) for edge in edges] == [("B", "A")]
