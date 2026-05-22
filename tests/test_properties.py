from hypothesis import given
from hypothesis import strategies as st

from nesy_reasoning_mcp.reasoning import build_graph
from nesy_reasoning_mcp.schemas import ConfidencePolicy, RelationInput, RelationType
from nesy_reasoning_mcp.store import RelationStore


def _dag_edges(max_node: int = 5):
    return st.lists(
        st.tuples(st.integers(0, max_node - 1), st.integers(1, max_node)),
        min_size=1,
        max_size=8,
        unique=True,
    ).map(lambda edges: [(left, right) for left, right in edges if left < right])


@given(_dag_edges())
def test_deleting_edge_does_not_add_reachability(edges: list[tuple[int, int]]) -> None:
    if not edges:
        return
    relations = [
        RelationInput(
            source=f"N{left}",
            target=f"N{right}",
            relation_type=RelationType.SUFFICIENT,
        )
        for left, right in edges
    ]
    before_store = RelationStore()
    before_store.assert_relations(relations)
    before = _reachable_pairs(before_store)

    after_store = RelationStore()
    after_store.assert_relations(relations[1:])
    after = _reachable_pairs(after_store)

    assert after <= before


@given(st.floats(min_value=0, max_value=1), st.floats(min_value=0, max_value=1))
def test_product_confidence_never_exceeds_path_edges(first: float, second: float) -> None:
    store = RelationStore()
    store.assert_relations(
        [
            RelationInput(
                source="A",
                target="B",
                relation_type=RelationType.SUFFICIENT,
                confidence=first,
            ),
            RelationInput(
                source="B",
                target="C",
                relation_type=RelationType.SUFFICIENT,
                confidence=second,
            ),
        ]
    )
    path = build_graph(store.list_relations()).find_paths(
        "A",
        "C",
        max_depth=2,
        confidence_policy=ConfidencePolicy.PRODUCT_INDEPENDENT,
    )[0]

    assert path.evidence_confidence <= first
    assert path.evidence_confidence <= second


@given(st.text(min_size=1, max_size=16), st.text(min_size=1, max_size=16))
def test_equivalent_creates_bidirectional_paths(left: str, right: str) -> None:
    if left.strip() == right.strip() or not left.strip() or not right.strip():
        return
    store = RelationStore()
    store.assert_relations(
        [
            RelationInput(
                source=left,
                target=right,
                relation_type=RelationType.EQUIVALENT,
            )
        ]
    )
    index = build_graph(store.list_relations())

    assert index.find_paths(left.strip(), right.strip(), max_depth=1)
    assert index.find_paths(right.strip(), left.strip(), max_depth=1)


def _reachable_pairs(store: RelationStore) -> set[tuple[str, str]]:
    index = build_graph(store.list_relations())
    nodes = sorted(str(node) for node in index.graph.nodes)
    pairs: set[tuple[str, str]] = set()
    for source in nodes:
        for target in nodes:
            if source == target:
                continue
            if index.find_paths(source, target, max_depth=len(nodes)):
                pairs.add((source, target))
    return pairs
