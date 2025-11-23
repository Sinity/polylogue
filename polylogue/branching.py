from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class MessageRecord:
    message_id: str
    parent_id: Optional[str]
    role: str
    text: str
    token_count: int
    word_count: int
    timestamp: Optional[str]
    attachments: int
    chunk: Dict[str, object]
    links: Sequence[Tuple[str, object]] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    branch_hint: Optional[str] = None
    content_hash: Optional[str] = None


@dataclass
class BranchInfo:
    branch_id: str
    parent_branch_id: Optional[str]
    message_ids: List[str]
    is_canonical: bool
    depth: int
    divergence_index: int


@dataclass
class BranchPlan:
    branches: Dict[str, BranchInfo]
    canonical_branch_id: str

    @property
    def branch_ids(self) -> List[str]:
        return list(self.branches.keys())

    def messages_for_branch(self, branch_id: str) -> List[str]:
        return self.branches[branch_id].message_ids


def _find_root_ids(nodes: Dict[str, MessageRecord]) -> List[str]:
    roots: List[str] = []
    for message_id, node in nodes.items():
        if not node.parent_id or node.parent_id not in nodes:
            roots.append(message_id)
    if not roots and nodes:
        # Fallback to first message when all parents exist but create a cycle.
        roots.append(next(iter(nodes)))
    return roots


def _build_paths(
    nodes: Dict[str, MessageRecord],
    child_map: Dict[str, List[str]],
    root_id: str,
) -> List[List[str]]:
    paths: List[List[str]] = []
    stack: List[Tuple[str, List[str], set[str]]] = [(root_id, [], set())]

    while stack:
        current_id, acc, visited = stack.pop()
        if current_id in visited:
            # Cycle detected; treat the accumulated path (including the current node) as a leaf.
            paths.append(acc + [current_id])
            continue
        new_path = acc + [current_id]
        new_visited = visited | {current_id}
        children = child_map.get(current_id, [])
        unvisited_children = [child for child in children if child not in new_visited]
        if not unvisited_children:
            paths.append(new_path)
            continue
        for child in reversed(unvisited_children):
            stack.append((child, new_path, new_visited))

    return paths


def build_branch_plan(
    messages: Iterable[MessageRecord],
    *,
    canonical_leaf_id: Optional[str] = None,
) -> BranchPlan:
    node_map: Dict[str, MessageRecord] = {}
    child_map: Dict[str, List[str]] = {}
    for record in messages:
        node_map[record.message_id] = record
    for record in node_map.values():
        if record.parent_id and record.parent_id in node_map:
            child_map.setdefault(record.parent_id, []).append(record.message_id)

    roots = _find_root_ids(node_map)
    if not roots:
        return BranchPlan(
            branches={
                "branch-000": BranchInfo(
                    branch_id="branch-000",
                    parent_branch_id=None,
                    message_ids=[],
                    is_canonical=True,
                    depth=0,
                    divergence_index=0,
                )
            },
            canonical_branch_id="branch-000",
        )

    all_paths: List[List[str]] = []
    for root_id in roots:
        all_paths.extend(_build_paths(node_map, child_map, root_id))

    if all_paths:
        # Deduplicate while preserving traversal order.
        seen: set[tuple[str, ...]] = set()
        unique_paths: List[List[str]] = []
        for path in all_paths:
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            unique_paths.append(path)
        all_paths = unique_paths

    if not all_paths:
        # No explicit leaves – treat roots as single-node branches.
        all_paths = [[root_id] for root_id in roots]

    def path_score(path: List[str]) -> Tuple[int, str]:
        last = node_map[path[-1]].timestamp if path else None
        return (len(path), last or "")  # length first, then timestamp ordering

    canonical_path: List[str]
    if canonical_leaf_id:
        canonical_candidates = [p for p in all_paths if canonical_leaf_id in p]
        canonical_path = canonical_candidates[0] if canonical_candidates else max(all_paths, key=path_score)
    else:
        canonical_path = max(all_paths, key=path_score)

    branches: Dict[str, BranchInfo] = {}
    canonical_branch_id = "branch-000"
    branches[canonical_branch_id] = BranchInfo(
        branch_id=canonical_branch_id,
        parent_branch_id=None,
        message_ids=canonical_path,
        is_canonical=True,
        depth=len(canonical_path),
        divergence_index=0,
    )

    assigned_paths = {tuple(canonical_path): canonical_branch_id}
    counter = 1

    def assign_branch_id() -> str:
        nonlocal counter
        branch_id = f"branch-{counter:03d}"
        counter += 1
        return branch_id

    for path in sorted(all_paths, key=path_score):
        t_path = tuple(path)
        if t_path in assigned_paths:
            continue
        parent_branch_id = canonical_branch_id
        parent_prefix = canonical_path
        divergence_index = 0
        # Compare against already assigned paths to find best parent (longest common prefix)
        for assigned_path, branch_id in assigned_paths.items():
            common_length = _common_prefix_length(assigned_path, path)
            if common_length > divergence_index:
                divergence_index = common_length
                parent_branch_id = branches[branch_id].branch_id
                parent_prefix = list(assigned_path)
        branch_id = assign_branch_id()
        branches[branch_id] = BranchInfo(
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            message_ids=path,
            is_canonical=False,
            depth=len(path),
            divergence_index=divergence_index,
        )
        assigned_paths[t_path] = branch_id

    return BranchPlan(branches=branches, canonical_branch_id=canonical_branch_id)


def _common_prefix_length(a: Sequence[str], b: Sequence[str]) -> int:
    length = 0
    for left, right in zip(a, b):
        if left != right:
            break
        length += 1
    return length
