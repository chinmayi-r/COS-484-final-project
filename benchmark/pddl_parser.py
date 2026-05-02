"""Parse PlanBench Blocksworld PDDL instances into structured form."""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Tuple


@dataclass
class BlocksworldInstance:
    name: str
    blocks: List[str]
    # init predicates
    on_table: Set[str]       # on-table ?x
    on: List[Tuple[str, str]]  # (on ?x ?y) meaning x is on y
    clear: Set[str]
    arm_empty: bool
    # goal predicates
    goal_on: List[Tuple[str, str]]
    goal_on_table: Set[str]

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def difficulty(self) -> str:
        n = self.num_blocks
        if n <= 4:
            return "easy"
        elif n <= 5:
            return "medium"
        else:
            return "hard"


def _strip_comments(text: str) -> str:
    return re.sub(r";[^\n]*", "", text)


def _tokenize(text: str) -> List[str]:
    text = _strip_comments(text)
    text = text.replace("(", " ( ").replace(")", " ) ")
    return text.split()


def _parse_sexp(tokens: List[str], pos: int) -> Tuple[object, int]:
    """Recursive s-expression parser. Returns (parsed, next_pos)."""
    if tokens[pos] != "(":
        raise ValueError(f"Expected '(' at {pos}, got {tokens[pos]!r}")
    pos += 1
    result = []
    while tokens[pos] != ")":
        if tokens[pos] == "(":
            sub, pos = _parse_sexp(tokens, pos)
            result.append(sub)
        else:
            result.append(tokens[pos])
            pos += 1
    return result, pos + 1


def parse_pddl_instance(path: Path) -> BlocksworldInstance:
    text = path.read_text()
    tokens = _tokenize(text)

    # Find outer (define ...) block
    sexp, _ = _parse_sexp(tokens, 0)
    # sexp[0] == 'define', sexp[1] == ['problem', name], rest are sections
    name = sexp[1][1]

    sections = {s[0]: s[1:] for s in sexp[2:] if isinstance(s, list)}

    # :objects
    blocks = [t for t in sections[":objects"] if t != "-" and t != "object"]

    # :init
    on_table: Set[str] = set()
    on: List[Tuple[str, str]] = []
    clear: Set[str] = set()
    arm_empty = False

    for pred in sections[":init"]:
        if not isinstance(pred, list):
            continue
        pname = pred[0].lower()
        if pname == "on-table":
            on_table.add(pred[1])
        elif pname == "on":
            on.append((pred[1], pred[2]))
        elif pname == "clear":
            clear.add(pred[1])
        elif pname == "arm-empty":
            arm_empty = True

    # :goal — wrapped in (and ...)
    goal_on: List[Tuple[str, str]] = []
    goal_on_table: Set[str] = set()
    goal_body = sections[":goal"][0]  # (and ...)
    goal_preds = goal_body[1:] if goal_body[0].lower() == "and" else [goal_body]
    for pred in goal_preds:
        if not isinstance(pred, list):
            continue
        pname = pred[0].lower()
        if pname == "on":
            goal_on.append((pred[1], pred[2]))
        elif pname == "on-table":
            goal_on_table.add(pred[1])

    return BlocksworldInstance(
        name=name,
        blocks=blocks,
        on_table=on_table,
        on=on,
        clear=clear,
        arm_empty=arm_empty,
        goal_on=goal_on,
        goal_on_table=goal_on_table,
    )


def load_instances(instances_dir: Path) -> List[BlocksworldInstance]:
    """Load and sort all PDDL instances from a directory."""
    paths = sorted(instances_dir.glob("*.pddl"), key=lambda p: _instance_sort_key(p.name))
    instances = []
    for p in paths:
        try:
            instances.append(parse_pddl_instance(p))
        except Exception as e:
            print(f"Warning: could not parse {p.name}: {e}")
    return instances


def _instance_sort_key(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def state_to_nl(instance: BlocksworldInstance) -> str:
    """Convert instance init state to natural language."""
    parts = []
    # Build stacks from on predicates
    on_dict = {x: y for x, y in instance.on}  # x is on y
    for b in sorted(instance.blocks):
        if b in on_dict:
            parts.append(f"{b} is on {on_dict[b]}")
        elif b in instance.on_table:
            parts.append(f"{b} is on the table")
    return "; ".join(parts) + "; hand is empty" if instance.arm_empty else "; ".join(parts)


def goal_to_nl(instance: BlocksworldInstance) -> str:
    """Convert goal state to natural language."""
    parts = []
    for x, y in instance.goal_on:
        parts.append(f"{x} is on {y}")
    for b in sorted(instance.goal_on_table):
        parts.append(f"{b} is on the table")
    return "; ".join(parts)
