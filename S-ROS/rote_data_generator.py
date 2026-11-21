"""
rote_data_generator.py
S-ROS Training Data: Option B Rote Pairs Organized by Rule Class

Each rule class contains:
  - A rule_id and human-readable description
  - Training pairs: input line + context -> semantic output
  - Held-out test pairs: same rule, novel instances never seen during training

The crystallization manager uses the held-out pairs to test generalization.
A rule crystallizes only when the network gets the held-out pairs right.
That is the test for understanding vs memorization.

TRAINING DATA PHILOSOPHY
We are not building a token predictor. We are building a generalization engine.
The training pairs teach instances. The test pairs verify that the system
learned the rule, not the instances. Crystallization locks the rule.

Genesis Labs Research, 2026
Amellia Mendel, Lisa Adler
"""

import json
import os
from typing import Dict, List


def build_rule_classes() -> List[Dict]:
    """
    Returns all rule classes for C64 BASIC V2 rote learning.
    Each class has train_pairs and test_pairs (held out).
    """

    rules = []

    # RULE 1: PRINT string literal
    rules.append({
        "rule_id": "print_string_literal",
        "description": "PRINT with a quoted string literal displays the string to screen",
        "train_pairs": [
            {"input": 'line 10: PRINT "Hello world"',
             "output": "Print 'Hello world' to screen, next line 20"},
            {"input": 'line 10: PRINT "Goodbye"',
             "output": "Print 'Goodbye' to screen, next line 20"},
            {"input": 'line 10: PRINT "READY."',
             "output": "Print 'READY.' to screen, next line 20"},
            {"input": 'line 10: PRINT "ABC123"',
             "output": "Print 'ABC123' to screen, next line 20"},
        ],
        "test_pairs": [
            {"input": 'line 10: PRINT "Testing"',
             "output": "Print 'Testing' to screen, next line 20"},
            {"input": 'line 10: PRINT "Genesis"',
             "output": "Print 'Genesis' to screen, next line 20"},
        ],
    })

    # RULE 2: PRINT with comma separator (zone output)
    rules.append({
        "rule_id": "print_comma_zones",
        "description": "PRINT with comma separator outputs values in tab zones",
        "train_pairs": [
            {"input": 'line 10: PRINT "Hello","world"',
             "output": "Print 'Hello' in zone1, 'world' in zone2, next line 20"},
            {"input": 'line 10: PRINT "A","B"',
             "output": "Print 'A' in zone1, 'B' in zone2, next line 20"},
            {"input": 'line 10: PRINT "X","Y","Z"',
             "output": "Print 'X' in zone1, 'Y' in zone2, 'Z' in zone3, next line 20"},
        ],
        "test_pairs": [
            {"input": 'line 10: PRINT "Cat","Dog"',
             "output": "Print 'Cat' in zone1, 'Dog' in zone2, next line 20"},
            {"input": 'line 10: PRINT "1","2","3"',
             "output": "Print '1' in zone1, '2' in zone2, '3' in zone3, next line 20"},
        ],
    })

    # RULE 3: PRINT with semicolon (concatenated output)
    rules.append({
        "rule_id": "print_semicolon_concat",
        "description": "PRINT with semicolon concatenates values with no space",
        "train_pairs": [
            {"input": 'line 10: PRINT "Hello";"world"',
             "output": "Print 'Helloworld' (no space), next line 20"},
            {"input": 'line 10: PRINT "A";"B"',
             "output": "Print 'AB' (no space), next line 20"},
            {"input": 'line 10: PRINT "Good";"bye"',
             "output": "Print 'Goodbye' (no space), next line 20"},
        ],
        "test_pairs": [
            {"input": 'line 10: PRINT "Sun";"rise"',
             "output": "Print 'Sunrise' (no space), next line 20"},
            {"input": 'line 10: PRINT "X";"Y";"Z"',
             "output": "Print 'XYZ' (no space), next line 20"},
        ],
    })

    # RULE 4: PRINT bare (blank line)
    rules.append({
        "rule_id": "print_blank",
        "description": "PRINT with no arguments outputs a blank line",
        "train_pairs": [
            {"input": "line 10: PRINT",
             "output": "Print blank line, next line 20"},
            {"input": "line 50: PRINT",
             "output": "Print blank line, next line 60"},
        ],
        "test_pairs": [
            {"input": "line 30: PRINT",
             "output": "Print blank line, next line 40"},
            {"input": "line 730: PRINT",
             "output": "Print blank line, next line 740"},
        ],
    })

    # RULE 5: PRINT numeric literal
    rules.append({
        "rule_id": "print_numeric",
        "description": "PRINT with a numeric literal displays the number",
        "train_pairs": [
            {"input": "line 10: PRINT 42",
             "output": "Print '42', next line 20"},
            {"input": "line 10: PRINT 3.14",
             "output": "Print '3.14', next line 20"},
            {"input": "line 10: PRINT 0",
             "output": "Print '0', next line 20"},
        ],
        "test_pairs": [
            {"input": "line 10: PRINT 99",
             "output": "Print '99', next line 20"},
            {"input": "line 10: PRINT 2.718",
             "output": "Print '2.718', next line 20"},
        ],
    })

    # RULE 6: PRINT variable value
    rules.append({
        "rule_id": "print_variable",
        "description": "PRINT with a variable displays its current value",
        "train_pairs": [
            {"input": "line 10: PRINT X (X=5)",
             "output": "Print '5', next line 20"},
            {"input": "line 10: PRINT Y (Y=10)",
             "output": "Print '10', next line 20"},
            {"input": "line 10: PRINT A (A=3.5)",
             "output": "Print '3.5', next line 20"},
        ],
        "test_pairs": [
            {"input": "line 10: PRINT Z (Z=7)",
             "output": "Print '7', next line 20"},
            {"input": "line 10: PRINT N (N=100)",
             "output": "Print '100', next line 20"},
        ],
    })

    # RULE 7: PRINT mixed string and variable
    rules.append({
        "rule_id": "print_mixed_string_var",
        "description": "PRINT with string literal and variable shows label then value",
        "train_pairs": [
            {"input": 'line 10: PRINT "X=",X (X=5)',
             "output": "Print 'X= ' then '5' in next zone, next line 20"},
            {"input": 'line 10: PRINT "Score:",S (S=100)',
             "output": "Print 'Score:' then '100' in next zone, next line 20"},
            {"input": 'line 10: PRINT "Total=",T (T=42)',
             "output": "Print 'Total=' then '42' in next zone, next line 20"},
        ],
        "test_pairs": [
            {"input": 'line 10: PRINT "Value:",V (V=88)',
             "output": "Print 'Value:' then '88' in next zone, next line 20"},
            {"input": 'line 10: PRINT "N=",N (N=3)',
             "output": "Print 'N=' then '3' in next zone, next line 20"},
        ],
    })

    # RULE 8: LET simple assignment
    rules.append({
        "rule_id": "let_simple_assign",
        "description": "LET assigns a numeric value to a variable",
        "train_pairs": [
            {"input": "line 10: LET X=5",
             "output": "X=5, next line 20"},
            {"input": "line 10: LET Y=3.14",
             "output": "Y=3.14, next line 20"},
            {"input": "line 10: LET A=0",
             "output": "A=0, next line 20"},
            {"input": "line 10: LET Z=100",
             "output": "Z=100, next line 20"},
        ],
        "test_pairs": [
            {"input": "line 10: LET N=42",
             "output": "N=42, next line 20"},
            {"input": "line 10: LET B=2.718",
             "output": "B=2.718, next line 20"},
        ],
    })

    # RULE 9: LET with arithmetic expression
    rules.append({
        "rule_id": "let_arithmetic",
        "description": "LET with arithmetic evaluates the expression and assigns result",
        "train_pairs": [
            {"input": "line 10: LET Y=X^2 (X=5)",
             "output": "Y=25, next line 20"},
            {"input": "line 10: LET Z=A+B (A=3,B=7)",
             "output": "Z=10, next line 20"},
            {"input": "line 10: LET R=X*2 (X=4)",
             "output": "R=8, next line 20"},
            {"input": "line 10: LET D=X-Y (X=10,Y=3)",
             "output": "D=7, next line 20"},
        ],
        "test_pairs": [
            {"input": "line 10: LET Q=A+1 (A=9)",
             "output": "Q=10, next line 20"},
            {"input": "line 10: LET P=X*X (X=3)",
             "output": "P=9, next line 20"},
        ],
    })

    # RULE 10: LET with built-in function
    rules.append({
        "rule_id": "let_builtin_function",
        "description": "LET with a built-in function evaluates the function",
        "train_pairs": [
            {"input": "line 10: LET Z=SQR(9)",
             "output": "Z=3, next line 20"},
            {"input": "line 10: LET A=ABS(-5)",
             "output": "A=5, next line 20"},
            {"input": "line 10: LET B=INT(3.7)",
             "output": "B=3, next line 20"},
        ],
        "test_pairs": [
            {"input": "line 10: LET C=SQR(16)",
             "output": "C=4, next line 20"},
            {"input": "line 10: LET D=ABS(-12)",
             "output": "D=12, next line 20"},
        ],
    })

    # RULE 11: GOTO unconditional jump
    rules.append({
        "rule_id": "goto_unconditional",
        "description": "GOTO jumps execution to the specified line number",
        "train_pairs": [
            {"input": "line 10: GOTO 100",
             "output": "Jump to line 100"},
            {"input": "line 50: GOTO 10",
             "output": "Jump to line 10"},
            {"input": "line 20: GOTO 200",
             "output": "Jump to line 200"},
        ],
        "test_pairs": [
            {"input": "line 30: GOTO 500",
             "output": "Jump to line 500"},
            {"input": "line 10: GOTO 40",
             "output": "Jump to line 40"},
        ],
    })

    # RULE 12: IF/THEN conditional (true)
    rules.append({
        "rule_id": "if_then_true",
        "description": "IF condition is true, execute THEN clause",
        "train_pairs": [
            {"input": "line 10: IF X>3 THEN GOTO 100 (X=5)",
             "output": "Condition true (5>3), jump to line 100"},
            {"input": "line 10: IF A=1 THEN GOTO 50 (A=1)",
             "output": "Condition true (1=1), jump to line 50"},
            {"input": "line 10: IF Y<10 THEN GOTO 200 (Y=3)",
             "output": "Condition true (3<10), jump to line 200"},
        ],
        "test_pairs": [
            {"input": "line 10: IF Z>=5 THEN GOTO 300 (Z=5)",
             "output": "Condition true (5>=5), jump to line 300"},
            {"input": "line 10: IF N<>0 THEN GOTO 80 (N=7)",
             "output": "Condition true (7<>0), jump to line 80"},
        ],
    })

    # RULE 13: IF/THEN conditional (false)
    rules.append({
        "rule_id": "if_then_false",
        "description": "IF condition is false, skip to next line",
        "train_pairs": [
            {"input": "line 10: IF X>3 THEN GOTO 100 (X=1)",
             "output": "Condition false (1>3), next line 20"},
            {"input": "line 10: IF A=5 THEN GOTO 50 (A=3)",
             "output": "Condition false (3=5), next line 20"},
            {"input": "line 10: IF Y<0 THEN GOTO 200 (Y=10)",
             "output": "Condition false (10<0), next line 20"},
        ],
        "test_pairs": [
            {"input": "line 10: IF Z>100 THEN GOTO 300 (Z=5)",
             "output": "Condition false (5>100), next line 20"},
            {"input": "line 10: IF N=0 THEN GOTO 80 (N=7)",
             "output": "Condition false (7=0), next line 20"},
        ],
    })

    # RULE 14: END statement
    rules.append({
        "rule_id": "end_halt",
        "description": "END halts program execution",
        "train_pairs": [
            {"input": "line 10: END",
             "output": "Halt program"},
            {"input": "line 100: END",
             "output": "Halt program"},
        ],
        "test_pairs": [
            {"input": "line 50: END",
             "output": "Halt program"},
            {"input": "line 730: END",
             "output": "Halt program"},
        ],
    })

    # RULE 15: REM comment
    rules.append({
        "rule_id": "rem_comment",
        "description": "REM is a comment, no operation, advance to next line",
        "train_pairs": [
            {"input": "line 10: REM This is a comment",
             "output": "No operation (comment), next line 20"},
            {"input": "line 50: REM Initialize variables",
             "output": "No operation (comment), next line 60"},
        ],
        "test_pairs": [
            {"input": "line 30: REM Loop start",
             "output": "No operation (comment), next line 40"},
        ],
    })

    return rules


def export_jsonl(rules: List[Dict], output_path: str) -> None:
    """Write all pairs (train + test, labeled) to JSONL for inspection."""
    with open(output_path, "w") as f:
        for rule in rules:
            for pair in rule["train_pairs"]:
                record = {
                    "rule_id": rule["rule_id"],
                    "split": "train",
                    "input": pair["input"],
                    "output": pair["output"],
                }
                f.write(json.dumps(record) + "\n")
            for pair in rule["test_pairs"]:
                record = {
                    "rule_id": rule["rule_id"],
                    "split": "test",
                    "input": pair["input"],
                    "output": pair["output"],
                }
                f.write(json.dumps(record) + "\n")


def summary(rules: List[Dict]) -> None:
    """Print rule class summary."""
    total_train = 0
    total_test = 0
    for r in rules:
        nt = len(r["train_pairs"])
        nh = len(r["test_pairs"])
        total_train += nt
        total_test += nh
        print(f"  {r['rule_id']:30s}  train:{nt:3d}  test:{nh:3d}  {r['description']}")
    print(f"\n  Total: {len(rules)} rule classes, {total_train} train pairs, {total_test} test pairs")


# =========================================================================
# SELF-TESTS
# =========================================================================

def test_all_rules_have_test_pairs():
    rules = build_rule_classes()
    for r in rules:
        assert len(r["test_pairs"]) > 0, f"Rule {r['rule_id']} has no test pairs"
    print(f"PASS: test_all_rules_have_test_pairs ({len(rules)} rules)")


def test_no_test_pair_in_train():
    """Verify no held-out pair appears in the training set."""
    rules = build_rule_classes()
    for r in rules:
        train_inputs = {p["input"] for p in r["train_pairs"]}
        for tp in r["test_pairs"]:
            assert tp["input"] not in train_inputs, (
                f"Rule {r['rule_id']}: test pair '{tp['input']}' leaked into train set"
            )
    print("PASS: test_no_test_pair_in_train")


def test_export_jsonl():
    rules = build_rule_classes()
    path = "/tmp/rote_test.jsonl"
    export_jsonl(rules, path)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) > 0
    for line in lines:
        record = json.loads(line)
        assert "rule_id" in record
        assert "split" in record
        assert record["split"] in ("train", "test")
    os.remove(path)
    print(f"PASS: test_export_jsonl ({len(lines)} records)")


if __name__ == "__main__":
    test_all_rules_have_test_pairs()
    test_no_test_pair_in_train()
    test_export_jsonl()
    print()
    rules = build_rule_classes()
    summary(rules)
