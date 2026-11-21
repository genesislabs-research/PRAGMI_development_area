"""
curriculum_data.py
Curriculum-Staged Training Data for Theo SNN Execution Core

Stage 1: Keyword Recognition (12-way classification of isolated keywords)
Stage 2: Syntax Pattern Recognition (15-way classification of full statements)
Stage 3: Full Statement Processing (classification + template rendering verification)

The held-out test sets are strictly disjoint from training sets at every stage.
Crystallization is gated on generalization, not memorization.

Genesis Labs Research, 2026
Amellia Mendel, Lisa Adler
"""

from typing import Dict, List, Tuple
import re


# =========================================================================
# STAGE 1: KEYWORD RECOGNITION
# =========================================================================

KEYWORDS = [
    "PRINT", "LET", "GOTO", "IF", "END", "REM",
    "GOSUB", "FOR", "RETURN", "READ", "DIM", "DATA",
]

KEYWORD_TO_ID = {kw: i for i, kw in enumerate(KEYWORDS)}
NUM_KEYWORDS = len(KEYWORDS)


def build_stage1_data() -> List[Dict]:
    """
    Returns keyword recognition data. Each rule class is one keyword.
    Training examples vary case and whitespace. Test examples are held out.
    The label is the keyword index (0-11).
    """
    rules = []
    for kw in KEYWORDS:
        lower = kw.lower()
        mixed = kw[0] + kw[1:].lower()
        train_pairs = [
            {"input": kw, "label": KEYWORD_TO_ID[kw]},
            {"input": f"  {kw}", "label": KEYWORD_TO_ID[kw]},
            {"input": f"{kw}  ", "label": KEYWORD_TO_ID[kw]},
            {"input": f"  {kw}  ", "label": KEYWORD_TO_ID[kw]},
            {"input": lower, "label": KEYWORD_TO_ID[kw]},
            {"input": f" {lower} ", "label": KEYWORD_TO_ID[kw]},
            {"input": mixed, "label": KEYWORD_TO_ID[kw]},
            {"input": f"  {mixed}", "label": KEYWORD_TO_ID[kw]},
            {"input": kw[0].lower() + kw[1:], "label": KEYWORD_TO_ID[kw]},
            {"input": kw[:2] + kw[2:].lower(), "label": KEYWORD_TO_ID[kw]},
            {"input": f" {kw[:2]}{kw[2:].lower()} ", "label": KEYWORD_TO_ID[kw]},
            {"input": kw.lower()[:1].upper() + kw.lower()[1:], "label": KEYWORD_TO_ID[kw]},
            {"input": f"\t{kw}", "label": KEYWORD_TO_ID[kw]},
            {"input": f"{kw}\t", "label": KEYWORD_TO_ID[kw]},
            {"input": f"  {lower}  ", "label": KEYWORD_TO_ID[kw]},
            {"input": kw[0] + kw[1:].lower().upper()[:1] + kw[2:].lower(), "label": KEYWORD_TO_ID[kw]},
            {"input": f"   {kw}   ", "label": KEYWORD_TO_ID[kw]},
            {"input": kw.swapcase(), "label": KEYWORD_TO_ID[kw]},
            {"input": f" {kw.swapcase()} ", "label": KEYWORD_TO_ID[kw]},
            {"input": kw.capitalize(), "label": KEYWORD_TO_ID[kw]},
        ]
        test_pairs = [
            {"input": f"\t {kw} \t", "label": KEYWORD_TO_ID[kw]},
            {"input": f"    {lower}", "label": KEYWORD_TO_ID[kw]},
            {"input": f"{mixed}   ", "label": KEYWORD_TO_ID[kw]},
            {"input": f" {kw.swapcase()}  ", "label": KEYWORD_TO_ID[kw]},
            {"input": f"\t{lower}\t", "label": KEYWORD_TO_ID[kw]},
        ]
        rules.append({
            "rule_id": f"keyword_{kw.lower()}",
            "description": f"Recognize keyword {kw}",
            "train_pairs": train_pairs,
            "test_pairs": test_pairs,
        })
    return rules


# =========================================================================
# STAGE 2: SYNTAX PATTERN RECOGNITION (15-way)
# =========================================================================

PATTERN_NAMES = [
    "print_string_literal",
    "print_comma_zones",
    "print_semicolon_concat",
    "print_blank",
    "print_numeric",
    "print_variable",
    "print_mixed_string_var",
    "let_simple_assign",
    "let_arithmetic",
    "let_builtin_function",
    "goto_unconditional",
    "if_then_true",
    "if_then_false",
    "end_halt",
    "rem_comment",
]

PATTERN_TO_ID = {name: i for i, name in enumerate(PATTERN_NAMES)}
NUM_PATTERNS = len(PATTERN_NAMES)


def build_stage2_data() -> List[Dict]:
    """
    Returns syntax pattern recognition data. Each rule class is one syntax pattern.
    Training examples use varying values. Test examples are held out with novel values.
    The label is the pattern index (0-14).
    """
    rules = []

    # PRINT string literal
    rules.append({
        "rule_id": "print_string_literal",
        "description": "PRINT with a quoted string literal",
        "train_pairs": [
            {"input": 'line 10: PRINT "Hello world"', "label": 0},
            {"input": 'line 10: PRINT "Goodbye"', "label": 0},
            {"input": 'line 10: PRINT "READY."', "label": 0},
            {"input": 'line 10: PRINT "ABC123"', "label": 0},
            {"input": 'line 20: PRINT "Test string"', "label": 0},
            {"input": 'line 30: PRINT "Foo bar"', "label": 0},
            {"input": 'line 40: PRINT "Welcome"', "label": 0},
            {"input": 'line 50: PRINT "Loading..."', "label": 0},
            {"input": 'line 60: PRINT "Done"', "label": 0},
            {"input": 'line 70: PRINT "Press any key"', "label": 0},
            {"input": 'line 80: PRINT "C64 BASIC"', "label": 0},
            {"input": 'line 90: PRINT "Hello there"', "label": 0},
            {"input": 'line 100: PRINT "Genesis"', "label": 0},
            {"input": 'line 110: PRINT "System ready"', "label": 0},
            {"input": 'line 120: PRINT "Alpha beta"', "label": 0},
            {"input": 'line 130: PRINT "Running"', "label": 0},
            {"input": 'line 140: PRINT "Data loaded"', "label": 0},
            {"input": 'line 150: PRINT "Complete"', "label": 0},
            {"input": 'line 160: PRINT "Started"', "label": 0},
            {"input": 'line 170: PRINT "Initialized"', "label": 0},
        ],
        "test_pairs": [
            {"input": 'line 10: PRINT "Testing"', "label": 0},
            {"input": 'line 10: PRINT "Novel string"', "label": 0},
            {"input": 'line 200: PRINT "Never seen"', "label": 0},
            {"input": 'line 300: PRINT "Holdout"', "label": 0},
            {"input": 'line 400: PRINT "Verify"', "label": 0},
        ],
    })

    # PRINT comma zones
    rules.append({
        "rule_id": "print_comma_zones",
        "description": "PRINT with comma separator outputs values in tab zones",
        "train_pairs": [
            {"input": 'line 10: PRINT "Hello","world"', "label": 1},
            {"input": 'line 10: PRINT "A","B"', "label": 1},
            {"input": 'line 10: PRINT "X","Y","Z"', "label": 1},
            {"input": 'line 20: PRINT "cat","dog"', "label": 1},
            {"input": 'line 30: PRINT "one","two","three"', "label": 1},
            {"input": 'line 40: PRINT "left","right"', "label": 1},
            {"input": 'line 50: PRINT "up","down"', "label": 1},
            {"input": 'line 60: PRINT "red","blue"', "label": 1},
            {"input": 'line 70: PRINT "yes","no"', "label": 1},
            {"input": 'line 80: PRINT "first","second"', "label": 1},
            {"input": 'line 90: PRINT "alpha","beta"', "label": 1},
            {"input": 'line 100: PRINT "start","end"', "label": 1},
            {"input": 'line 110: PRINT "high","low"', "label": 1},
            {"input": 'line 120: PRINT "north","south"', "label": 1},
            {"input": 'line 130: PRINT "hot","cold"', "label": 1},
            {"input": 'line 140: PRINT "big","small"', "label": 1},
            {"input": 'line 150: PRINT "open","close"', "label": 1},
            {"input": 'line 160: PRINT "win","lose"', "label": 1},
            {"input": 'line 170: PRINT "day","night"', "label": 1},
            {"input": 'line 180: PRINT "fast","slow"', "label": 1},
        ],
        "test_pairs": [
            {"input": 'line 10: PRINT "Cat","Dog"', "label": 1},
            {"input": 'line 10: PRINT "1","2","3"', "label": 1},
            {"input": 'line 200: PRINT "new","old"', "label": 1},
            {"input": 'line 300: PRINT "top","bottom"', "label": 1},
            {"input": 'line 400: PRINT "in","out"', "label": 1},
        ],
    })

    # PRINT semicolon concat
    rules.append({
        "rule_id": "print_semicolon_concat",
        "description": "PRINT with semicolon concatenates values",
        "train_pairs": [
            {"input": 'line 10: PRINT "Hello";"world"', "label": 2},
            {"input": 'line 10: PRINT "A";"B"', "label": 2},
            {"input": 'line 10: PRINT "Good";"bye"', "label": 2},
            {"input": 'line 20: PRINT "sun";"rise"', "label": 2},
            {"input": 'line 30: PRINT "moon";"light"', "label": 2},
            {"input": 'line 40: PRINT "fire";"fly"', "label": 2},
            {"input": 'line 50: PRINT "star";"dust"', "label": 2},
            {"input": 'line 60: PRINT "rain";"bow"', "label": 2},
            {"input": 'line 70: PRINT "snow";"flake"', "label": 2},
            {"input": 'line 80: PRINT "thunder";"storm"', "label": 2},
            {"input": 'line 90: PRINT "water";"fall"', "label": 2},
            {"input": 'line 100: PRINT "earth";"quake"', "label": 2},
            {"input": 'line 110: PRINT "wind";"mill"', "label": 2},
            {"input": 'line 120: PRINT "sand";"stone"', "label": 2},
            {"input": 'line 130: PRINT "day";"break"', "label": 2},
            {"input": 'line 140: PRINT "night";"fall"', "label": 2},
            {"input": 'line 150: PRINT "sea";"shell"', "label": 2},
            {"input": 'line 160: PRINT "blue";"bird"', "label": 2},
            {"input": 'line 170: PRINT "gold";"fish"', "label": 2},
            {"input": 'line 180: PRINT "green";"house"', "label": 2},
        ],
        "test_pairs": [
            {"input": 'line 10: PRINT "Sun";"rise"', "label": 2},
            {"input": 'line 10: PRINT "X";"Y";"Z"', "label": 2},
            {"input": 'line 200: PRINT "black";"out"', "label": 2},
            {"input": 'line 300: PRINT "break";"fast"', "label": 2},
            {"input": 'line 400: PRINT "over";"flow"', "label": 2},
        ],
    })

    # PRINT blank
    rules.append({
        "rule_id": "print_blank",
        "description": "PRINT with no arguments outputs a blank line",
        "train_pairs": [
            {"input": "line 10: PRINT", "label": 3},
            {"input": "line 50: PRINT", "label": 3},
            {"input": "line 20: PRINT", "label": 3},
            {"input": "line 30: PRINT", "label": 3},
            {"input": "line 40: PRINT", "label": 3},
            {"input": "line 60: PRINT", "label": 3},
            {"input": "line 70: PRINT", "label": 3},
            {"input": "line 80: PRINT", "label": 3},
            {"input": "line 90: PRINT", "label": 3},
            {"input": "line 100: PRINT", "label": 3},
            {"input": "line 110: PRINT", "label": 3},
            {"input": "line 120: PRINT", "label": 3},
            {"input": "line 130: PRINT", "label": 3},
            {"input": "line 140: PRINT", "label": 3},
            {"input": "line 150: PRINT", "label": 3},
            {"input": "line 160: PRINT", "label": 3},
            {"input": "line 170: PRINT", "label": 3},
            {"input": "line 180: PRINT", "label": 3},
            {"input": "line 190: PRINT", "label": 3},
            {"input": "line 200: PRINT", "label": 3},
        ],
        "test_pairs": [
            {"input": "line 250: PRINT", "label": 3},
            {"input": "line 300: PRINT", "label": 3},
            {"input": "line 350: PRINT", "label": 3},
            {"input": "line 400: PRINT", "label": 3},
            {"input": "line 500: PRINT", "label": 3},
        ],
    })

    # PRINT numeric
    rules.append({
        "rule_id": "print_numeric",
        "description": "PRINT with a numeric literal displays the number",
        "train_pairs": [
            {"input": "line 10: PRINT 42", "label": 4},
            {"input": "line 10: PRINT 3.14", "label": 4},
            {"input": "line 10: PRINT 0", "label": 4},
            {"input": "line 20: PRINT 100", "label": 4},
            {"input": "line 30: PRINT 7", "label": 4},
            {"input": "line 40: PRINT 255", "label": 4},
            {"input": "line 50: PRINT 1024", "label": 4},
            {"input": "line 60: PRINT 999", "label": 4},
            {"input": "line 70: PRINT 1", "label": 4},
            {"input": "line 80: PRINT 65535", "label": 4},
            {"input": "line 90: PRINT 2.718", "label": 4},
            {"input": "line 100: PRINT 9.81", "label": 4},
            {"input": "line 110: PRINT 6.28", "label": 4},
            {"input": "line 120: PRINT 12345", "label": 4},
            {"input": "line 130: PRINT 8", "label": 4},
            {"input": "line 140: PRINT 16", "label": 4},
            {"input": "line 150: PRINT 32", "label": 4},
            {"input": "line 160: PRINT 64", "label": 4},
            {"input": "line 170: PRINT 128", "label": 4},
            {"input": "line 180: PRINT 256", "label": 4},
        ],
        "test_pairs": [
            {"input": "line 10: PRINT 99", "label": 4},
            {"input": "line 200: PRINT 512", "label": 4},
            {"input": "line 300: PRINT 77", "label": 4},
            {"input": "line 400: PRINT 1.41", "label": 4},
            {"input": "line 500: PRINT 4096", "label": 4},
        ],
    })

    # PRINT variable
    rules.append({
        "rule_id": "print_variable",
        "description": "PRINT with a variable displays its current value",
        "train_pairs": [
            {"input": "line 10: PRINT X (X=5)", "label": 5},
            {"input": "line 10: PRINT Y (Y=10)", "label": 5},
            {"input": "line 10: PRINT A (A=3.5)", "label": 5},
            {"input": "line 20: PRINT B (B=7)", "label": 5},
            {"input": "line 30: PRINT C (C=42)", "label": 5},
            {"input": "line 40: PRINT D (D=0)", "label": 5},
            {"input": "line 50: PRINT E (E=99)", "label": 5},
            {"input": "line 60: PRINT F (F=1)", "label": 5},
            {"input": "line 70: PRINT G (G=256)", "label": 5},
            {"input": "line 80: PRINT H (H=3.14)", "label": 5},
            {"input": "line 90: PRINT I (I=100)", "label": 5},
            {"input": "line 100: PRINT J (J=50)", "label": 5},
            {"input": "line 110: PRINT K (K=25)", "label": 5},
            {"input": "line 120: PRINT L (L=12)", "label": 5},
            {"input": "line 130: PRINT M (M=8)", "label": 5},
            {"input": "line 140: PRINT P (P=64)", "label": 5},
            {"input": "line 150: PRINT Q (Q=128)", "label": 5},
            {"input": "line 160: PRINT R (R=2.5)", "label": 5},
            {"input": "line 170: PRINT S (S=33)", "label": 5},
            {"input": "line 180: PRINT T (T=77)", "label": 5},
        ],
        "test_pairs": [
            {"input": "line 10: PRINT Z (Z=7)", "label": 5},
            {"input": "line 10: PRINT N (N=100)", "label": 5},
            {"input": "line 200: PRINT U (U=55)", "label": 5},
            {"input": "line 300: PRINT V (V=13)", "label": 5},
            {"input": "line 400: PRINT W (W=91)", "label": 5},
        ],
    })

    # PRINT mixed string and variable
    rules.append({
        "rule_id": "print_mixed_string_var",
        "description": "PRINT with string literal and variable",
        "train_pairs": [
            {"input": 'line 10: PRINT "X=",X (X=5)', "label": 6},
            {"input": 'line 10: PRINT "Score:",S (S=100)', "label": 6},
            {"input": 'line 10: PRINT "Total=",T (T=42)', "label": 6},
            {"input": 'line 20: PRINT "A=",A (A=7)', "label": 6},
            {"input": 'line 30: PRINT "Result:",R (R=99)', "label": 6},
            {"input": 'line 40: PRINT "Count=",C (C=10)', "label": 6},
            {"input": 'line 50: PRINT "Value:",V (V=3)', "label": 6},
            {"input": 'line 60: PRINT "Sum=",S (S=55)', "label": 6},
            {"input": 'line 70: PRINT "Avg:",A (A=8.5)', "label": 6},
            {"input": 'line 80: PRINT "Max=",M (M=1000)', "label": 6},
            {"input": 'line 90: PRINT "Min:",N (N=0)', "label": 6},
            {"input": 'line 100: PRINT "Len=",L (L=25)', "label": 6},
            {"input": 'line 110: PRINT "Idx:",I (I=4)', "label": 6},
            {"input": 'line 120: PRINT "Pos=",P (P=16)', "label": 6},
            {"input": 'line 130: PRINT "Temp:",T (T=72)', "label": 6},
            {"input": 'line 140: PRINT "Rate=",R (R=5.5)', "label": 6},
            {"input": 'line 150: PRINT "Time:",T (T=30)', "label": 6},
            {"input": 'line 160: PRINT "Dist=",D (D=100)', "label": 6},
            {"input": 'line 170: PRINT "Speed:",S (S=60)', "label": 6},
            {"input": 'line 180: PRINT "Mass=",M (M=9.8)', "label": 6},
        ],
        "test_pairs": [
            {"input": 'line 10: PRINT "Value:",V (V=88)', "label": 6},
            {"input": 'line 10: PRINT "N=",N (N=3)', "label": 6},
            {"input": 'line 200: PRINT "Pwr=",P (P=44)', "label": 6},
            {"input": 'line 300: PRINT "Cnt:",C (C=17)', "label": 6},
            {"input": 'line 400: PRINT "Err=",E (E=0)', "label": 6},
        ],
    })

    # LET simple assignment
    rules.append({
        "rule_id": "let_simple_assign",
        "description": "LET assigns a numeric value to a variable",
        "train_pairs": [
            {"input": "line 10: LET X=5", "label": 7},
            {"input": "line 10: LET Y=3.14", "label": 7},
            {"input": "line 10: LET A=0", "label": 7},
            {"input": "line 10: LET Z=100", "label": 7},
            {"input": "line 20: LET B=7", "label": 7},
            {"input": "line 30: LET C=42", "label": 7},
            {"input": "line 40: LET D=1", "label": 7},
            {"input": "line 50: LET E=255", "label": 7},
            {"input": "line 60: LET F=999", "label": 7},
            {"input": "line 70: LET G=2.718", "label": 7},
            {"input": "line 80: LET H=50", "label": 7},
            {"input": "line 90: LET I=10", "label": 7},
            {"input": "line 100: LET J=64", "label": 7},
            {"input": "line 110: LET K=128", "label": 7},
            {"input": "line 120: LET L=16", "label": 7},
            {"input": "line 130: LET M=32", "label": 7},
            {"input": "line 140: LET P=8", "label": 7},
            {"input": "line 150: LET Q=4", "label": 7},
            {"input": "line 160: LET R=2", "label": 7},
            {"input": "line 170: LET S=1.5", "label": 7},
        ],
        "test_pairs": [
            {"input": "line 10: LET N=42", "label": 7},
            {"input": "line 10: LET B=2.718", "label": 7},
            {"input": "line 200: LET T=99", "label": 7},
            {"input": "line 300: LET U=0", "label": 7},
            {"input": "line 400: LET V=77", "label": 7},
        ],
    })

    # LET arithmetic
    rules.append({
        "rule_id": "let_arithmetic",
        "description": "LET with arithmetic expression",
        "train_pairs": [
            {"input": "line 10: LET Y=X^2 (X=5)", "label": 8},
            {"input": "line 10: LET Z=A+B (A=3,B=7)", "label": 8},
            {"input": "line 10: LET R=X*2 (X=4)", "label": 8},
            {"input": "line 10: LET D=X-Y (X=10,Y=3)", "label": 8},
            {"input": "line 20: LET E=A*B (A=5,B=6)", "label": 8},
            {"input": "line 30: LET F=X/2 (X=8)", "label": 8},
            {"input": "line 40: LET G=A+1 (A=9)", "label": 8},
            {"input": "line 50: LET H=X*X (X=3)", "label": 8},
            {"input": "line 60: LET I=A-B (A=20,B=5)", "label": 8},
            {"input": "line 70: LET J=X+Y (X=11,Y=22)", "label": 8},
            {"input": "line 80: LET K=N*3 (N=7)", "label": 8},
            {"input": "line 90: LET L=X^3 (X=2)", "label": 8},
            {"input": "line 100: LET M=A/B (A=10,B=2)", "label": 8},
            {"input": "line 110: LET P=X+10 (X=5)", "label": 8},
            {"input": "line 120: LET Q=A*A (A=4)", "label": 8},
            {"input": "line 130: LET R=X-1 (X=100)", "label": 8},
            {"input": "line 140: LET S=A+B+C (A=1,B=2,C=3)", "label": 8},
            {"input": "line 150: LET T=X*Y (X=6,Y=7)", "label": 8},
            {"input": "line 160: LET U=N+N (N=15)", "label": 8},
            {"input": "line 170: LET V=A^2 (A=10)", "label": 8},
        ],
        "test_pairs": [
            {"input": "line 10: LET Q=A+1 (A=9)", "label": 8},
            {"input": "line 10: LET P=X*X (X=3)", "label": 8},
            {"input": "line 200: LET W=A-B (A=50,B=25)", "label": 8},
            {"input": "line 300: LET Y=X/4 (X=20)", "label": 8},
            {"input": "line 400: LET Z=N+M (N=8,M=12)", "label": 8},
        ],
    })

    # LET builtin function
    rules.append({
        "rule_id": "let_builtin_function",
        "description": "LET with a built-in function",
        "train_pairs": [
            {"input": "line 10: LET Z=SQR(9)", "label": 9},
            {"input": "line 10: LET A=ABS(-5)", "label": 9},
            {"input": "line 10: LET B=INT(3.7)", "label": 9},
            {"input": "line 20: LET C=SQR(16)", "label": 9},
            {"input": "line 30: LET D=ABS(-12)", "label": 9},
            {"input": "line 40: LET E=INT(7.9)", "label": 9},
            {"input": "line 50: LET F=SQR(25)", "label": 9},
            {"input": "line 60: LET G=ABS(-100)", "label": 9},
            {"input": "line 70: LET H=INT(2.1)", "label": 9},
            {"input": "line 80: LET I=SQR(4)", "label": 9},
            {"input": "line 90: LET J=ABS(-1)", "label": 9},
            {"input": "line 100: LET K=INT(9.99)", "label": 9},
            {"input": "line 110: LET L=SQR(36)", "label": 9},
            {"input": "line 120: LET M=ABS(-50)", "label": 9},
            {"input": "line 130: LET N=INT(4.5)", "label": 9},
            {"input": "line 140: LET P=SQR(49)", "label": 9},
            {"input": "line 150: LET Q=ABS(-7)", "label": 9},
            {"input": "line 160: LET R=INT(1.1)", "label": 9},
            {"input": "line 170: LET S=SQR(64)", "label": 9},
            {"input": "line 180: LET T=ABS(-33)", "label": 9},
        ],
        "test_pairs": [
            {"input": "line 10: LET C=SQR(81)", "label": 9},
            {"input": "line 10: LET D=ABS(-99)", "label": 9},
            {"input": "line 200: LET U=INT(5.5)", "label": 9},
            {"input": "line 300: LET V=SQR(100)", "label": 9},
            {"input": "line 400: LET W=ABS(-42)", "label": 9},
        ],
    })

    # GOTO unconditional
    rules.append({
        "rule_id": "goto_unconditional",
        "description": "GOTO jumps to specified line number",
        "train_pairs": [
            {"input": "line 10: GOTO 100", "label": 10},
            {"input": "line 50: GOTO 10", "label": 10},
            {"input": "line 20: GOTO 200", "label": 10},
            {"input": "line 30: GOTO 500", "label": 10},
            {"input": "line 40: GOTO 40", "label": 10},
            {"input": "line 60: GOTO 300", "label": 10},
            {"input": "line 70: GOTO 150", "label": 10},
            {"input": "line 80: GOTO 80", "label": 10},
            {"input": "line 90: GOTO 50", "label": 10},
            {"input": "line 100: GOTO 10", "label": 10},
            {"input": "line 110: GOTO 250", "label": 10},
            {"input": "line 120: GOTO 400", "label": 10},
            {"input": "line 130: GOTO 30", "label": 10},
            {"input": "line 140: GOTO 700", "label": 10},
            {"input": "line 150: GOTO 60", "label": 10},
            {"input": "line 160: GOTO 160", "label": 10},
            {"input": "line 170: GOTO 90", "label": 10},
            {"input": "line 180: GOTO 1000", "label": 10},
            {"input": "line 190: GOTO 20", "label": 10},
            {"input": "line 200: GOTO 350", "label": 10},
        ],
        "test_pairs": [
            {"input": "line 35: GOTO 550", "label": 10},
            {"input": "line 10: GOTO 40", "label": 10},
            {"input": "line 250: GOTO 800", "label": 10},
            {"input": "line 300: GOTO 100", "label": 10},
            {"input": "line 400: GOTO 999", "label": 10},
        ],
    })

    # IF/THEN true
    rules.append({
        "rule_id": "if_then_true",
        "description": "IF condition is true, execute THEN clause",
        "train_pairs": [
            {"input": "line 10: IF X>3 THEN GOTO 100 (X=5)", "label": 11},
            {"input": "line 10: IF A=1 THEN GOTO 50 (A=1)", "label": 11},
            {"input": "line 10: IF Y<10 THEN GOTO 200 (Y=3)", "label": 11},
            {"input": "line 20: IF B>0 THEN GOTO 80 (B=5)", "label": 11},
            {"input": "line 30: IF C=7 THEN GOTO 120 (C=7)", "label": 11},
            {"input": "line 40: IF D<100 THEN GOTO 60 (D=50)", "label": 11},
            {"input": "line 50: IF E>=5 THEN GOTO 300 (E=5)", "label": 11},
            {"input": "line 60: IF F>1 THEN GOTO 90 (F=10)", "label": 11},
            {"input": "line 70: IF G=0 THEN GOTO 400 (G=0)", "label": 11},
            {"input": "line 80: IF H<50 THEN GOTO 110 (H=25)", "label": 11},
            {"input": "line 90: IF I>10 THEN GOTO 500 (I=100)", "label": 11},
            {"input": "line 100: IF J=99 THEN GOTO 150 (J=99)", "label": 11},
            {"input": "line 110: IF K<1000 THEN GOTO 200 (K=500)", "label": 11},
            {"input": "line 120: IF L>=10 THEN GOTO 250 (L=10)", "label": 11},
            {"input": "line 130: IF M>0 THEN GOTO 70 (M=1)", "label": 11},
            {"input": "line 140: IF N<>0 THEN GOTO 80 (N=7)", "label": 11},
            {"input": "line 150: IF P=42 THEN GOTO 600 (P=42)", "label": 11},
            {"input": "line 160: IF Q>5 THEN GOTO 350 (Q=8)", "label": 11},
            {"input": "line 170: IF R<20 THEN GOTO 100 (R=15)", "label": 11},
            {"input": "line 180: IF S>=1 THEN GOTO 50 (S=1)", "label": 11},
        ],
        "test_pairs": [
            {"input": "line 10: IF Z>=5 THEN GOTO 300 (Z=5)", "label": 11},
            {"input": "line 10: IF N<>0 THEN GOTO 80 (N=7)", "label": 11},
            {"input": "line 200: IF T>0 THEN GOTO 400 (T=3)", "label": 11},
            {"input": "line 300: IF U=1 THEN GOTO 50 (U=1)", "label": 11},
            {"input": "line 400: IF V<99 THEN GOTO 700 (V=50)", "label": 11},
        ],
    })

    # IF/THEN false
    rules.append({
        "rule_id": "if_then_false",
        "description": "IF condition is false, skip to next line",
        "train_pairs": [
            {"input": "line 10: IF X>3 THEN GOTO 100 (X=1)", "label": 12},
            {"input": "line 10: IF A=5 THEN GOTO 50 (A=3)", "label": 12},
            {"input": "line 10: IF Y<0 THEN GOTO 200 (Y=10)", "label": 12},
            {"input": "line 20: IF B>100 THEN GOTO 80 (B=5)", "label": 12},
            {"input": "line 30: IF C=7 THEN GOTO 120 (C=3)", "label": 12},
            {"input": "line 40: IF D<0 THEN GOTO 60 (D=50)", "label": 12},
            {"input": "line 50: IF E>=10 THEN GOTO 300 (E=5)", "label": 12},
            {"input": "line 60: IF F>100 THEN GOTO 90 (F=10)", "label": 12},
            {"input": "line 70: IF G=1 THEN GOTO 400 (G=0)", "label": 12},
            {"input": "line 80: IF H<5 THEN GOTO 110 (H=25)", "label": 12},
            {"input": "line 90: IF I>1000 THEN GOTO 500 (I=100)", "label": 12},
            {"input": "line 100: IF J=99 THEN GOTO 150 (J=50)", "label": 12},
            {"input": "line 110: IF K<0 THEN GOTO 200 (K=500)", "label": 12},
            {"input": "line 120: IF L>=100 THEN GOTO 250 (L=10)", "label": 12},
            {"input": "line 130: IF M>99 THEN GOTO 70 (M=1)", "label": 12},
            {"input": "line 140: IF N=0 THEN GOTO 80 (N=7)", "label": 12},
            {"input": "line 150: IF P=42 THEN GOTO 600 (P=41)", "label": 12},
            {"input": "line 160: IF Q>50 THEN GOTO 350 (Q=8)", "label": 12},
            {"input": "line 170: IF R<0 THEN GOTO 100 (R=15)", "label": 12},
            {"input": "line 180: IF S>=100 THEN GOTO 50 (S=1)", "label": 12},
        ],
        "test_pairs": [
            {"input": "line 10: IF Z>100 THEN GOTO 300 (Z=5)", "label": 12},
            {"input": "line 10: IF N=0 THEN GOTO 80 (N=7)", "label": 12},
            {"input": "line 200: IF T>99 THEN GOTO 400 (T=3)", "label": 12},
            {"input": "line 300: IF U=5 THEN GOTO 50 (U=1)", "label": 12},
            {"input": "line 400: IF V<0 THEN GOTO 700 (V=50)", "label": 12},
        ],
    })

    # END halt
    rules.append({
        "rule_id": "end_halt",
        "description": "END halts program execution",
        "train_pairs": [
            {"input": "line 10: END", "label": 13},
            {"input": "line 100: END", "label": 13},
            {"input": "line 20: END", "label": 13},
            {"input": "line 30: END", "label": 13},
            {"input": "line 40: END", "label": 13},
            {"input": "line 50: END", "label": 13},
            {"input": "line 60: END", "label": 13},
            {"input": "line 70: END", "label": 13},
            {"input": "line 80: END", "label": 13},
            {"input": "line 90: END", "label": 13},
            {"input": "line 110: END", "label": 13},
            {"input": "line 120: END", "label": 13},
            {"input": "line 130: END", "label": 13},
            {"input": "line 140: END", "label": 13},
            {"input": "line 150: END", "label": 13},
            {"input": "line 160: END", "label": 13},
            {"input": "line 170: END", "label": 13},
            {"input": "line 180: END", "label": 13},
            {"input": "line 190: END", "label": 13},
            {"input": "line 200: END", "label": 13},
        ],
        "test_pairs": [
            {"input": "line 250: END", "label": 13},
            {"input": "line 300: END", "label": 13},
            {"input": "line 500: END", "label": 13},
            {"input": "line 750: END", "label": 13},
            {"input": "line 999: END", "label": 13},
        ],
    })

    # REM comment
    rules.append({
        "rule_id": "rem_comment",
        "description": "REM is a comment, no operation",
        "train_pairs": [
            {"input": "line 10: REM This is a comment", "label": 14},
            {"input": "line 50: REM Initialize variables", "label": 14},
            {"input": "line 20: REM Main loop start", "label": 14},
            {"input": "line 30: REM Calculate result", "label": 14},
            {"input": "line 40: REM Print output", "label": 14},
            {"input": "line 60: REM End of program", "label": 14},
            {"input": "line 70: REM Set up screen", "label": 14},
            {"input": "line 80: REM Load data", "label": 14},
            {"input": "line 90: REM Save state", "label": 14},
            {"input": "line 100: REM Check bounds", "label": 14},
            {"input": "line 110: REM Error handler", "label": 14},
            {"input": "line 120: REM Sorting routine", "label": 14},
            {"input": "line 130: REM Input section", "label": 14},
            {"input": "line 140: REM Output section", "label": 14},
            {"input": "line 150: REM Subroutine call", "label": 14},
            {"input": "line 160: REM Variable setup", "label": 14},
            {"input": "line 170: REM Loop counter", "label": 14},
            {"input": "line 180: REM Boundary check", "label": 14},
            {"input": "line 190: REM Done processing", "label": 14},
            {"input": "line 200: REM Return to caller", "label": 14},
        ],
        "test_pairs": [
            {"input": "line 30: REM Loop start", "label": 14},
            {"input": "line 250: REM Debug info", "label": 14},
            {"input": "line 300: REM Placeholder", "label": 14},
            {"input": "line 400: REM Unused", "label": 14},
            {"input": "line 500: REM Testing only", "label": 14},
        ],
    })

    return rules


# =========================================================================
# STAGE 3: FULL STATEMENT PROCESSING
# Uses Stage 2 data but adds template rendering verification.
# The rote_data_generator output strings become the ground truth.
# =========================================================================

def build_stage3_data() -> List[Dict]:
    """
    Returns full statement processing data. Same inputs as Stage 2 but
    paired with expected rendered output strings from the template engine.
    The label is still the pattern index. The output field is the expected
    rendered string for template verification.
    """
    from rote_data_generator import build_rule_classes
    rote_rules = build_rule_classes()
    # The Stage 3 model starts from Stage 2's weights, so anything Stage 2
    # trained on has already been seen. Such lines cannot serve as held-out.
    seen_in_stage2 = {p["input"] for r in build_stage2_data() for p in r["train_pairs"]}
    stage3_rules = []
    for i, rule in enumerate(rote_rules):
        s3_train = []
        for pair in rule["train_pairs"]:
            s3_train.append({
                "input": pair["input"],
                "label": i,
                "output": pair["output"],
            })
        s3_test = []
        for pair in rule["test_pairs"]:
            if pair["input"] in seen_in_stage2:
                continue
            s3_test.append({
                "input": pair["input"],
                "label": i,
                "output": pair["output"],
            })
        stage3_rules.append({
            "rule_id": rule["rule_id"],
            "description": rule["description"],
            "train_pairs": s3_train,
            "test_pairs": s3_test,
        })
    return stage3_rules


# =========================================================================
# SUMMARY
# =========================================================================

def summary():
    """Print data summary for all stages."""
    print("STAGE 1: Keyword Recognition")
    s1 = build_stage1_data()
    t1_train = sum(len(r["train_pairs"]) for r in s1)
    t1_test = sum(len(r["test_pairs"]) for r in s1)
    print(f"  {len(s1)} classes, {t1_train} train, {t1_test} test")
    for r in s1:
        print(f"    {r['rule_id']:25s}  train:{len(r['train_pairs']):3d}  test:{len(r['test_pairs']):3d}")

    print("\nSTAGE 2: Syntax Pattern Recognition")
    s2 = build_stage2_data()
    t2_train = sum(len(r["train_pairs"]) for r in s2)
    t2_test = sum(len(r["test_pairs"]) for r in s2)
    print(f"  {len(s2)} classes, {t2_train} train, {t2_test} test")
    for r in s2:
        print(f"    {r['rule_id']:25s}  train:{len(r['train_pairs']):3d}  test:{len(r['test_pairs']):3d}")

    print("\nSTAGE 3: Full Statement Processing")
    s3 = build_stage3_data()
    t3_train = sum(len(r["train_pairs"]) for r in s3)
    t3_test = sum(len(r["test_pairs"]) for r in s3)
    print(f"  {len(s3)} classes, {t3_train} train, {t3_test} test")


# =========================================================================
# SELF-TESTS
# =========================================================================

def test_stage1_coverage():
    rules = build_stage1_data()
    assert len(rules) == 12, f"Expected 12 keyword classes, got {len(rules)}"
    for r in rules:
        assert len(r["train_pairs"]) >= 20, f"{r['rule_id']} has <20 train pairs"
        assert len(r["test_pairs"]) >= 5, f"{r['rule_id']} has <5 test pairs"
        train_inputs = {p["input"] for p in r["train_pairs"]}
        for tp in r["test_pairs"]:
            assert tp["input"] not in train_inputs, f"Test leak in {r['rule_id']}"
    print("PASS: test_stage1_coverage")


def test_stage2_coverage():
    rules = build_stage2_data()
    assert len(rules) == 15, f"Expected 15 pattern classes, got {len(rules)}"
    for r in rules:
        assert len(r["train_pairs"]) >= 20, f"{r['rule_id']} has <20 train pairs"
        assert len(r["test_pairs"]) >= 5, f"{r['rule_id']} has <5 test pairs"
        train_inputs = {p["input"] for p in r["train_pairs"]}
        for tp in r["test_pairs"]:
            assert tp["input"] not in train_inputs, f"Test leak in {r['rule_id']}: {tp['input']!r}"
    print("PASS: test_stage2_coverage")


def test_no_leak_across_rules_or_stages():
    """No held-out line may appear in ANY training set the same weights have seen."""
    s1, s2, s3 = build_stage1_data(), build_stage2_data(), build_stage3_data()
    for name, rules, earlier in (("stage1", s1, []), ("stage2", s2, []), ("stage3", s3, s2)):
        seen = {p["input"] for r in rules + earlier for p in r["train_pairs"]}
        for r in rules:
            assert r["test_pairs"], f"{name}/{r['rule_id']} has no held-out pairs left"
            for tp in r["test_pairs"]:
                assert tp["input"] not in seen, f"{name}/{r['rule_id']} held-out leak: {tp['input']!r}"
    print("PASS: test_no_leak_across_rules_or_stages")


def test_stage3_has_outputs():
    rules = build_stage3_data()
    for r in rules:
        for p in r["train_pairs"] + r["test_pairs"]:
            assert "output" in p, f"Missing output in {r['rule_id']}"
            assert "label" in p, f"Missing label in {r['rule_id']}"
    print("PASS: test_stage3_has_outputs")


def test_no_label_collision():
    rules = build_stage2_data()
    for r in rules:
        labels = set(p["label"] for p in r["train_pairs"])
        assert len(labels) == 1, f"Mixed labels in {r['rule_id']}: {labels}"
    print("PASS: test_no_label_collision")


if __name__ == "__main__":
    test_stage1_coverage()
    test_stage2_coverage()
    test_no_leak_across_rules_or_stages()
    test_stage3_has_outputs()
    test_no_label_collision()
    print("\nAll curriculum_data tests passed.\n")
    summary()
