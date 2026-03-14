"""
Template-based problem generator — guaranteed parse success, no LLM needed.

When the LLM proposer fails (model thinks too long, eats token budget),
this generates problems by filling templates with random numbers.
Always produces valid PROBLEM/ANSWER pairs, verified against ground truth.
"""

import random
import math
from .proposer import Problem
from .seed_problems import _NUM_TEST


def _gcd(a, b):
    while b: a, b = b, a % b
    return a


def _is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0: return False
    return True


def generate_math_problem(difficulty: str = "medium") -> Problem:
    """Generate a random math problem with guaranteed correct answer."""
    templates = {
        "easy": [
            _linear_eq,
            _percentage,
            _area_rectangle,
            _speed_time_distance,
        ],
        "medium": [
            _quadratic_integer,
            _combinatorics_perm,
            _modular_arithmetic,
            _sequence_sum,
            _gcd_problem,
        ],
        "hard": [
            _prime_count_range,
            _divisor_count,
            _circular_arrangement,
            _probability_fraction,
        ],
    }
    fns = templates.get(difficulty, templates["medium"])
    # Try up to 10 times to get a valid problem
    for _ in range(10):
        try:
            fn = random.choice(fns)
            prompt, answer = fn()
            if answer is not None and str(answer).strip():
                return Problem(
                    domain="math",
                    difficulty=difficulty,
                    prompt=prompt,
                    problem_code=f"expected = {repr(str(answer))}",
                    test_code=_NUM_TEST,
                    metadata={"expected_answer": str(answer)},
                )
        except Exception:
            continue
    # Final fallback
    a, b = random.randint(2, 15), random.randint(1, 50)
    return Problem(
        domain="math", difficulty="easy",
        prompt=f"What is {a} × {b}?",
        problem_code=f"expected = '{a*b}'",
        test_code=_NUM_TEST,
        metadata={"expected_answer": str(a * b)},
    )


def generate_code_problem(difficulty: str = "medium") -> Problem:
    """Generate a 'what does this print?' code problem."""
    templates = [_code_sum_range, _code_fibonacci, _code_count_evens,
                 _code_factorial, _code_list_ops]
    for _ in range(10):
        try:
            fn = random.choice(templates)
            prompt, answer = fn()
            return Problem(
                domain="code", difficulty=difficulty,
                prompt=prompt,
                problem_code=f"expected = {repr(str(answer))}",
                test_code=_NUM_TEST,
                metadata={"expected_answer": str(answer)},
            )
        except Exception:
            continue
    return Problem(
        domain="code", difficulty="easy",
        prompt="What does `print(2 + 2)` output? Give only the number.",
        problem_code="expected = '4'",
        test_code=_NUM_TEST,
        metadata={"expected_answer": "4"},
    )


def generate_logic_problem(difficulty: str = "medium") -> Problem:
    """Generate a logic/word problem."""
    templates = [_workers_days, _age_problem, _coin_problem, _rate_problem]
    for _ in range(10):
        try:
            fn = random.choice(templates)
            prompt, answer = fn()
            return Problem(
                domain="logic", difficulty=difficulty,
                prompt=prompt,
                problem_code=f"expected = {repr(str(answer))}",
                test_code=_NUM_TEST,
                metadata={"expected_answer": str(answer)},
            )
        except Exception:
            continue
    return Problem(
        domain="logic", difficulty="easy",
        prompt="If 2 cats catch 2 mice in 2 minutes, how many cats are needed to catch 6 mice in 6 minutes?",
        problem_code="expected = '2'",
        test_code=_NUM_TEST,
        metadata={"expected_answer": "2"},
    )


# ── MATH TEMPLATES ──────────────────────────────────────────────────────────

def _linear_eq():
    a = random.randint(2, 9)
    x = random.randint(1, 20)
    b = random.randint(1, 30)
    c = a * x + b
    return f"Solve for x: {a}x + {b} = {c}. Give only the integer answer.", x

def _percentage():
    total = random.choice([50, 80, 100, 120, 200, 250])
    pct = random.choice([10, 15, 20, 25, 30, 40, 50])
    answer = total * pct // 100
    return f"What is {pct}% of {total}?", answer

def _area_rectangle():
    l, w = random.randint(3, 20), random.randint(3, 20)
    choice = random.choice(["area", "perimeter"])
    if choice == "area":
        return f"A rectangle has length {l} and width {w}. What is its area?", l * w
    else:
        return f"A rectangle has length {l} and width {w}. What is its perimeter?", 2*(l+w)

def _speed_time_distance():
    s = random.choice([40, 50, 60, 80, 100])
    t = random.choice([2, 3, 4, 5])
    return f"A car travels at {s} km/h for {t} hours. How many km does it travel?", s * t

def _quadratic_integer():
    r1, r2 = random.randint(1, 8), random.randint(1, 8)
    b, c = -(r1+r2), r1*r2
    sign_b = f"- {abs(b)}" if b < 0 else f"+ {b}"
    sign_c = f"- {abs(c)}" if c < 0 else f"+ {c}"
    smaller = min(r1, r2)
    return (f"What is the smaller positive root of x² {sign_b}x {sign_c} = 0?", smaller)

def _combinatorics_perm():
    n = random.randint(4, 7)
    r = random.randint(2, min(n, 4))
    # nPr = n! / (n-r)!
    result = math.factorial(n) // math.factorial(n - r)
    return f"How many ways can you arrange {r} books chosen from {n} distinct books?", result

def _modular_arithmetic():
    base = random.choice([2, 3, 5, 7])
    exp = random.randint(6, 12)
    mod = random.choice([5, 7, 9, 11, 13])
    result = pow(base, exp, mod)
    return f"What is the remainder when {base}^{exp} is divided by {mod}?", result

def _sequence_sum():
    start = random.randint(1, 5)
    step = random.randint(1, 4)
    n = random.randint(5, 10)
    # Sum of arithmetic: n/2 * (2*start + (n-1)*step)
    total = n * (2 * start + (n-1) * step) // 2
    last = start + (n-1) * step
    return f"Find the sum of the arithmetic sequence: {start}, {start+step}, ..., {last} ({n} terms).", total

def _gcd_problem():
    a = random.randint(12, 50)
    b = random.randint(6, a-1)
    g = _gcd(a, b)
    return f"What is the GCD of {a} and {b}?", g

def _prime_count_range():
    upper = random.choice([20, 30, 50])
    count = sum(1 for i in range(2, upper+1) if _is_prime(i))
    return f"How many prime numbers are there between 1 and {upper} (inclusive)?", count

def _divisor_count():
    n = random.choice([12, 18, 24, 36, 48, 60, 72, 100])
    count = sum(1 for i in range(1, n+1) if n % i == 0)
    return f"How many positive divisors does {n} have?", count

def _circular_arrangement():
    n = random.randint(4, 7)
    result = math.factorial(n - 1)
    return f"In how many ways can {n} distinct people be seated around a circular table? (Rotations are the same)", result

def _probability_fraction():
    total = random.randint(5, 10)
    fav = random.randint(1, total - 1)
    g = _gcd(fav, total)
    num, den = fav // g, total // g
    answer = f"{num}/{den}" if den > 1 else str(num)
    return f"A bag has {fav} red balls and {total - fav} blue balls. What is the probability of drawing a red ball? Express as a simplified fraction.", answer


# ── CODE TEMPLATES ──────────────────────────────────────────────────────────

def _code_sum_range():
    a = random.randint(1, 5)
    b = random.randint(10, 20)
    answer = sum(range(a, b+1))
    code = f"s = 0\nfor i in range({a}, {b+1}):\n    s += i\nprint(s)"
    return f"What does this code print?\n```python\n{code}\n```\nGive only the number.", answer

def _code_fibonacci():
    n = random.randint(5, 10)
    a, b = 0, 1
    for _ in range(n - 1): a, b = b, a + b
    code = f"def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\nprint(fib({n}))"
    return f"What does this code print?\n```python\n{code}\n```\nGive only the number.", a

def _code_count_evens():
    lst = [random.randint(1, 20) for _ in range(random.randint(5, 9))]
    evens = sum(1 for x in lst if x % 2 == 0)
    lst_str = str(lst)
    code = f"nums = {lst_str}\nprint(sum(1 for x in nums if x % 2 == 0))"
    return f"What does this code print?\n```python\n{code}\n```\nGive only the number.", evens

def _code_factorial():
    n = random.randint(4, 8)
    result = math.factorial(n)
    code = f"import math\nprint(math.factorial({n}))"
    return f"What does this code print?\n```python\n{code}\n```\nGive only the number.", result

def _code_list_ops():
    lst = sorted(random.sample(range(1, 30), random.randint(4, 7)))
    ops = [
        (f"print(sum({lst}))", sum(lst)),
        (f"print(max({lst}) - min({lst}))", max(lst) - min(lst)),
        (f"print(len({lst}))", len(lst)),
    ]
    code, answer = random.choice(ops)
    return f"What does this code print?\n```python\n{code}\n```\nGive only the number.", answer


# ── LOGIC TEMPLATES ─────────────────────────────────────────────────────────

def _workers_days():
    w1 = random.randint(2, 6)
    d1 = random.randint(4, 10)
    w2 = random.choice([i for i in range(2, 12) if (w1 * d1) % i == 0])
    d2 = (w1 * d1) // w2
    return f"If {w1} workers finish a job in {d1} days, how many days would {w2} workers take?", d2

def _age_problem():
    child = random.randint(5, 15)
    mult = random.randint(2, 4)
    parent = child * mult
    years = random.randint(5, 15)
    return (f"Alice is {mult} times Bob's age. Bob is {child}. "
            f"In {years} years, what will be the sum of their ages?"), (child + parent + 2 * years)

def _coin_problem():
    n = random.randint(8, 20)
    val = random.choice([5, 10, 25])
    total = n * val
    return f"You have {n} coins each worth {val} cents. What is the total value in cents?", total

def _rate_problem():
    rate = random.randint(10, 30)
    time = random.randint(2, 8)
    return f"A tap fills a tank at {rate} liters per minute. How many liters in {time} minutes?", rate * time


# ── PUBLIC API ──────────────────────────────────────────────────────────────

def generate_problem(domain: str, difficulty: str = "medium") -> Problem:
    """Generate a problem for the given domain using templates (never fails)."""
    if domain in ("math", "data", "science", "spatial"):
        return generate_math_problem(difficulty)
    elif domain == "code":
        return generate_code_problem(difficulty)
    elif domain == "logic":
        return generate_logic_problem(difficulty)
    else:
        return generate_math_problem(difficulty)
