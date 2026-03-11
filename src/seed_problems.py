"""
Seed problems for bootstrapping GRPO training.

These are hand-crafted problems at the right difficulty level:
- Hard enough that a 4B model won't always get them right
- Easy enough that it sometimes will
- Clean numeric answers for reliable verification
"""

from .proposer import Problem

SEED_PROBLEMS = [
    # Medium algebra
    Problem(
        domain="math", difficulty="medium",
        prompt="If 3x + 7 = 28, what is the value of x?",
        problem_code="expected = 7",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "7"},
    ),
    Problem(
        domain="math", difficulty="medium",
        prompt="What is the sum of the first 10 positive integers?",
        problem_code="expected = 55",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "55"},
    ),
    Problem(
        domain="math", difficulty="medium",
        prompt="A rectangle has a length of 12 and a width of 5. What is the perimeter?",
        problem_code="expected = 34",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "34"},
    ),
    Problem(
        domain="math", difficulty="medium",
        prompt="What is the value of 2^8?",
        problem_code="expected = 256",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "256"},
    ),
    Problem(
        domain="math", difficulty="medium",
        prompt="If a car travels at 60 mph for 2.5 hours, how many miles does it travel?",
        problem_code="expected = 150",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "150"},
    ),
    # Harder math
    Problem(
        domain="math", difficulty="hard",
        prompt="What is the remainder when 2^10 is divided by 7?",
        problem_code="expected = 2",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "2"},
    ),
    Problem(
        domain="math", difficulty="hard",
        prompt="How many prime numbers are there between 1 and 30?",
        problem_code="expected = 10",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "10"},
    ),
    Problem(
        domain="math", difficulty="hard",
        prompt="What is the greatest common divisor (GCD) of 48 and 36?",
        problem_code="expected = 12",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "12"},
    ),
    Problem(
        domain="math", difficulty="hard",
        prompt="In how many ways can 5 books be arranged on a shelf?",
        problem_code="expected = 120",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "120"},
    ),
    Problem(
        domain="math", difficulty="hard",
        prompt="What is the sum of all integers from 1 to 100 that are divisible by 3?",
        problem_code="expected = 1683",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "1683"},
    ),
    # Logic / reasoning (not pure math)
    Problem(
        domain="logic", difficulty="medium",
        prompt="If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer 1 for Yes, 0 for No.",
        problem_code="expected = 0",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "0"},
    ),
    Problem(
        domain="logic", difficulty="medium",
        prompt="A sequence follows the rule: each term is the previous term times 2 plus 1. If the first term is 1, what is the 6th term?",
        problem_code="expected = 63",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "63"},
    ),
    Problem(
        domain="logic", difficulty="hard",
        prompt="You have 3 boxes. Box A has 2 red balls, Box B has 2 blue balls, Box C has 1 red and 1 blue. All labels are WRONG. You pick 1 ball from Box A and it's red. What color are the balls actually in Box B? Answer 1 for red, 2 for blue, 3 for mixed.",
        problem_code="expected = 3",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "3"},
    ),
    # Spatial / counting
    Problem(
        domain="spatial", difficulty="medium",
        prompt="A cube has 6 faces. If you paint all faces and then cut it into 27 equal smaller cubes (3x3x3), how many small cubes have exactly 2 painted faces?",
        problem_code="expected = 12",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "12"},
    ),
    Problem(
        domain="spatial", difficulty="medium",
        prompt="How many diagonals does a hexagon have?",
        problem_code="expected = 9",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "9"},
    ),
    # Science / physics
    Problem(
        domain="science", difficulty="medium",
        prompt="An object is dropped from rest. Ignoring air resistance, approximately how far (in meters) does it fall in 3 seconds? Use g=10 m/s^2. Round to nearest integer.",
        problem_code="expected = 45",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "45"},
    ),
    Problem(
        domain="science", difficulty="hard",
        prompt="How many electrons does a neutral carbon atom have?",
        problem_code="expected = 6",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "6"},
    ),
    # Data analysis
    Problem(
        domain="data", difficulty="medium",
        prompt="Given the dataset [2, 4, 4, 4, 5, 5, 7, 9], what is the median?",
        problem_code="expected = 4.5",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "4.5"},
    ),
    Problem(
        domain="data", difficulty="hard",
        prompt="A dataset has values [10, 20, 30, 40, 50]. What is the standard deviation? Round to 1 decimal place.",
        problem_code="expected = 14.1",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "14.1"},
    ),
    # === HARDER PROBLEMS (goldilocks zone for 4B model) ===
    # Number theory
    Problem(
        domain="math", difficulty="hard",
        prompt="What is the last two digits of 7^2025? Give only the number.",
        problem_code="expected = 7",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "7"},
    ),
    Problem(
        domain="math", difficulty="hard",
        prompt="How many positive divisors does 360 have?",
        problem_code="expected = 24",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "24"},
    ),
    Problem(
        domain="math", difficulty="hard",
        prompt="What is the sum of all prime factors of 2310? (Count each prime once)",
        problem_code="expected = 28",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "28"},
    ),
    # Combinatorics
    Problem(
        domain="math", difficulty="hard",
        prompt="How many 4-digit numbers can be formed using digits 1,2,3,4,5 with no repetition?",
        problem_code="expected = 120",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "120"},
    ),
    Problem(
        domain="math", difficulty="hard",
        prompt="In how many ways can 8 people be seated around a circular table? (Rotations are the same)",
        problem_code="expected = 5040",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "5040"},
    ),
    # Probability
    Problem(
        domain="math", difficulty="hard",
        prompt="Two dice are rolled. What is the probability the sum is 7? Express as a fraction like a/b.",
        problem_code="expected = '1/6'",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "1/6"},
    ),
    Problem(
        domain="math", difficulty="hard",
        prompt="A bag has 5 red and 3 blue balls. If 2 balls are drawn without replacement, what is the probability both are red? Express as a fraction like a/b.",
        problem_code="expected = '5/14'",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "5/14"},
    ),
    # Logic / reasoning (harder)
    Problem(
        domain="logic", difficulty="hard",
        prompt="If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?",
        problem_code="expected = 5",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "5"},
    ),
    Problem(
        domain="logic", difficulty="hard",
        prompt="A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?",
        problem_code="expected = 5",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "5"},
    ),
    Problem(
        domain="logic", difficulty="hard",
        prompt="There are 100 lockers in a row, all closed. 100 students walk by. Student 1 opens every locker. Student 2 toggles every 2nd locker. Student 3 toggles every 3rd, etc. How many lockers are open at the end?",
        problem_code="expected = 10",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "10"},
    ),
    # Spatial reasoning (harder)
    Problem(
        domain="spatial", difficulty="hard",
        prompt="A regular tetrahedron has how many edges?",
        problem_code="expected = 6",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "6"},
    ),
    Problem(
        domain="spatial", difficulty="hard",
        prompt="If you fold a standard cross-shaped net into a cube, which face is opposite the center face? Answer with the number of faces that could be opposite: how many valid nets have this property? Actually — simpler: A cube net is a cross shape (1 center + 4 sides + 1 extension). How many distinct nets can fold into a cube? Give the total count.",
        problem_code="expected = 11",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "11"},
    ),
    # Science (harder)
    Problem(
        domain="science", difficulty="hard",
        prompt="What is the speed of light in meters per second, rounded to the nearest million? Give just the number in millions (e.g., 300 for 300 million).",
        problem_code="expected = 300",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "300"},
    ),
    Problem(
        domain="science", difficulty="hard",
        prompt="A projectile is launched straight up at 30 m/s. Using g=10 m/s^2, what is the maximum height in meters?",
        problem_code="expected = 45",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "45"},
    ),
    Problem(
        domain="science", difficulty="hard",
        prompt="In a circuit with a 12V battery and a 4-ohm resistor, what is the current in amperes?",
        problem_code="expected = 3",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "3"},
    ),
    # Planning / multi-step reasoning
    Problem(
        domain="logic", difficulty="hard",
        prompt="You need to measure exactly 4 liters using only a 3-liter jug and a 5-liter jug. What is the minimum number of pour/fill/empty operations needed?",
        problem_code="expected = 6",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "6"},
    ),
    Problem(
        domain="logic", difficulty="hard",
        prompt="Three people check into a hotel room that costs $30. They each pay $10. The manager realizes the room is only $25 and gives $5 to the bellboy to return. The bellboy keeps $2 and returns $1 to each person. Now each person paid $9 (total $27), plus $2 the bellboy kept = $29. Where is the missing $1? Answer 0 if there is no missing dollar, 1 if there is.",
        problem_code="expected = 0",
        test_code="assert str(student_answer).strip() == str(expected).strip()",
        metadata={"expected_answer": "0"},
    ),
]
