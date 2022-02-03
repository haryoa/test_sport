"""
DO IT package
"""


def task_python_version():
    """
    TESTING
    """
    return {"actions": [["python", "--version"]], "doc": "guguk", "verbosity": 2}


def task_mypy():
    return {
        "actions": [
            [
                "mypy",
                "sport_iseng",
                "--strict",
                "--ignore-missing-imports",
                "--follow-imports=silent",
                "--show-column-numbers",
                "--implicit-reexport",
            ]
        ],
        "doc": " do mypy",
        "verbosity": 2,
    }
