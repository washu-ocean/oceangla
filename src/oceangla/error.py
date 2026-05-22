from pprint import pformat


def print_unique_conditions(cur: str):
    unique_conditions = [
        row[0]
        for row in cur.execute(
            "SELECT DISTINCT condition FROM subject_activation"
        ).fetchall()
    ]
    print(f"unique conditions:\n{pformat(unique_conditions)}")


def print_unique_tasks(cur: str):
    unique_tasks = [
        row[0]
        for row in cur.execute(
            "SELECT DISTINCT task FROM subject_activation"
        ).fetchall()
    ]
    print(f"unique tasks:\n{pformat(unique_tasks)}")


def print_unique_sessions(cur: str):
    unique_sessions = [
        row[0]
        for row in cur.execute(
            "SELECT DISTINCT session FROM subject_activation"
        ).fetchall()
    ]
    print(f"unique sessions:\n{pformat(unique_sessions)}")


def print_unique_spaces(cur: str):
    unique_spaces = [
        row[0]
        for row in cur.execute(
            "SELECT DISTINCT space FROM subject_activation"
        ).fetchall()
    ]
    print(f"unique spaces:\n{pformat(unique_spaces)}")
