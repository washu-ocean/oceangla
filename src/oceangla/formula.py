import logging
import sqlite3
from collections import namedtuple
from enum import Enum, auto
from pprint import pformat

import nibabel as nib
import numpy as np
import pandas as pd

from .config import config

logger = logging.getLogger(__name__)

VALID_FUNCS = ("Z",)


class TokenType(Enum):
    INVALID = auto()
    VAR = auto()
    PLUS = auto()
    MINUS = auto()
    TILDE = auto()
    MUL = auto()
    INTERACT = auto()
    INTERCEPT = auto()
    NUMBER = auto()
    LPAREN = auto()
    RPAREN = auto()
    PIPE = auto()
    ZSCORE = auto()


Token = namedtuple("Token", ["type", "value"])
INTERCEPT_TOKEN = Token(type=TokenType.INTERCEPT, value="1")


def lex_formula_str(formula_str: str) -> list[Token]:
    if "~" not in formula_str:
        raise ValueError(
            "Invalid model spec; must include char '~' to separate dependent from independent variables."
        )
    elif formula_str.count("~") != 1:
        raise ValueError(
            "Invalid model spec; dependent/independent variable separator '~' can only be included once."
        )
    pos = 0

    tokens = []

    def is_var_char(c: str):
        return c.isalnum() or c in "_"

    while pos < len(formula_str):
        if formula_str[pos].isspace():
            pos += 1
        elif formula_str[pos] in "+-*:()~":
            tokens.append(
                Token(
                    {
                        "+": TokenType.PLUS,
                        "-": TokenType.MINUS,
                        "*": TokenType.MUL,
                        ":": TokenType.INTERACT,
                        "(": TokenType.LPAREN,
                        ")": TokenType.RPAREN,
                        "~": TokenType.TILDE,
                    }[formula_str[pos]],
                    formula_str[pos],
                )
            )
            pos += 1
        elif is_var_char(formula_str[pos]):
            varname = ""
            while pos < len(formula_str) and is_var_char(formula_str[pos]):
                varname += formula_str[pos]
                pos += 1
            if varname.isdigit():
                tokens.append(Token(TokenType.NUMBER, varname))
            else:
                tokens.append(Token(TokenType.VAR, varname))
        else:
            tokens.append(Token(TokenType.INVALID, formula_str[pos]))
            pos += 1
    return tokens


def _get_depvars_from_tokens(tokens) -> list[str]:
    """
    Return a list of strings denoting the condition(s) to include in the contrast
    representing the dependent variable in a GLM.

    Strings in the returned list will be prepended with a "+" or "-", which signifies
    they'll be scaled either by 1 or -1 in the contrast.

    Example outputs:

    ["+correct"] -> dependent variable should be all subjects' "correct" statistical map
    at the first level

    ["+correct" "-incorrect"] -> dependent variable is a contrast representing the difference
    between "correct" and "incorrect" conditions for each subject. The contrast map will be made
    if it does not exist

    ["+correct" "+incorrect"] -> dependent variable represents average of "correct" and "incorrect"
    conditions for each subject
    """
    pos = 0
    depvars = []
    while pos < len(tokens):
        if tokens[pos].type in (TokenType.PLUS, TokenType.MINUS):
            if not pos + 1 < len(tokens):
                raise ValueError(
                    f"Cannot have token '{tokens[pos].value}' at end of depvar."
                )
            if not tokens[pos + 1].type == TokenType.VAR:
                raise ValueError(
                    f"Illegal token after '{tokens[pos].value}' : '{tokens[pos + 1].value}'"
                )
            depvars.append(f"{tokens[pos].value}{tokens[pos + 1].value}")
            pos += 2
        elif tokens[pos].type == TokenType.VAR and len(depvars) == 0:
            depvars.append(f"+{tokens[pos].value}")
            pos += 1
        elif tokens[pos].type == TokenType.INTERCEPT:
            depvars.append(f"+{tokens[pos].value}")
            pos += 1
        else:
            raise ValueError(f"Illegal token in depvar: {tokens[pos]}")
    return depvars


class UnexpectedTokenError(Exception):
    def __init__(self, parser):
        super().__init__(f"Unexpected token: {parser.peek()!r}")


def _is_scaled_value_node(node):
    return all(
        (
            isinstance(node, tuple),
            len(node) == 2,
            isinstance(node[0][0], Token)
            and node[0][0].type in (TokenType.PLUS, TokenType.MINUS),
            isinstance(node[0][1], Token) and node[0][1].type == TokenType.NUMBER,
            isinstance(node[1], Token) and node[1].type == TokenType.VAR,
        )
    )


class FormulaParser:
    def __init__(self, tokens):
        self.pos = 0
        if (
            isinstance(tokens, list)
            and len(tokens) > 0
            and isinstance(tokens[0], Token)
        ):
            self.tokens = tokens
        elif isinstance(tokens, str):
            self.tokens = lex_formula_str(tokens)
        else:
            raise TypeError(
                f"`tokens` should be of type str or list[Token], received: {type(tokens)}"
            )
        self.tree = self.parse()

    def __str__(self):
        s = ""
        if self.tree is None:
            return s
        deptree, indeptree = self.tree
        for node in deptree:
            s += f"({node[0][0].value}{node[0][1].value}){node[1].value} "
        s += "~ "
        for node in indeptree:
            if isinstance(node, Token) and node.type == TokenType.INTERCEPT:
                s += "intercept "
            elif _is_scaled_value_node(node):
                s += f"({node[0][0].value}{node[0][1].value}){node[1].value} "
            elif isinstance(node, list) and node[0].type == TokenType.MUL:
                childnodes = [
                    f"({childnode[0][0].value}{childnode[0][1].value}){childnode[1].value}"
                    for childnode in node[1:]
                ]
                interaction_term = ":".join(childnodes)
                s += " ".join([*childnodes, interaction_term])
            elif isinstance(node, list) and node[0].type == TokenType.INTERACT:
                childnodes = [
                    f"({childnode[0][0].value}{childnode[0][1].value}){childnode[1].value}"
                    for childnode in node[1:]
                ]
                interaction_term = ":".join(childnodes)
                s += f" {interaction_term} "
        return s.strip()

    def reset(self):
        self.tokens = self.orig_tokens
        self.pos = 0

    def peek(self):
        return (
            self.tokens[self.pos]
            if self.pos < len(self.tokens)
            else Token(type=TokenType.INVALID, value="")
        )

    def consume(self):
        token = self.peek()
        self.pos += 1
        return token

    def parse(self):
        depvar = self.depvar()
        indepvar = self.indepvar()
        return (depvar, indepvar)

    def depvar(self):
        nodes = []
        nodes.append(self.unscaled_var())
        while self.peek().type != TokenType.TILDE:
            nodes.append(self.scaled_var())
        self.consume()
        return nodes

    def unscaled_var(self):
        if self.peek().type in (TokenType.PLUS, TokenType.MINUS, TokenType.LPAREN):
            return self.scaled_var()
        elif (
            self.peek().type == TokenType.VAR
        ):  # Scale by positive 1 when no scalar present
            op = (
                Token(type=TokenType.PLUS, value="+"),
                Token(type=TokenType.NUMBER, value="1"),
            )
            varname = self.consume()
            return (op, varname)
        else:
            raise UnexpectedTokenError(self)

    def scaled_var(self):
        op = self.scalar()
        if not self.peek().type == TokenType.VAR:
            raise UnexpectedTokenError(self)
        varname = self.consume()
        return (op, varname)

    def scalar(self):
        if self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            sign = self.consume()
            scalar = Token(type=TokenType.NUMBER, value="1")
            return (sign, scalar)
        elif self.peek().type == TokenType.LPAREN:
            self.consume()
            if self.peek().type not in (
                TokenType.PLUS,
                TokenType.MINUS,
                TokenType.NUMBER,
            ):
                raise UnexpectedTokenError(self)
            if self.peek().type == TokenType.NUMBER:
                sign = Token(type=TokenType.PLUS, value="+")
                scalar = self.consume()
            elif self.peek().type in (TokenType.PLUS, TokenType.MINUS):
                sign = self.consume()
                if not self.peek().type == TokenType.NUMBER:
                    raise UnexpectedTokenError(self)
                scalar = self.consume()
            if not self.peek().type == TokenType.RPAREN:
                raise UnexpectedTokenError(self)
            return (sign, scalar)
        else:
            raise UnexpectedTokenError(self)

    def indepvar(self):
        nodes = []
        if self.peek() == Token(type=TokenType.NUMBER, value="1"):
            nodes.append(INTERCEPT_TOKEN)
            self.consume()
        else:
            nodes.append(self.interaction(start=True))
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS, TokenType.LPAREN):
            nodes.append(self.interaction())
        if INTERCEPT_TOKEN not in nodes:
            nodes.insert(0, INTERCEPT_TOKEN)
        return nodes

    def interaction(self, start=False):
        node = self.unscaled_var() if start else self.scaled_var()
        if self.peek().type in (TokenType.MUL, TokenType.INTERACT):
            op = self.consume()
            right = self.unscaled_var()
            node = [op, node, right]
        while isinstance(node, list) and self.peek().type == node[0].type:
            node.append(self.unscaled_var())
        return node


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


@config.joblib_memory.cache
def query_depvar(
    condition, db_path: str, space: str = "fsLR", task: str = None, session: str = None
) -> dict:
    activation = {"space": space}
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        query = f"""
        SELECT path FROM subject_activation
        WHERE subject IN subs_with_all_variables AND (condition='{condition}' OR condition='{condition.replace("_", "-")}')
        AND space='{space}'
        """
        if task is not None:
            query += f"AND task='{task}' "
        if session is not None:
            query += f"AND session='{session}' "
        else:  # Try and get the most common session
            session, _ = cur.execute(
                """ SELECT session, COUNT(session) as frequency FROM subject_activation GROUP BY session ORDER BY frequency DESC LIMIT 1 """
            ).fetchone()
            query += f"AND session='{session}'"
        query += " ORDER BY subject_activation.subject"
        print(f"Running query:\n{query}")
        paths = [row[0] for row in cur.execute(query)]
        try:
            first_img = nib.load(paths[0])
        except IndexError:
            print("Query failed.")
            print_unique_conditions(cur)
            print_unique_sessions(cur)
            print_unique_tasks(cur)
            print_unique_spaces(cur)
            exit()
        print("Loading activation...")
        if len(first_img.dataobj.shape) == 2:  # CIFTI
            activation["type"] = "CIFTI"
            activation["header"] = first_img.header
            activation["nifti_header"] = first_img.nifti_header
            activation["activation"] = np.concatenate(
                [nib.load(path).get_fdata() for path in paths], axis=0
            )
        elif len(first_img.dataobj.shape) == 3:  # NIFTI
            activation["type"] = "NIFTI"
            activation["affine"] = first_img.affine
            activation["header"] = first_img.header
            activation["activation"] = np.concatenate(
                [nib.load(path).get_fdata()[..., np.newaxis] for path in paths], axis=3
            )
        elif len(first_img.dataobj.shape) == 4:  # NIFTI
            activation["type"] = "NIFTI"
            activation["affine"] = first_img.affine
            activation["header"] = first_img.header
            activation["activation"] = np.concatenate(
                [nib.load(path).get_fdata() for path in paths], axis=3
            )
        else:
            raise ValueError(
                f"Number of axes for image at path {paths[0]} must be 2 (for CIFTI) 3, or 4 (for NIFTI), but contains {len(first_img.dataobj.shape)}"
            )
        return activation


def get_activation_and_design_matrix(
    formula: str,
    db_path: str,
    space: str = "fsLR",
    task: str = None,
    session: str = None,
) -> tuple[pd.DataFrame, dict]:
    deptree, indeptree = FormulaParser(formula).tree[0], FormulaParser(formula).tree[1]
    columns_to_query = []
    indeptree = FormulaParser(formula).tree[1]

    def _eval_indep_node(node):
        if isinstance(node, Token) and node.type == TokenType.INTERCEPT:
            return
        elif _is_scaled_value_node(node):
            (sign, scalar), varname = node
            sign, scalar, varname = sign.value, scalar.value, varname.value
            columns_to_query.append(f"{sign}{scalar} * {varname} AS {varname}")
        elif (
            isinstance(node, list) and node[0].type == TokenType.MUL
        ):  # full interaction
            for node2 in node[1:]:
                (sign, scalar), varname = node2
                sign, scalar, varname = sign.value, scalar.value, varname.value
                if (
                    subquery := f"{sign}{scalar} * {varname} AS {varname}"
                ) not in columns_to_query:
                    columns_to_query.append(subquery)
            columns_to_query.append(
                " * ".join(
                    [
                        f"({sign.value}{scalar.value} * {varname.value})"
                        for (sign, scalar), varname in node[1:]
                    ]
                )
            )
            columns_to_query[-1] += " AS interaction_" + "_".join(
                varname.value for (_, _), varname in node[1:]
            )
        elif (
            isinstance(node, list) and node[0].type == TokenType.INTERACTION
        ):  # just interaction term
            columns_to_query.append(
                " * ".join(
                    [
                        f"({sign.value}{scalar.value} * {varname.value})"
                        for (sign, scalar), varname in node[1:]
                    ]
                )
            )
            columns_to_query[-1] += " AS interaction_" + "_".join(
                varname.value for (_, _), varname in node[1:]
            )
        else:
            raise NotImplementedError(
                "Can only handle scaled nodes in depvar as of now"
            )

    for node in indeptree:
        _eval_indep_node(node)

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        is_not_null_condition = " AND ".join(
            [f"indepvar.{v.split()[-1].strip()} IS NOT NULL" for v in columns_to_query]
        )
        is_null_condition = is_not_null_condition.replace(
            "IS NOT NULL", "IS NULL"
        ).replace("AND", "OR")
        cur.execute(f"""
        CREATE VIEW subs_with_all_variables AS
        SELECT DISTINCT indepvar.subject FROM indepvar INNER JOIN subject_activation ON subject_activation.subject = indepvar.subject
        WHERE {is_not_null_condition}
        """)
        cur.execute(f"""
        CREATE VIEW subs_without_all_variables AS
        SELECT DISTINCT indepvar.subject FROM indepvar INNER JOIN subject_activation ON subject_activation.subject = indepvar.subject
        WHERE {is_null_condition}
        """)
        if (
            len(
                subs_without_variables := [
                    row[0]
                    for row in cur.execute(
                        "SELECT subject FROM subs_without_all_variables"
                    ).fetchall()
                ]
            )
            > 0
        ):
            logger.warning(
                "Subjects who have missing data for one or more independent variables: "
                + ",".join(subs_without_variables)
            )
        query = (
            "SELECT "
            + ",".join(columns_to_query)
            + " FROM indepvar WHERE subject IN subs_with_all_variables ORDER BY subject"
        )
        df = pd.read_sql_query(query, con)
    df["intercept"] = 1
    cols = ["intercept"] + [
        c for c in df.columns if c != "intercept"
    ]  # rearrange so intercept is first
    df = df[cols]
    activations = {}
    final_activation = {}

    def _query_activation(condition, scalar=1) -> dict:
        activation = query_depvar(condition, db_path, space, task, session)
        activation["activation"] *= scalar
        return activation

    def _eval_depvar_node(node):
        if _is_scaled_value_node(node):
            (sign, scalar), condition = node
            sign, scalar, condition = sign.value, scalar.value, condition.value
            scalar_int = int(f"{sign}{scalar}")
            activations[condition] = _query_activation(condition, scalar=scalar_int)
            if not final_activation:
                for key in activations[condition].keys():
                    if key != "activation":
                        final_activation[key] = activations[condition][key]
            return condition
        else:
            raise NotImplementedError(
                "Can only handle scaled nodes in depvar as of now"
            )

    for node in deptree:
        _eval_depvar_node(node)

    final_activation["activation"] = np.squeeze(
        np.sum(
            np.concatenate(
                [
                    activation["activation"][np.newaxis, ...]
                    for activation in activations.values()
                ]
            ),
            axis=0,
        )
    )
    return df, final_activation
