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


def is_scaled_value_node(node):
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
            elif is_scaled_value_node(node):
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
