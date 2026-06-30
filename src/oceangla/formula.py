import logging
import re
from collections import namedtuple
from enum import Enum, auto
from pathlib import Path
from textwrap import dedent
import math

from pathvalidate import sanitize_filename
import pandas as pd
import numpy as np

from .model import Model
from .dataframes import zscore

logger = logging.getLogger(__name__)

VALID_FUNCS = ("Z",)


class TokenType(Enum):
    INVALID = auto()
    WORD = auto()
    PLUS = auto()
    MINUS = auto()
    TILDE = auto()
    MUL = auto()
    COLON = auto()
    INT = auto()
    FLOAT = auto()
    LPAREN = auto()
    RPAREN = auto()
    BAR = auto()
    DIV = auto()


Token = namedtuple("Token", ["type", "value"])
Contrast = namedtuple("Contrast", ["scalar", "condition"])
FixedEffect = namedtuple("FixedEffect", ["type", "name"])
FixedEffectInteraction = namedtuple("FixedEffectInteraction", ["type", "terms"])
RandomEffect = namedtuple("RandomEffect", ["modifier_type", "modifier_value", "grouping_factor_type", "level1group", "level2group"])
type ContrastOrEffect = Contrast | FixedEffect | FixedEffectInteraction | RandomEffect
type ModelMetadata = dict[str, list[ContrastOrEffect] | bool]

def lex_formula_str(formula_str: str) -> list[Token]:
    pos = 0
    tokens = []

    def is_var_char(c: str):
        return c.isalnum() or c in "_."

    while pos < len(formula_str):
        if formula_str[pos].isspace():
            pos += 1
        elif formula_str[pos] in "+-*:()|/~":
            tokens.append(
                Token(
                    {
                        "+": TokenType.PLUS,
                        "-": TokenType.MINUS,
                        "*": TokenType.MUL,
                        ":": TokenType.COLON,
                        "(": TokenType.LPAREN,
                        ")": TokenType.RPAREN,
                        "|": TokenType.BAR,
                        "/": TokenType.DIV,
                        "~": TokenType.TILDE,
                    }[formula_str[pos]],
                    None,
                )
            )
            pos += 1
        elif is_var_char(formula_str[pos]):
            word = ""
            while pos < len(formula_str) and is_var_char(formula_str[pos]):
                word += formula_str[pos]
                pos += 1
            if word.isdigit():
                tokens.append(Token(TokenType.INT, int(word)))
            elif word.replace(".", "", 1).isdigit():
                tokens.append(Token(TokenType.FLOAT, float(word)))
            else:
                tokens.append(Token(TokenType.WORD, word))
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
            if not tokens[pos + 1].type == TokenType.WORD:
                raise ValueError(
                    f"Illegal token after '{tokens[pos].value}' : '{tokens[pos + 1].value}'"
                )
            depvars.append(f"{tokens[pos].value}{tokens[pos + 1].value}")
            pos += 2
        elif tokens[pos].type == TokenType.WORD and len(depvars) == 0:
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
            isinstance(node[0][1], Token) and node[0][1].type == TokenType.INT,
            isinstance(node[1], Token) and node[1].type == TokenType.WORD,
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
        nodes.append(self.depvarname())
        while self.peek().type != TokenType.TILDE:
            nodes.append(self.depvarname_())
        self.consume()
        return nodes

    def depvarname(self):
        match self.peek().type:
            case TokenType.WORD:
                condition = self.consume().value
                return Contrast(scalar=1, condition=condition)
            case _:
                raise UnexpectedTokenError(self)

    def depvarname_(self):
        scalar = None
        match self.peek().type:
            case TokenType.PLUS:
                self.consume()
                scalar = 1
            case TokenType.MINUS:
                self.consume()
                scalar = -1
            case _:
                raise UnexpectedTokenError(self)
        assert scalar is not None, "Scalar must be a number"
        if self.peek().type != TokenType.WORD:
            raise UnexpectedTokenError(self)
        condition = self.consume().value
        return Contrast(scalar=scalar, condition=condition)

    def indepvar(self):
        nodes = []
        nodes.append(self.term())
        while self.peek() != Token(type=TokenType.INVALID, value=""):
            nodes.append(self.indepvar_())
        return nodes

    def indepvar_(self):
        match self.peek().type:
            case TokenType.PLUS:
                self.consume()
                return self.term()
            case _:
                raise UnexpectedTokenError(self)

    def term(self):
        match self.peek().type:
            case TokenType.WORD | TokenType.INT:
                return self.fixedeffect()
            case TokenType.LPAREN:
                return self.randomeffect()

    def fixedeffect(self):
        node = FixedEffect(type="intercept", name=None)
        match self.peek().type:
            case TokenType.WORD:
                word = self.consume()
                if word == "C" and self.peek().type == TokenType.LPAREN:
                    self.consume()
                    if self.peek().type != TokenType.WORD:
                        raise UnexpectedTokenError(self)
                    word = self.consume()
                    if self.peek().type != TokenType.RPAREN:
                        raise UnexpectedTokenError(self)
                    self.consume()
                    node = FixedEffect(type="categorical", name=word.value)
                elif word == "fir_frame":
                    node = FixedEffect(type="fir_frame", name=None)
                else:
                    node = FixedEffect(type="continuous", name=word.value)
            case TokenType.INT:
                if self.peek().value == 1:
                    self.consume()
                elif self.peek().value == 0:
                    self.consume()
                    node = FixedEffect(type="no_intercept", name=None)
            case _:
                raise UnexpectedTokenError(self)
        if node.type not in ("invalid", "no_intercept") and self.peek().type in (TokenType.MUL, TokenType.COLON):
            op = "interact_expand" if self.consume().type == TokenType.MUL else "interact"
            if not self.peek().type == TokenType.WORD:
                raise UnexpectedTokenError(self)
            word = self.consume()
            if word.value == "C" and self.peek().type == TokenType.LPAREN:
                self.consume()
                if self.peek().type != TokenType.WORD:
                    raise UnexpectedTokenError(self)
                word = self.consume()
                if self.peek().type != TokenType.RPAREN:
                    raise UnexpectedTokenError(self)
                self.consume()
                return FixedEffectInteraction(type=op, terms=[node, FixedEffect(
                    type="categorical", name=word.value
                )])
            elif word.value == "fir_frame":
                return FixedEffectInteraction(type=op, terms=[node, FixedEffect(
                    type="fir_frame", name=None
                )])
            else:
                return FixedEffectInteraction(type=op, terms=[node, FixedEffect(
                    type="continuous", name=word.value
                )])
        else:
            return node

    def randomeffect(self):
        self.consume()
        modifier_type, modifier_value = None, None
        match self.peek().type:  # modifier
            case TokenType.WORD:
                modifier_type, modifier_value = "slope_modifier", self.consume().value
            case TokenType.INT:
                if self.peek().value == 1:
                    self.consume()
                    modifier_type, modifier_value = "intercept_modifier", None
                elif self.peek().value == 0:
                    self.consume()
                    if not self.peek().type == TokenType.PLUS:
                        raise UnexpectedTokenError(self)
                    self.consume()
                    if not self.peek().type == TokenType.WORD:
                        raise UnexpectedTokenError(self)
                    modifier_type, modifier_value = "slope_modifier_no_intercept", self.consume().value
                else:
                    raise UnexpectedTokenError(self)
            case _:
                raise UnexpectedTokenError(self)
        if not self.peek().type == TokenType.BAR:
            raise UnexpectedTokenError(self)
        while self.peek().type == TokenType.BAR:
            self.consume()  # No difference in single- and double-bars for now
        if self.peek().type != TokenType.WORD:
            raise UnexpectedTokenError(self)
        grouping_var = self.consume().value
        match self.peek().type:
            case TokenType.DIV:
                self.consume()
                if self.peek().type != TokenType.WORD:
                    raise UnexpectedTokenError(self)
                nested_var = self.consume().value
                grouping_factor_type, level1group, level2group = "nested_grouping_factor", grouping_var, nested_var
            case TokenType.COLON:
                self.consume()
                if self.peek().type != TokenType.WORD:
                    raise UnexpectedTokenError(self)
                interacting_var = self.consume().value
                grouping_factor_type, level1group, level2group = "interacting_grouping_factor", grouping_var, interacting_var
            case _:
                grouping_factor_type, level1group, level2group = "grouping_factor", grouping_var, None
        if not self.peek().type == TokenType.RPAREN:
            raise UnexpectedTokenError(self)
        self.consume()
        return RandomEffect(
            modifier_type=modifier_type,
            modifier_value=modifier_value,
            grouping_factor_type=grouping_factor_type,
            level1group=level1group,
            level2group=level2group
        )


def evaluate(tree) -> dict[str, list[Contrast | FixedEffect | FixedEffectInteraction | RandomEffect]]:
    contrasts, indepvar_tree = tree
    fixed_effects = [FixedEffect(type="intercept", name=None)]
    random_effects = []
    unassumed_model = False
    for node in indepvar_tree:
        if isinstance(node, FixedEffect):
            match node.type:
                case 'invalid':
                    raise ValueError(f"Received an invalid FixedEffect node. Full list: {fixed_effects}")
                case 'no_intercept':
                    fixed_effects.pop(0)
                case 'continuous' | 'categorical':
                    fixed_effects.append(node)
                case 'fir_frame':
                    unassumed_model = True
                    fixed_effects.append(node)
        elif isinstance(node, FixedEffectInteraction):
            match node.type:
                case 'interact_expand':
                    for term in node.terms[:2]:
                        fixed_effects.append(term)
                    fixed_effects.append(FixedEffectInteraction(type="interact", terms=node.terms[:2]))
                case 'interact':
                    fixed_effects.append(node)
                case _:
                    raise ValueError(f"Unexpected FixedEffectInteraction.type {node.type}")
        elif isinstance(node, RandomEffect):  # Just assume that the only grouping variable is subject for now
            unassumed_model = True
            random_effects.append(node)
    return {
        "contrasts": contrasts,
        "fixed_effects": fixed_effects,
        "random_effects": random_effects,
        "unassumed_model": unassumed_model
    }


def parse_model_file(model_file: Path) -> tuple[list[str], list[str]]:
    model_names, models = [], []
    with open(model_file) as f:
        lines = f.readlines()
    for line in lines:
        if len(re.findall("->", line)) != 1:
            raise ValueError(
                f"Models in model file {model_file.resolve()} "
                "must contain one arrow -> separating the model "
                "name on the left, and the formula on the right."
            )
        model_name, formula = [chunk.strip() for chunk in line.split("->")]
        if len(model_name) == 0 or len(formula) == 0:
            raise ValueError(
                dedent(f"""
                Each model specified in {model_file.resolve()}
                must contain a model name and formula, separated by an
                arrow '->'. Example file contents:

                model1     ->     depvar ~ indepvar1 + indepvar2

                ^                 ^
                |                 |
                model name        model spec
                """)
            )
        model_name = sanitize_filename(model_name)
        FormulaParser(formula)  # quick parse, should error out if invalid
        model_names.append(model_name)
        models.append(formula)
    return (model_names, models)
