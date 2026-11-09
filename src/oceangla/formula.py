from copy import deepcopy
import logging
import re
from collections import namedtuple
from enum import Enum, auto
from pathlib import Path
from textwrap import dedent
from typing import Sequence, List

from pathvalidate import sanitize_filename

from .model import ModelSpec, FirstLevelContrast
from .ols import OLSModel

logger = logging.getLogger(__name__)

class TokenType(Enum):
    INVALID = auto()
    VAR = auto()
    PLUS = auto()
    MINUS = auto()
    TILDE = auto()
    MUL = auto()
    INTERACT = auto()
    FUNCNAME = auto()
    NUMBER = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    ALL_INDIVIDUAL_CONDITIONS = auto()
    PIPE = auto()
    ZSCORE = auto()


Token = namedtuple("Token", ["type", "value"])

def lex_formula_str(formula_str: str) -> list[Token]:
    pos = 0

    tokens = []

    def is_var_char(c: str):
        return c.isalnum() or c in "_"

    while pos < len(formula_str):
        if formula_str[pos].isspace():
            pos += 1
        elif formula_str[pos] in "+-*:()~,":
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
                        ",": TokenType.COMMA,
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
            # TODO: reimplement 'ALL'
            # if varname == "ALL":
            #     tokens.append(Token(TokenType.ALL_INDIVIDUAL_CONDITIONS, varname))
            if varname.isdigit():
                tokens.append(Token(TokenType.NUMBER, varname))
            else:
                tokens.append(Token(TokenType.VAR, varname))
        else:
            tokens.append(Token(TokenType.INVALID, formula_str[pos]))
            pos += 1
    return tokens


class UnexpectedTokenError(Exception):
    def __init__(self,
                 received_token: Token,
                 expected_token_type: TokenType | Sequence[TokenType]):
        if isinstance(expected_token_type, TokenType):
            super().__init__(f"Unexpected token: {received_token!r}. Expected type {expected_token_type!r}")
        else:
            super().__init__(f"Unexpected token: {received_token!r}. Expected one of {expected_token_type!r}")


def eval_scalar(name: str) -> int:
    minuses = 0
    for i in range(len(name)):
        if name[i] == "-":
            minuses += 1
    return 1 if minuses % 2 == 0 else -1


class FormulaParser:
    def __init__(self, tokens, parse=True):
        self.pos = 0
        self.string = tokens
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
        self.token_stream = deepcopy(self.tokens)
        if parse:
            self.model_spec: ModelSpec = self.parse()

    # May get around to reimplementing this with the new syntax
    #
    def __str__(self):
        return str(self.parsed_formula)

    def reset(self):
        self.tokens = self.orig_tokens
        self.pos = 0

    def peek(self):
        return (
            self.tokens[self.pos]
            if self.pos < len(self.tokens)
            else Token(type=TokenType.INVALID, value="")
        )

    def expect(self, tokentype: TokenType | Sequence[TokenType], consume: bool=False):
        t = self.consume().type if consume else self.peek().type
        if isinstance(tokentype, TokenType):
            if t != tokentype:
                raise UnexpectedTokenError(t, tokentype)
        else:
            if t not in tokentype:
                raise UnexpectedTokenError(t, tokentype)


    def consume(self):
        token = self.peek()
        self.pos += 1
        return token

    def parse(self) -> ModelSpec:
        return self.statement()
        # firstlevel_contrasts = self.firstlevel_contrasts()
        # subject_variables = self.subject_variables()
        # return (firstlevel_contrasts, subject_variables)

    def statement(self) -> ModelSpec:
        if len(list(filter(lambda t: t.type == TokenType.TILDE, self.tokens))) == 1:
            return self.expression()
        else:
            return self.function()

    def function(self) -> ModelSpec:
        self.expect(TokenType.VAR)
        funcname=self.consume().value
        self.expect(TokenType.LPAREN, consume=True)
        arglist=self.arglist()
        self.expect(TokenType.RPAREN, consume=True)
        return ModelSpec(
            model_type=funcname,
            function_args=arglist
        )

    def arglist(self) -> list[str]:
        arglist=[]
        self.expect(TokenType.VAR)
        arglist.append(self.consume().value)
        while self.peek().type == TokenType.COMMA:
            self.consume()
            self.expect(TokenType.VAR)
            arglist.append(self.consume().value)
        return arglist

    def expression(self) -> ModelSpec:
        firstlevel_contrasts=self.firstlevel_contrasts()
        subject_variables=self.subject_variables()
        return ModelSpec(
            model_type="OLS",
            firstlevel_contrasts=firstlevel_contrasts,
            subject_variables=subject_variables
        )

    def firstlevel_contrasts(self) -> List[FirstLevelContrast]:
        firstlevel_contrasts = [self.firstlevel_contrasts_name()]
        while self.peek().type != TokenType.TILDE:
            firstlevel_contrasts.append(self.firstlevel_contrasts_name())
        self.consume()
        return firstlevel_contrasts

    def firstlevel_contrasts_name(self) -> FirstLevelContrast:
        name=""

        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            name += self.consume().value

        scalar = 1 if name == "" else eval_scalar(name)

        self.expect(TokenType.VAR)
        name = self.consume().value
        return FirstLevelContrast(name=name, scalar=scalar)

    def subject_variables(self) -> str:
        return self.string.split('~')[-1] # this is handled by patsy now


def parse_model_file(model_file: Path) -> tuple[list[str], list[str]]:
    model_names, models=[], []
    with open(model_file) as f:
        lines=f.readlines()
    for line in lines:
        if len(re.findall("->", line)) != 1:
            raise ValueError(
                f"Models in model file {model_file.resolve()} "
                "must contain one arrow -> separating the model "
                "name on the left, and the formula on the right."
            )
        model_name, formula=[chunk.strip() for chunk in line.split("->")]
        if len(model_name) == 0 or len(formula) == 0:
            raise ValueError(
                dedent(f"""
                Each model specified in {model_file.resolve()}
                must contain a model name and formula, separated by an
                arrow '->'. Example file contents:

                model1     ->     firstlevel_contrasts ~ subject_variables1 + subject_variables2
                model2     ->     fir_rmanova(oddball)
                ^                 ^
                |                 |
                model name        model spec
                """)
            )
        model_name=sanitize_filename(model_name)
        FormulaParser(formula)  # quick parse, should error out if invalid
        model_names.append(model_name)
        models.append(formula)
    return (model_names, models)
