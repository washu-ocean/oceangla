from copy import deepcopy
import logging
import re
from collections import namedtuple
from enum import Enum, auto
from pathlib import Path
from textwrap import dedent
from typing import Sequence

from pathvalidate import sanitize_filename

from .model import ModelDesc

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
            super().__init__(f"Unexpected token: {received_token!r}. Expected one of {expected_token_type!r}")=


def eval_scalar(name: str) -> str:
    minuses = 0
    for i in range(len(name)):
        if name[i] == "-":
            minuses += 1
    return "+" if minuses % 2 == 0 else "-"


class FormulaParser:
    def __init__(self, tokens, parse=True):
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
        self.token_stream = deepcopy(self.tokens)
        if parse:
            self.model_desc: ModelDesc = self.parse()

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

    def parse(self) -> ModelDesc:
        return self.statement()
        # depvar = self.depvar()
        # indepvar = self.indepvar()
        # return (depvar, indepvar)

    def statement(self) -> ModelDesc:
        if len(list(filter(lambda t: t.type == TokenType.TILDE, self.tokens))) == 1:
            return self.expression()
        else:
            return self.function()

    def function(self) -> ModelDesc:
        self.expect(TokenType.VAR)
        funcname=self.consume().value
        self.expect(TokenType.LPAREN, consume=True)
        arglist=self.arglist()
        self.expect(TokenType.RPAREN, consume=True)
        return {
            "model_type": funcname,
            "function_args": arglist
        }

    def arglist(self) -> list[str]:
        arglist=[]
        self.expect(TokenType.VAR)
        arglist.append(self.consume().value)
        while self.peek().type == TokenType.COMMA:
            self.consume()
            self.expect(TokenType.VAR)
            arglist.append(self.consume().value)
        return arglist

    def expression(self) -> ModelDesc:
        depvar=self.depvar()
        indepvar=self.indepvar()
        return {
            "model_type": "OLS",
            "depvars": depvar,
            "indepvars": indepvar
        }

    def depvar(self) -> list[str]:
        depvarnames=[]
        depvarnames.append(self.depvarname())
        while self.peek().type != TokenType.TILDE:
            depvarnames.append(self.depvarname())
        self.consume()
        return depvarnames

    def depvarname(self) -> str:
        name=""

        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            name += self.consume().value

        name="+" if name == "" else eval_scalar(name)  # add scalar if one not present

        self.expect(TokenType.VAR)
        name += self.consume().value
        return name

    def indepvar(self) -> list[str]:
        indepvars=[]
        if self.peek() == Token(type=TokenType.NUMBER, value="1"):
            self.consume()
        indepvars.extend(self.interaction())
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS, TokenType.LPAREN):
            indepvars.extend(self.interaction())
        return indepvars

    def interaction(self) -> list[str]:
        names=[""]
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            names[0] += self.consume().value
        names[0]="+" if names[0] == "" else eval_scalar(names[0])
        self.expect(TokenType.VAR)
        names[0] += self.consume().value
        op=None  # We can't chain expanded and explicit interactions, i.e. a*b:c would be an invalid interaction term, so we choose only one
        while self.peek().type in (TokenType.MUL, TokenType.INTERACT):
            if self.peek().type == TokenType.MUL and op in (None, TokenType.MUL):
                if op is None:
                    op=TokenType.MUL
                self.consume()
                names.append("")
                while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
                    names[-1] += self.consume().value
                names[-1]="+" if names[-1] == "" else eval_scalar(names[-1])
                self.expect(TokenType.VAR)
                names[-1] += self.consume().value
                cur_name = names[-1]
                names_ = deepcopy(names)
                for name in names_[:-1]:
                    names.append(f"{name}:{cur_name}")
            elif self.peek().type == TokenType.INTERACT and op in (None, TokenType.INTERACT):
                if op is None:
                    op=TokenType.INTERACT
                self.consume()
                sign=""
                while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
                    sign += self.consume().value
                sign="+" if sign == "" else eval_scalar(sign)
                self.expect(TokenType.VAR)
                names[0].append(f":{sign}{self.consume().value}")
        return names


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

                model1     ->     depvar ~ indepvar1 + indepvar2

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
