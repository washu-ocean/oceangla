from copy import deepcopy
import logging
import re
from collections import namedtuple
from enum import Enum, auto
from pathlib import Path
from textwrap import dedent

from pathvalidate import sanitize_filename

logger = logging.getLogger(__name__)

VALID_FUNCS = ("onesampttest", "fir_rmanova")


class TokenType(Enum):
    INVALID = auto()
    VAR = auto()
    PLUS = auto()
    MINUS = auto()
    TILDE = auto()
    MUL = auto()
    INTERACT = auto()
    INTERCEPT = auto()
    FUNCNAME = auto()
    NUMBER = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    ALL_INDIVIDUAL_CONDITIONS = auto()
    PIPE = auto()
    ZSCORE = auto()


Token = namedtuple("Token", ["type", "value"])
INTERCEPT_TOKEN = Token(type=TokenType.INTERCEPT, value="1")


def lex_formula_str(formula_str: str) -> list[Token]:
    # if "~" not in formula_str:
    #     raise ValueError(
    #         "Invalid model spec; must include char '~' to separate dependent from independent variables."
    #     )
    # elif formula_str.count("~") != 1:
    #     raise ValueError(
    #         "Invalid model spec; dependent/independent variable separator '~' can only be included once."
    #     )
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
            if varname == "ALL":
                tokens.append(Token(TokenType.ALL_INDIVIDUAL_CONDITIONS, varname))
            elif varname.isdigit():
                tokens.append(Token(TokenType.NUMBER, varname))
            else:
                tokens.append(Token(TokenType.VAR, varname))
        else:
            tokens.append(Token(TokenType.INVALID, formula_str[pos]))
            pos += 1
    return tokens


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


def eval_scalar(name: str) -> str:
    minuses = 0
    for i in range(len(name)):
        if name[i] == "-":
            minuses += 1
    return "+" if minuses % 2 == 0 else "-"


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

    # May get around to reimplementing this with the new syntax
    #
    # def __str__(self):
    #     s = ""
    #     if self.tree is None:
    #         return s
    #     deptree, indeptree = self.tree
    #     for node in deptree:
    #         s += f"({node[0][0].value}{node[0][1].value}){node[1].value} "
    #     s += "~ "
    #     for node in indeptree:
    #         if isinstance(node, Token) and node.type == TokenType.INTERCEPT:
    #             s += "intercept "
    #         elif is_scaled_value_node(node):
    #             s += f"({node[0][0].value}{node[0][1].value}){node[1].value} "
    #         elif isinstance(node, list) and node[0].type == TokenType.MUL:
    #             childnodes = [
    #                 f"({childnode[0][0].value}{childnode[0][1].value}){childnode[1].value}"
    #                 for childnode in node[1:]
    #             ]
    #             interaction_term = ":".join(childnodes)
    #             s += " ".join([*childnodes, interaction_term])
    #         elif isinstance(node, list) and node[0].type == TokenType.INTERACT:
    #             childnodes = [
    #                 f"({childnode[0][0].value}{childnode[0][1].value}){childnode[1].value}"
    #                 for childnode in node[1:]
    #             ]
    #             interaction_term = ":".join(childnodes)
    #             s += f" {interaction_term} "
    #     return s.strip()

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

    def parse(self) -> dict[str, list]:
        return self.statement()
        # depvar = self.depvar()
        # indepvar = self.indepvar()
        # return (depvar, indepvar)

    def statement(self) -> dict[str, list]:
        if len(list(filter(lambda t: t.type == TokenType.TILDE, self.tokens))) == 1:
            return self.expression()
        else:
            return self.function()

    def function(self) -> dict[str, list]:
        if self.peek().type != TokenType.VAR:
            raise UnexpectedTokenError(self)
        elif self.peek().value not in VALID_FUNCS:
            raise UnexpectedTokenError(self)
        funcname=self.consume().value
        if self.consume().type != TokenType.LPAREN:
            raise UnexpectedTokenError(self)
        arglist=self.arglist()
        if self.consume().type != TokenType.RPAREN:
            raise UnexpectedTokenError(self)
        return {funcname: arglist}

    def arglist(self) -> list[str]:
        arglist=[]
        if self.peek().type != TokenType.VAR:
            raise UnexpectedTokenError(self)
        arglist.append(self.consume().value)
        while self.peek().type == TokenType.COMMA:
            self.consume()
            if self.peek().type != TokenType.VAR:
                raise UnexpectedTokenError(self)
            arglist.append(self.consume().value)
        return arglist

    def expression(self) -> dict[str, list]:
        depvar=self.depvar()
        indepvar=self.indepvar()
        return {
            "depvars": depvar,
            "indepvars": indepvar
        }

    def depvar(self):
        depvarnames=[]
        depvarnames.append(self.depvarname())
        while self.peek().type != TokenType.TILDE:
            depvarnames.append(self.depvarname())
        self.consume()
        return depvarnames

    def depvarname(self):
        name=""

        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            name += self.consume().value

        name="+" if name == "" else eval_scalar(name)  # add scalar if one not present

        if self.peek().type != TokenType.VAR:
            raise UnexpectedTokenError(self)
        name += self.consume().value
        return name

    def indepvar(self):
        indepvars=[]
        if self.peek() == Token(type=TokenType.NUMBER, value="1"):
            self.consume()
        indepvars.extend(self.interaction())
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS, TokenType.LPAREN):
            indepvars.extend(self.interaction())
        if "intercept" not in indepvars:
            indepvars.insert(0, "intercept")
        return indepvars

    def interaction(self):
        names=[""]
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            names[0] += self.consume().value
        names[0]="+" if names[0] == "" else eval_scalar(names[0])
        if self.peek().type != TokenType.VAR:
            raise UnexpectedTokenError(self)
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
                if self.peek().type != TokenType.VAR:
                    raise UnexpectedTokenError(self)
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
                if self.peek().type != TokenType.VAR:
                    raise UnexpectedTokenError(self)
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
