from textual.app import ComposeResult, Widget
from textual.widgets import (
    Input,
    Label,
    ListItem,
    ListView,
)


class FLASourceManager(Widget):
    def compose(self) -> ComposeResult:
        with ListView(id="fla_directories"):
            yield ListItem(Label("+ Add new FLA directory"))
            yield Input(id="fla_directory_input")
