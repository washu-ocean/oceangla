from textual.app import App, ComposeResult, Widget
from textual.containers import Container
from textual.widgets import Footer, Label, ListItem, ListView, Header, Log, Static, DirectoryTree
from textual.reactive import reactive

class FLASourceManager(Widget):
    def compose(self) -> ComposeResult:
        with ListView(id="fla_directories"):
            yield ListItem(Label("+ Add new FLA directory"))
            yield Input(id="fla_directory_input")
