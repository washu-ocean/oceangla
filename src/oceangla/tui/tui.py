from pathlib import Path

from textual.app import App, ComposeResult, Widget
from textual.containers import Container
from textual.widgets import Footer, Label, ListItem, ListView, Header, Log, Static, DirectoryTree
from textual.reactive import reactive

from .choosefile import FLASourceManager

class OptionsPane(Widget):
    fla_source_manager = FLASourceManager()

    def compose(self) -> ComposeResult:
        with ListView():
            yield ListItem(Label("Manage first-level analysis directories"), id="fla_directories")
            yield ListItem(Label("Manage variables"), id="variable_files")

    def on_list_view_highlighted(self, event: ListView.Highlighted):
        right_pane = self.screen.query_one(RightPane)
        if event.item.id == "fla_directories":
            right_pane.widget = self.fla_dirs_pane
        else:
            right_pane.widget = Static()


class RightPane(Container):
    widget = reactive(Static(), recompose=True)

    def compose(self):
        yield self.widget


class TUI(App):
    CSS_PATH = Path(__file__).parent / "styles.tcss"

    # right_pane = Static(id="right-pane"), recompose=True

    def compose(self) -> ComposeResult:
        yield Header()
        yield OptionsPane()
        yield RightPane()
        yield Footer()

    def on_mount(self) -> None:
        self.title = "oceangla"
        self.sub_title = "v0.1.0"


def main():
    app = TUI()
    app.run()


if __name__ == "__main__":
    main()
