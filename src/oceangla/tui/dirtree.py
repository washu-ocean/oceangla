from pathlib import Path

from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.message import Message
from textual.widgets import (
    DirectoryTree,
    Footer,
    Log,
)
from textual.widgets._directory_tree import DirEntry
from textual.widgets._tree import TreeNode


class FolderChoosingDirectoryTree(DirectoryTree):
    BINDINGS: list[BindingType] = [
        Binding(
            "shift+left",
            "cursor_parent",
            "Cursor to parent",
            show=False,
        ),
        Binding(
            "shift+right",
            "cursor_parent_next_sibling",
            "Cursor to next ancestor",
            show=False,
        ),
        Binding(
            "shift+up",
            "cursor_previous_sibling",
            "Cursor to previous sibling",
            show=False,
        ),
        Binding(
            "shift+down",
            "cursor_next_sibling",
            "Cursor to next sibling",
            show=False,
        ),
        Binding("enter", "select_cursor", "Select", show=False),
        Binding("c", "choose_dir", "Choose this directory"),
        Binding("space", "toggle_node", "Toggle", show=False),
        Binding(
            "shift+space",
            "toggle_expand_all",
            "Expand or collapse all",
            show=False,
        ),
        Binding("up", "cursor_up", "Cursor Up", show=False),
        Binding("down", "cursor_down", "Cursor Down", show=False),
    ]

    class DirectoryChose(Message):
        def __init__(self, node: TreeNode[DirEntry], path: Path):
            super().__init__()
            self.node: TreeNode[DirEntry] = node
            self.path: Path = path


class FolderChooser(Container):
    dir_tree = FolderChoosingDirectoryTree(Path(".").resolve())
    chosen_dir = None

    def compose(self):
        yield self.dir_tree
        yield Log(id="folder-chooser-log")
        yield Footer()
