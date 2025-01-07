class UndoManager:
    def __init__(self, max_steps=20):
        self.undo_stack = []
        self.redo_stack = []
        self.max_steps = max_steps

    def push(self, image):
        """Add new state to undo stack"""
        if image is None:
            return
        self.undo_stack.append(image.copy())
        # Clear redo stack when new action is performed
        self.redo_stack.clear()
        # Keep stack size under limit
        if len(self.undo_stack) > self.max_steps:
            self.undo_stack.pop(0)

    def undo(self):
        """Move current state to redo stack and return previous state"""
        if not self.undo_stack:
            return None
        state = self.undo_stack.pop()
        if state:
            self.redo_stack.append(state)
        return self.undo_stack[-1].copy() if self.undo_stack else None

    def redo(self):
        """Restore state from redo stack"""
        if not self.redo_stack:
            return None
        state = self.redo_stack.pop()
        if state:
            self.undo_stack.append(state)
        return state

    def can_undo(self):
        """Check if undo is available"""
        return len(self.undo_stack) > 1  # Need at least 2 states to undo

    def can_redo(self):
        """Check if redo is available"""
        return len(self.redo_stack) > 0
