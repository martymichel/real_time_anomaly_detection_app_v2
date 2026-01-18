"""
Simple CSET File Editor
Bearbeitet Kamera-Einstellungen (ReverseX/ReverseY) in .cset Dateien
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, List, Tuple


class CsetEditor:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CSET File Editor")
        self.root.geometry("600x400")

        self.file_path = None
        self.lines: List[str] = []
        self.reverse_x_line_idx = None
        self.reverse_y_line_idx = None

        self._create_ui()

        # Try to load default camera_setup.cset
        default_path = Path(__file__).parent / "camera_setup.cset"
        if default_path.exists():
            self.load_file(str(default_path))

    def _create_ui(self):
        """Create the user interface"""
        # Top frame: File operations
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Datei:").pack(side=tk.LEFT, padx=5)

        self.file_label = ttk.Label(top_frame, text="Keine Datei geladen",
                                     foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        ttk.Button(top_frame, text="Öffnen...",
                   command=self._open_file).pack(side=tk.RIGHT, padx=2)
        ttk.Button(top_frame, text="Speichern",
                   command=self._save_file).pack(side=tk.RIGHT, padx=2)

        # Separator
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Middle frame: Settings
        settings_frame = ttk.LabelFrame(self.root, text="Spiegeleinstellungen",
                                        padding=20)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Horizontal flip
        h_frame = ttk.Frame(settings_frame)
        h_frame.pack(fill=tk.X, pady=10)

        ttk.Label(h_frame, text="Horizontal spiegeln (ReverseX):",
                  font=("", 10, "bold")).pack(side=tk.LEFT, padx=10)

        self.reverse_x_var = tk.IntVar(value=0)
        self.reverse_x_check = ttk.Checkbutton(
            h_frame,
            variable=self.reverse_x_var,
            command=self._on_value_changed,
            state=tk.DISABLED
        )
        self.reverse_x_check.pack(side=tk.LEFT, padx=10)

        self.reverse_x_status = ttk.Label(h_frame, text="Aus", foreground="gray")
        self.reverse_x_status.pack(side=tk.LEFT, padx=5)

        # Vertical flip
        v_frame = ttk.Frame(settings_frame)
        v_frame.pack(fill=tk.X, pady=10)

        ttk.Label(v_frame, text="Vertikal spiegeln (ReverseY):",
                  font=("", 10, "bold")).pack(side=tk.LEFT, padx=10)

        self.reverse_y_var = tk.IntVar(value=0)
        self.reverse_y_check = ttk.Checkbutton(
            v_frame,
            variable=self.reverse_y_var,
            command=self._on_value_changed,
            state=tk.DISABLED
        )
        self.reverse_y_check.pack(side=tk.LEFT, padx=10)

        self.reverse_y_status = ttk.Label(v_frame, text="Aus", foreground="gray")
        self.reverse_y_status.pack(side=tk.LEFT, padx=5)

        # Info text
        info_frame = ttk.Frame(settings_frame)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        info_text = (
            "Hinweis:\n"
            "• Werte: 0 = Aus, 1 = Ein\n"
            "• Änderungen werden erst nach 'Speichern' wirksam\n"
            "• Kamera muss neu gestartet werden, um Änderungen zu laden"
        )
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT,
                  foreground="blue").pack(anchor=tk.W, padx=10)

        # Bottom frame: Status
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(status_frame, text="Bereit",
                                       relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, padx=2, pady=2)

    def _open_file(self):
        """Open file dialog to select .cset file"""
        file_path = filedialog.askopenfilename(
            title="CSET Datei öffnen",
            filetypes=[("CSET Files", "*.cset"), ("All Files", "*.*")],
            initialdir=Path(__file__).parent
        )

        if file_path:
            self.load_file(file_path)

    def load_file(self, file_path: str):
        """Load and parse .cset file"""
        try:
            self.file_path = file_path

            with open(file_path, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()

            # Find ReverseX and ReverseY lines
            self.reverse_x_line_idx = None
            self.reverse_y_line_idx = None

            for i, line in enumerate(self.lines):
                if line.startswith('ReverseXValueControl'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.reverse_x_line_idx = i
                        self.reverse_x_var.set(int(parts[-1]))

                elif line.startswith('ReverseYValueControl'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.reverse_y_line_idx = i
                        self.reverse_y_var.set(int(parts[-1]))

            # Update UI
            self.file_label.config(text=Path(file_path).name, foreground="black")

            if self.reverse_x_line_idx is not None:
                self.reverse_x_check.config(state=tk.NORMAL)
                self._update_status_label(self.reverse_x_status, self.reverse_x_var.get())
            else:
                messagebox.showwarning("Warnung",
                                       "ReverseXValueControl nicht gefunden!")

            if self.reverse_y_line_idx is not None:
                self.reverse_y_check.config(state=tk.NORMAL)
                self._update_status_label(self.reverse_y_status, self.reverse_y_var.get())
            else:
                messagebox.showwarning("Warnung",
                                       "ReverseYValueControl nicht gefunden!")

            self.status_label.config(text=f"Geladen: {file_path}")

        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Laden: {e}")
            self.status_label.config(text=f"Fehler: {e}")

    def _save_file(self):
        """Save modified .cset file"""
        if not self.file_path:
            messagebox.showwarning("Warnung", "Keine Datei geladen!")
            return

        try:
            # Update lines with new values
            if self.reverse_x_line_idx is not None:
                parts = self.lines[self.reverse_x_line_idx].strip().split('\t')
                parts[-1] = str(self.reverse_x_var.get())
                self.lines[self.reverse_x_line_idx] = '\t'.join(parts) + '\n'

            if self.reverse_y_line_idx is not None:
                parts = self.lines[self.reverse_y_line_idx].strip().split('\t')
                parts[-1] = str(self.reverse_y_var.get())
                self.lines[self.reverse_y_line_idx] = '\t'.join(parts) + '\n'

            # Write to file
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.writelines(self.lines)

            self.status_label.config(text=f"Gespeichert: {self.file_path}")
            messagebox.showinfo("Erfolg",
                                "Datei erfolgreich gespeichert!\n\n"
                                "Kamera neu starten, um Änderungen zu laden.")

        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Speichern: {e}")
            self.status_label.config(text=f"Fehler: {e}")

    def _on_value_changed(self):
        """Update status labels when checkbox changes"""
        self._update_status_label(self.reverse_x_status, self.reverse_x_var.get())
        self._update_status_label(self.reverse_y_status, self.reverse_y_var.get())
        self.status_label.config(text="Nicht gespeichert (Änderungen vorgenommen)")

    def _update_status_label(self, label: ttk.Label, value: int):
        """Update status label based on value"""
        if value == 1:
            label.config(text="Ein", foreground="green")
        else:
            label.config(text="Aus", foreground="gray")


def main():
    root = tk.Tk()
    app = CsetEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
