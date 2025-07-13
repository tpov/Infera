import tkinter as tk
from tkinter import scrolledtext
import sys
import io

# We will import the AGI pipeline here
# To make this runnable from the root directory, we need to adjust the path
# A better way is to ensure the python path is set correctly when running.
# For simplicity, we'll add the src directory.
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.main import AGIPipeline, FinalMockCommandNetwork


class App(tk.Tk):
    """
    A simple Tkinter GUI for interacting with the AGI Pipeline.
    """
    def __init__(self):
        super().__init__()

        self.title("AGI Declarative Model Interface")
        self.geometry("800x600")

        # To be implemented in the next steps
        self.create_widgets()

    def create_widgets(self):
        self.main_frame = tk.Frame(self, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Input Frame ---
        input_frame = tk.Frame(self.main_frame)
        input_frame.pack(fill=tk.X, pady=5)

        tk.Label(input_frame, text="Ваш запрос:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 10))
        self.query_entry = tk.Entry(input_frame, font=("Arial", 11))
        self.query_entry.pack(fill=tk.X, expand=True, side=tk.LEFT)
        self.submit_button = tk.Button(input_frame, text="Отправить", command=self.process_query)
        self.submit_button.pack(side=tk.LEFT, padx=(10, 0))

        # --- Output Panes ---
        paned_window = tk.PanedWindow(self.main_frame, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=10)

        # Top pane for Graph State
        graph_frame = tk.LabelFrame(paned_window, text="Состояние Графа Мира", padx=5, pady=5)
        self.graph_text = scrolledtext.ScrolledText(graph_frame, wrap=tk.WORD, font=("Courier New", 9))
        self.graph_text.pack(fill=tk.BOTH, expand=True)
        paned_window.add(graph_frame)

        # Bottom pane for Response
        response_frame = tk.LabelFrame(paned_window, text="Ответ Системы", padx=5, pady=5)
        self.response_text = scrolledtext.ScrolledText(response_frame, wrap=tk.WORD, font=("Arial", 11))
        self.response_text.pack(fill=tk.BOTH, expand=True)
        paned_window.add(response_frame)

    def process_query(self):
        query = self.query_entry.get()
        if not query:
            return

        # Disable button during processing
        self.submit_button.config(state=tk.DISABLED)
        self.response_text.delete('1.0', tk.END)
        self.graph_text.delete('1.0', tk.END)
        self.response_text.insert(tk.END, "Обработка запроса...")
        self.update_idletasks() # Force GUI update

        # --- Capture print statements from the pipeline ---
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        # --- Run the pipeline ---
        # We create a new pipeline for each query to ensure a clean state
        pipeline = AGIPipeline()
        pipeline.command_network = FinalMockCommandNetwork() # Use the final mock
        pipeline.run_text_query(query)

        # Restore stdout
        sys.stdout = old_stdout

        # --- Display the results ---
        output_str = captured_output.getvalue()

        # A simple way to separate the graph summary from the rest of the log
        try:
            graph_summary = output_str.split("--- World Graph Summary ---")[1].split("-------------------------")[0]
            # The final response is not printed by the mock, so we'll simulate it
            # In a real system, the pipeline would return the response directly.
            final_response = "Система обработала запрос. Финальный ответ будет здесь, когда ReasoningNetwork будет обучена."
        except IndexError:
            graph_summary = "Не удалось распарсить состояние графа из лога."
            final_response = "Произошла ошибка при обработке."

        self.graph_text.delete('1.0', tk.END)
        self.graph_text.insert(tk.END, graph_summary.strip())

        self.response_text.delete('1.0', tk.END)
        self.response_text.insert(tk.END, final_response)

        # Re-enable button
        self.submit_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    app = App()
    app.mainloop()
