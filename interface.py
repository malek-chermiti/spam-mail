import os
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")


def load_resources():
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        return vectorizer, model
    except Exception as exc:
        messagebox.showerror(
            "Erreur de chargement",
            f"Impossible de charger le modèle ou le vectoriseur.\n{exc}",
        )
        raise


class SpamClassifierApp(tk.Tk):
    def __init__(self, vectorizer, model):
        super().__init__()
        self.title("Spam Classifier")
        self.geometry("560x420")
        self.resizable(False, False)
        self.configure(background="#111827")

        self.vectorizer = vectorizer
        self.model = model

        self.style = ttk.Style(self)
        self.configure_style()
        self.create_widgets()

    def configure_style(self):
        self.style.theme_use("clam")
        self.style.configure(
            "TLabel",
            background="#111827",
            foreground="#f8fafc",
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "Header.TLabel",
            font=("Segoe UI", 16, "bold"),
            foreground="#f8fafc",
        )
        self.style.configure(
            "Subheader.TLabel",
            foreground="#d1d5db",
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "Accent.TButton",
            foreground="#ffffff",
            background="#2563eb",
            font=("Segoe UI", 10, "bold"),
            padding=8,
        )
        self.style.map(
            "Accent.TButton",
            background=[("active", "#1d4ed8"), ("disabled", "#9ca3af")],
        )
        self.style.configure(
            "Secondary.TButton",
            foreground="#111827",
            background="#f8fafc",
            font=("Segoe UI", 10, "bold"),
            padding=8,
        )
        self.style.map(
            "Secondary.TButton",
            background=[("active", "#e5e7eb")],
        )
        self.style.configure("Result.TLabel", font=("Segoe UI", 11, "bold"), foreground="#f8fafc")

    def create_widgets(self):
        header_frame = ttk.Frame(self, padding=(16, 16, 16, 8), style="Card.TFrame")
        header_frame.pack(fill=tk.X)

        title = ttk.Label(header_frame, text="Détecteur de spam SMS", style="Header.TLabel")
        title.pack(anchor="w")

        subtitle = ttk.Label(
            header_frame,
            text="Saisissez un message et cliquez sur Prédire pour obtenir la probabilité de spam.",
            style="Subheader.TLabel",
            wraplength=520,
        )
        subtitle.pack(anchor="w", pady=(6, 0))

        content_frame = ttk.Frame(self, padding=(16, 12, 16, 16), style="Card.TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True)

        self.textarea = ScrolledText(
            content_frame,
            width=62,
            height=10,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            fg="#111827",
            bg="#f8fafc",
            relief=tk.FLAT,
            insertbackground="#111827",
        )
        self.textarea.pack(fill=tk.BOTH, expand=True)

        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill=tk.X, pady=(12, 8))

        self.predict_button = ttk.Button(
            button_frame,
            text="Prédire",
            style="Accent.TButton",
            command=self.predict,
        )
        self.predict_button.pack(side=tk.LEFT, padx=(0, 8))

        self.clear_button = ttk.Button(
            button_frame,
            text="Effacer",
            style="Secondary.TButton",
            command=self.clear_text,
        )
        self.clear_button.pack(side=tk.LEFT)

        self.result_var = tk.StringVar(value="Résultat : aucune prédiction pour le moment.")
        self.result_label = ttk.Label(self, textvariable=self.result_var, style="Result.TLabel", padding=(16, 10))
        self.result_label.pack(fill=tk.X)

    def clear_text(self):
        self.textarea.delete("1.0", tk.END)
        self.result_var.set("Résultat : aucune prédiction pour le moment.")

    def predict(self):
        message = self.textarea.get("1.0", tk.END).strip()
        if not message:
            messagebox.showwarning("Message vide", "Entrez un message pour effectuer une prédiction.")
            return

        try:
            X = self.vectorizer.transform([message])
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
        except Exception as exc:
            messagebox.showerror("Erreur", f"Impossible de faire la prédiction.\n{exc}")
            return

        label = "SPAM" if prediction == 1 else "HAM"
        ham_pct = probabilities[0] * 100
        spam_pct = probabilities[1] * 100
        self.result_var.set(f"Résultat : {label} — Ham {ham_pct:.2f}% / Spam {spam_pct:.2f}%")


if __name__ == "__main__":
    vectorizer, model = load_resources()
    app = SpamClassifierApp(vectorizer, model)
    app.mainloop()
