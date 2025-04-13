from re import escape
import sys
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QPushButton, QHBoxLayout, QCheckBox, QComboBox, QSizePolicy, QTabWidget
)
from PySide6.QtGui import QTextCursor, QPixmap, QFontDatabase
from PySide6.QtCore import Qt, QTimer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from tamil_sentiment_classifier.llama_classifier import LlamaClassifier
from tamil_sentiment_classifier.muril_classifier import MurilClassifier
from tamil_sentiment_classifier.xlmr_classifier import XLMRClassifier

MODEL_MAP = {
    "llama": LlamaClassifier,
    "muril": MurilClassifier,
    "xlmr": XLMRClassifier
}

class TamilClassifierGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tamil Sentiment Analyzer")
        self.resize(800, 600)

        self.transliteration_enabled = True
        self.model_type = "llama"
        self.classifier = MODEL_MAP[self.model_type]()
        self.canvas = None

        self.initUI()
        self.setStyleSheet(self.styles())
        self.update_tabs_visibility()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Title
        self.title_bar = QHBoxLayout()
        image_path = os.path.join("images", "nitk_logo.png")
        self.image_placeholder = QLabel()
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_placeholder.setText("Logo")
        else:
            self.image_placeholder.setPixmap(pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.image_placeholder.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title_text = QLabel("SCaLAR Sentiment")
        self.title_text.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter | Qt.AlignRight)
        self.title_bar.addWidget(self.image_placeholder)
        self.title_bar.addStretch()
        self.title_bar.addWidget(self.title_text)
        self.animate_label_color(
            colors=["#2F4F4F", "#8B0000", "#6A5ACD", "#556B2F", "#4682B4"],
            font_sizes=[22, 24, 26, 24]
        )
        layout.addLayout(self.title_bar)

        # Input
        self.label = QLabel("Enter Tamil or Tanglish (SPACE to convert):")
        layout.addWidget(self.label)

        self.text_box = QTextEdit()
        self.text_box.setPlaceholderText("Type your Tamil or Tanglish text here...")
        self.text_box.setMinimumHeight(60)
        self.text_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.text_box.textChanged.connect(self.transliterate_on_space)
        layout.addWidget(self.text_box)

        # Controls
        controls = QHBoxLayout()
        self.translit_checkbox = QCheckBox("Transliterate (Tanglish → Tamil)")
        self.translit_checkbox.setChecked(True)
        self.translit_checkbox.stateChanged.connect(self.toggle_transliteration)
        controls.addWidget(self.translit_checkbox)

        self.model_selector = QComboBox()
        self.model_selector.addItems(MODEL_MAP.keys())
        self.model_selector.currentTextChanged.connect(self.change_model)
        controls.addWidget(self.model_selector)

        self.classify_button = QPushButton("Classify")
        self.classify_button.clicked.connect(self.classify_text)
        controls.addWidget(self.classify_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_inputs)
        controls.addWidget(self.clear_button)
        layout.addLayout(controls)

        # Tabs
        self.result_label = QLabel("Output:")
        layout.addWidget(self.result_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        self.tabs.addTab(self.result_output, "Classification Output")

        self.explanation_output = QTextEdit()
        self.explanation_output.setReadOnly(True)
        self.tabs.addTab(self.explanation_output, "Explanation Analysis")

        self.sentiment_bar_output = QWidget()
        self.sentiment_bar_layout = QVBoxLayout(self.sentiment_bar_output)
        self.tabs.addTab(self.sentiment_bar_output, "Sentiment Bar")

    def animate_label_color(self, colors, font_sizes, index=0):
        fancy_font = "Old English Text MT"
        available_fonts = QFontDatabase().families()
        font_family = fancy_font if fancy_font in available_fonts else "Times New Roman"

        color = colors[index % len(colors)]
        font_size = font_sizes[index % len(font_sizes)]

        self.title_text.setStyleSheet(
            f"""
            font-size: {font_size}px;
            font-weight: bold;
            font-style: italic;
            color: {color};
            font-family: '{font_family}';
            """
        )
        QTimer.singleShot(1000, lambda: self.animate_label_color(colors, font_sizes, index + 1))

    def toggle_transliteration(self):
        self.transliteration_enabled = self.translit_checkbox.isChecked()

    def transliterate_on_space(self):
        if not self.transliteration_enabled:
            return
        text = self.text_box.toPlainText()
        if text.endswith(" "):
            words = text.split()
            if words:
                try:
                    words[-1] = transliterate(words[-1], sanscript.ITRANS, sanscript.TAMIL)
                    updated = " ".join(words) + " "
                    self.text_box.blockSignals(True)
                    self.text_box.setPlainText(updated)
                    self.text_box.moveCursor(QTextCursor.End)
                    self.text_box.blockSignals(False)
                except Exception:
                    pass

    def change_model(self, model_name):
        if model_name != self.model_type:
            self.model_type = model_name
            self.classifier = MODEL_MAP[self.model_type]()
            self.clear_outputs()
            self.update_tabs_visibility()

    def update_tabs_visibility(self):
        is_llama = self.model_type == "llama"
        self.tabs.setTabVisible(1, not is_llama)  # Explanation tab
        self.tabs.setTabVisible(2, not is_llama)  # Sentiment Bar

    def classify_text(self):
        input_text = self.text_box.toPlainText().strip()
        if not input_text:
            return

        prediction = self.classifier.classify(input_text)
        self.result_output.setPlainText(f"Prediction: {prediction}")

        if self.model_type == "llama":
            self.explanation_output.clear()
            self.sentiment_bar_layout.setEnabled(False)
        else:
            self.show_explanation(input_text, prediction)
            self.show_sentiment_bar(input_text)

    def show_explanation(self, input_text, prediction):
        exp = self.classifier.explain(input_text)
        highlighted = self.highlight_text(input_text, exp)
        word_contribs = "".join([
            f'<span style="color:{"green" if weight > 0 else "red"}; font-weight:bold;">{word}: {weight:.4f}</span><br>'
            for word, weight in exp.as_list()
        ])
        self.explanation_output.setHtml(f"""
            <b>Prediction:</b> {prediction}<br><br>
            <b>Input with Highlights:</b><br><br>{highlighted}<br><br>
            <b>Word Contribution:</b><br>{word_contribs}
        """)

    def highlight_text(self, text, explanation):
        # Iterate over the list of words and weights to apply highlighting
        for word, weight in sorted(explanation.as_list(), key=lambda x: -len(x[0])):
            # Use a stronger intensity for color visibility
            intensity = min(255, int(abs(weight) * 255))  # Use a max of 255 for intensity

            # Choose colors based on intensity and weight (positive or negative)
            if weight > 0:
                bg_color = f"rgb(0, 255, 0)"  # Bright Green for positive words
                text_color = "black"
            else:
                bg_color = f"rgb(255, 0, 0)"  # Bright Red for negative words
                text_color = "white"

            # Add padding and border radius to make the highlight more noticeable
            span = f'<span style="background-color:{bg_color}; color:{text_color}; padding:4px; border-radius:4px;">{word}</span>'
            
            # Replace the word in the text with its highlighted version
            text = text.replace(word, span, 1)

        return text


    def show_sentiment_bar(self, input_text):
        probs = self.classifier.get_probs(input_text)
        class_names = list(self.classifier.label_map.values())
        if not probs:
            return

        if self.canvas:
            self.sentiment_bar_layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None

        values = [probs[c] for c in class_names]
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ["red" if c.lower() == "negative" else "green" for c in class_names]
        ax.bar(class_names, values, color=colors)
        ax.set_title("Sentiment Confidence")
        fig.tight_layout()

        self.canvas = FigureCanvas(fig)
        self.sentiment_bar_layout.addWidget(self.canvas)
        self.canvas.draw()

    def clear_outputs(self):
        self.result_output.clear()
        self.explanation_output.clear()
        if self.canvas:
            self.sentiment_bar_layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None

    def clear_inputs(self):
        self.text_box.clear()
        self.clear_outputs()

    def styles(self):
        return """
        QWidget {
            background-color: #f5f5f5;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }

        QLabel {
            font-weight: bold;
            color: #333;
        }

        QTextEdit {
            background-color: #ffffff;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 6px;
        }

        QComboBox {
            padding: 6px;
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 4px;
            color: #333;
        }

        QCheckBox, QPushButton {
            padding: 6px;
        }

        QPushButton {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
        }

        QPushButton:hover {
            background-color: #0069d9;
        }

        QPushButton:pressed {
            background-color: #005cbf;
        }
        """

def main():
    app = QApplication(sys.argv)
    window = TamilClassifierGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
