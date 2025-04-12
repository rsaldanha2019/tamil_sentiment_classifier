import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QPushButton, QHBoxLayout, QCheckBox, QComboBox, QSizePolicy
)
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import Qt
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

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
        self.setWindowTitle("Tamil Text Sentiment Classifier")
        self.resize(800, 600)

        self.transliteration_enabled = True
        self.model_type = "llama"
        self.classifier = MODEL_MAP[self.model_type]()

        self.initUI()
        # Hide the Explain button if initial model is llama
        if self.model_type == "llama":
            self.explain_button.setVisible(False)
        self.setStyleSheet(self.styles())

    def initUI(self):
        layout = QVBoxLayout(self)

        self.label = QLabel("Enter Tamil or Tanglish (SPACE to convert):")
        layout.addWidget(self.label)

        self.text_box = QTextEdit()
        self.text_box.setPlaceholderText("Type your Tamil or Tanglish text here...")
        self.text_box.setMinimumHeight(120)
        self.text_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.text_box.textChanged.connect(self.transliterate_on_space)
        layout.addWidget(self.text_box)

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

        self.explain_button = QPushButton("Explain")
        self.explain_button.clicked.connect(self.explain_text)
        controls.addWidget(self.explain_button)

        layout.addLayout(controls)

        self.result_label = QLabel("Prediction / Explanation Output:")
        layout.addWidget(self.result_label)

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        self.result_output.setMinimumHeight(200)
        self.result_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.result_output)

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

        # Toggle visibility of Explain button
        if self.model_type == "llama":
            self.explain_button.setVisible(False)
        else:
            self.explain_button.setVisible(True)

    def classify_text(self):
        input_text = self.text_box.toPlainText().strip()
        if input_text:
            prediction = self.classifier.classify(input_text)
            self.result_output.setPlainText(str(prediction))
            self.result_output.moveCursor(QTextCursor.End)

    def explain_text(self):
        input_text = self.text_box.toPlainText().strip()
        if not input_text:
            return

        self.result_output.append("\n=== Explanation ===")
        self.result_output.moveCursor(QTextCursor.End)
        exp = self.classifier.explain(input_text)

        if hasattr(self, "canvas") and self.canvas:
            self.layout().removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None

        if exp:
            words, weights = zip(*exp.as_list())
            y_pos = np.arange(len(words))

            fig = Figure(figsize=(6, 3))
            ax = fig.add_subplot(111)
            ax.barh(y_pos, weights, align='center', color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.invert_yaxis()
            ax.set_xlabel('Weight')
            ax.set_title('LIME Explanation')

            self.canvas = FigureCanvas(fig)
            self.layout().addWidget(self.canvas)

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

        QComboBox QAbstractItemView {
            background-color: #ffffff;
            selection-background-color: #007bff;
            selection-color: #ffffff;
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
