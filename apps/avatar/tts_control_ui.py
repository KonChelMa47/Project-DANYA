from __future__ import annotations

import json
import sys

from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class TTSControlWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DANYA TTS Control")
        self.resize(820, 560)
        self.setMinimumSize(640, 460)
        self.setStyleSheet(
            """
            QWidget {
                background: #f5f6f8;
                color: #22262d;
                font-family: "Noto Sans CJK JP", "Noto Sans", sans-serif;
                font-size: 13px;
            }
            QLabel#title {
                font-size: 20px;
                font-weight: 700;
                color: #151922;
            }
            QLabel#hint {
                color: #657080;
            }
            QLabel#sectionLabel {
                font-size: 12px;
                font-weight: 700;
                color: #4d5664;
            }
            QComboBox, QTextEdit {
                background: #ffffff;
                border: 1px solid #cfd5dd;
                border-radius: 6px;
                padding: 8px;
            }
            QTextEdit#speechText {
                font-size: 16px;
                line-height: 1.35;
            }
            QTextEdit#history {
                color: #454c57;
                background: #ffffff;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #c8ced8;
                border-radius: 6px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background: #eef2f7;
            }
            QPushButton#sendButton {
                background: #1f6feb;
                border: 1px solid #1f6feb;
                color: white;
                font-weight: 700;
                padding: 10px 18px;
            }
            QPushButton#sendButton:hover {
                background: #185fc8;
            }
            QPushButton.quick {
                padding: 7px 10px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("DANYA TTS Control")
        title.setObjectName("title")
        layout.addWidget(title)

        help_label = QLabel("Choose a voice expression, write the line, and send it to DANYA.")
        help_label.setObjectName("hint")
        help_label.setWordWrap(True)
        layout.addWidget(help_label, stretch=0)

        controls = QGridLayout()
        controls.setHorizontalSpacing(12)
        controls.setVerticalSpacing(8)

        emotion_label = QLabel("Emotion")
        emotion_label.setObjectName("sectionLabel")
        controls.addWidget(emotion_label, 0, 0)
        self.emotion = QComboBox()
        self.emotion.addItems(["happy", "sad", "angry", "surprised", "fear", "neutral"])
        controls.addWidget(self.emotion, 0, 1)

        intensity_label = QLabel("Intensity")
        intensity_label.setObjectName("sectionLabel")
        controls.addWidget(intensity_label, 0, 2)
        self.intensity = QComboBox()
        self.intensity.addItems(["normal", "high"])
        controls.addWidget(self.intensity, 0, 3)

        quick_grid = QGridLayout()
        quick_grid.setHorizontalSpacing(8)
        quick_grid.setVerticalSpacing(8)
        quick_buttons = [
            ("Happy", "happy", "normal"),
            ("Happy High", "happy", "high"),
            ("Sad", "sad", "normal"),
            ("Sad High", "sad", "high"),
            ("Angry", "angry", "normal"),
            ("Surprised", "surprised", "normal"),
            ("Fear", "fear", "normal"),
            ("Neutral", "neutral", "normal"),
        ]
        for index, (label, emotion, intensity) in enumerate(quick_buttons):
            button = QPushButton(label)
            button.setProperty("class", "quick")
            button.clicked.connect(lambda _checked=False, e=emotion, i=intensity: self.set_emotion(e, i))
            quick_grid.addWidget(button, index // 4, index % 4)
        controls.addLayout(quick_grid, 1, 0, 1, 4)
        layout.addLayout(controls)

        text_label = QLabel("Text")
        text_label.setObjectName("sectionLabel")
        layout.addWidget(text_label)
        self.text = QTextEdit()
        self.text.setObjectName("speechText")
        self.text.setPlaceholderText("こんにちは。ダーニャです。")
        layout.addWidget(self.text, stretch=2)

        input_row = QHBoxLayout()
        input_row.addStretch(1)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.text.clear)
        input_row.addWidget(clear_button)

        hide_button = QPushButton("Hide")
        hide_button.clicked.connect(self.hide)
        input_row.addWidget(hide_button)

        send_button = QPushButton("Send")
        send_button.setObjectName("sendButton")
        send_button.clicked.connect(self.submit)
        input_row.addWidget(send_button)

        layout.addLayout(input_row)

        history_label = QLabel("History")
        history_label.setObjectName("sectionLabel")
        layout.addWidget(history_label)
        self.history = QTextEdit()
        self.history.setObjectName("history")
        self.history.setReadOnly(True)
        layout.addWidget(self.history, stretch=1)

        self.text.setFocus()

    def set_emotion(self, emotion: str, intensity: str) -> None:
        self.emotion.setCurrentText(emotion)
        self.intensity.setCurrentText(intensity)
        self.text.setFocus()

    def submit(self) -> None:
        text = self.text.toPlainText().strip()
        if not text:
            return
        emotion = self.emotion.currentText()
        intensity = self.intensity.currentText()
        payload = {"text": text}
        if emotion != "neutral":
            payload["emotion"] = emotion
            payload["intensity"] = intensity
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        label = "neutral" if emotion == "neutral" else f"{emotion}_{intensity}"
        self.history.append(f"[{label}] {text}")
        self.text.clear()
        self.text.setFocus()

    def closeEvent(self, event) -> None:  # noqa: N802
        event.ignore()
        self.hide()


def main() -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = TTSControlWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
