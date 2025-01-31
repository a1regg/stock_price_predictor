import sys
import datetime
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QLineEdit, QSpinBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QDateEdit, QMessageBox
)
from PyQt6.QtCore import Qt, QDate

# For Matplotlib embedding in PyQt6
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# ----------------------------------------------------
# 1) Model Import
# ----------------------------------------------------
from model import train_and_predict


# ----------------------------------------------------
# 2) Data Class for App Configuration
# ----------------------------------------------------
@dataclass
class AppConfig:
    """
    Holds user input values for the prediction.
    """
    ticker: str = "AAPL"
    start_date: datetime.date = datetime.date(2015, 1, 1)
    end_date: datetime.date = datetime.date(2023, 1, 1)
    window_size: int = 60
    future_days: int = 7


# ----------------------------------------------------
# 3) Matplotlib Canvas for PyQt6
# ----------------------------------------------------
class MatplotlibCanvas(FigureCanvasQTAgg):
    """
    Helper class to embed Matplotlib figures in PyQt6.
    """
    def __init__(self, figure, parent=None):
        super().__init__(figure)
        self.setParent(parent)


# ----------------------------------------------------
# 4) Main Window / UI
# ----------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 Stock Predictor (LSTM)")

        # Central widget for the window
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Overall layout
        self.main_layout = QVBoxLayout(self.central_widget)

        # 1) Input layout (horizontal)
        self.input_layout = QHBoxLayout()
        self.main_layout.addLayout(self.input_layout)

        # 2) Button layout
        self.button_layout = QHBoxLayout()
        self.main_layout.addLayout(self.button_layout)

        # 3) Plot layout
        self.plot_layout = QHBoxLayout()
        self.main_layout.addLayout(self.plot_layout)

        # Create + set up input widgets
        self.setup_input_widgets()
        self.setup_button()

        # Canvas placeholders
        self.canvas_test = None
        self.canvas_future = None

    # ----------------------------------------------------
    # A) Setup Input Widgets
    # ----------------------------------------------------
    def setup_input_widgets(self):
        """Create widgets for user input fields, place them in input_layout."""
        # Ticker
        self.label_ticker = QLabel("Ticker Symbol:")
        self.line_ticker = QLineEdit("AAPL")

        # Start Date
        self.label_start = QLabel("Start Date:")
        self.date_start = QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_start.setDate(QDate(2015, 1, 1))

        # End Date
        self.label_end = QLabel("End Date:")
        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        self.date_end.setDate(QDate(2025, 1, 1))

        # Window Size
        self.label_window = QLabel("Window Size:")
        self.spin_window = QSpinBox()
        self.spin_window.setRange(1, 365)
        self.spin_window.setValue(60)

        # Future Days
        self.label_future = QLabel("Future Days:")
        self.spin_future = QSpinBox()
        self.spin_future.setRange(1, 60)
        self.spin_future.setValue(7)

        # Add them in a row
        self.input_layout.addWidget(self.label_ticker)
        self.input_layout.addWidget(self.line_ticker)
        self.input_layout.addWidget(self.label_start)
        self.input_layout.addWidget(self.date_start)
        self.input_layout.addWidget(self.label_end)
        self.input_layout.addWidget(self.date_end)
        self.input_layout.addWidget(self.label_window)
        self.input_layout.addWidget(self.spin_window)
        self.input_layout.addWidget(self.label_future)
        self.input_layout.addWidget(self.spin_future)

    # ----------------------------------------------------
    # B) Setup Button
    # ----------------------------------------------------
    def setup_button(self):
        """Create the Predict button and connect the signal."""
        self.btn_predict = QPushButton("Predict")
        self.btn_predict.clicked.connect(self.on_predict)
        self.button_layout.addWidget(self.btn_predict)

    # ----------------------------------------------------
    # C) Prediction Logic
    # ----------------------------------------------------
    def on_predict(self):
        """When the Predict button is clicked, gather inputs and run model."""
        # Gather user inputs into AppConfig
        config = AppConfig(
            ticker=self.line_ticker.text().strip(),
            start_date=self.qdate_to_date(self.date_start.date()),
            end_date=self.qdate_to_date(self.date_end.date()),
            window_size=self.spin_window.value(),
            future_days=self.spin_future.value()
        )

        # Call the train_and_predict function from model.py
        fig_test, fig_future = train_and_predict(
            config.ticker,
            config.start_date,
            config.end_date,
            config.window_size,
            config.future_days
        )

        # Clear old canvases
        self.clear_old_plots()

        if fig_test is None or fig_future is None:
            # Means no data was returned, show an error message
            QMessageBox.warning(self, "Error", "No data found for this ticker/date range.")
            return

        # Build new canvases for the returned figures
        self.canvas_test = MatplotlibCanvas(fig_test)
        self.canvas_future = MatplotlibCanvas(fig_future)

        # Add them to layout
        self.plot_layout.addWidget(self.canvas_test)
        self.plot_layout.addWidget(self.canvas_future)

        # Redraw
        self.canvas_test.draw()
        self.canvas_future.draw()

    # ----------------------------------------------------
    # D) Helper Methods
    # ----------------------------------------------------
    def clear_old_plots(self):
        """Remove old plot canvases from the layout."""
        for i in reversed(range(self.plot_layout.count())):
            item = self.plot_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    @staticmethod
    def qdate_to_date(qdate):
        """Helper to convert QDate -> Python datetime.date."""
        return datetime.date(qdate.year(), qdate.month(), qdate.day())


# ----------------------------------------------------
# 5) Entry Point
# ----------------------------------------------------
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
