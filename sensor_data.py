import csv
from datetime import datetime
import os
import sys
import traceback

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QTableWidget, QHeaderView,
                             QCheckBox, QDateEdit, QMessageBox, QTableWidgetItem)
from PyQt5.QtCore import QDate
from PyQt5.QtGui import QFont

class GroundData(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.plot_id_to_index_mapping = {
            '0013A200415D93A4': 1,
            '0013A2004180663B': 2,
            '0013A20041A998EC': 3,
            '0013A20041A99973': 4,
            '0013A20041A99919': 5,
            '0013A20041AEEABE': 6,
            '0013A200416DC52F': 7,
            '0013A200416DC540': 8,
            '0013A20041AEEA65': 9,
            '0013A20041AEEA54': 10,
            '0013A20041AEEA55': 11,
            '0013A20041A998FC': 12,
            '0013A200419AD6EF': 13,
            '0013A20041A99922': 14,
            '0013A200419AD6FB': 15,
            '0013A200417964E5': 16,
            '0013A20041A99927': 17,
            '0013A200416DC530': 18,
            '0013A20041AEEA44': 19,
            '0013A20041AEEA3B': 20,
            '0013A20041A998F6': 21,
            '0013A20041A998E8': 22,
            '0013A20041AEEAB2': 23,
            '0013A20041A998DE': 24,
        }
        self.outpudir = ''

    def initUI(self):
        self.setWindowTitle('Enhanced Ground Temperature Data Processor')
        self.setGeometry(100, 100, 800, 600)
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)

        # Styling
        font = QFont('Arial', 10)

        # Upload Button and File Name Display
        self.fileLabel = QLabel('No file selected', self)
        self.fileLabel.setFont(font)
        self.fileLabel.setStyleSheet("color:Black")
        uploadButton = QPushButton('Upload File', self)
        uploadButton.setFont(font)
        uploadButton.clicked.connect(self.uploadFile)
        fileLayout = QHBoxLayout()
        fileLayout.addWidget(uploadButton)
        fileLayout.addWidget(self.fileLabel)
        layout.addLayout(fileLayout)

        # Date Picker for Start and End Date
        dateLayout = QHBoxLayout()
        self.startDateEdit = QDateEdit(calendarPopup=True)
        self.endDateEdit = QDateEdit(calendarPopup=True)
        for widget in [self.startDateEdit, self.endDateEdit]:
            widget.setDate(QDate.currentDate())
            widget.setFont(font)
        dateLayout.addWidget(QLabel("<font color='Black'>Start Date:</font>"))
        dateLayout.addWidget(self.startDateEdit)
        dateLayout.addWidget(QLabel("<font color = 'Black'>End Date:</font>"))
        dateLayout.addWidget(self.endDateEdit)
        layout.addLayout(dateLayout)
        plotsLabel = QLabel('<font color = "Black">Check the plot you want, from the listed</font>')
        plotsLabel.setFont(font)
        layout.addWidget(plotsLabel)


        # Plot Selection Checkboxes;p'
        plotLayout = QHBoxLayout()
        listOf_numberLayout = QHBoxLayout()
        self.numbLabels = []
        self.plotCheckBoxes = []
        for i in range(1, 25):
            checkBox = QCheckBox(f"", self)
            checkBox.setFont(QFont('Arial', 5))
            plotLayout.addWidget(checkBox)
            self.plotCheckBoxes.append(checkBox)

            label = QLabel(f"{i}", self)
            label.setStyleSheet('color:Black;')
            #checkBox.setFont(QFont('Arial', 5))
            listOf_numberLayout.addWidget(label)
            self.numbLabels.append(label)
        layout.addLayout(plotLayout)
        layout.addLayout(listOf_numberLayout)

        # Buttons for Various Actions
        buttonLayout = QHBoxLayout()
        for btn_text in ["Process Data", "Save Data", "Reset", "Calculate Mean", "Save Mean"]:
            button = QPushButton(btn_text, self)
            button.setFont(font)
            button.clicked.connect(self.buttonClicked)
            buttonLayout.addWidget(button)
        layout.addLayout(buttonLayout)

        # Data Display Table
        self.table = QTableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Plot', 'Temperature', 'Date'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        self.showMaximized()

    def uploadFile(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open file", "", "Text files (*.txt *.csv)")
        if file_path:
            self.global_file_path = file_path
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            file_info = f"{file_name} ({file_size / 1024:.2f} KB)"
            self.fileLabel.setText(file_info)

    def buttonClicked(self):
        button = self.sender()
        if button.text() == "Process Data":
            self.processData()
        elif button.text() == "Save Data":
            self.saveData()
        elif button.text() == "Reset":
            self.reset()
        elif button.text() == "Calculate Mean":
            self.calculateMean()
        elif button.text() == "Save Mean":
            self.saveMean()

    # Placeholder methods for button actions
    import csv

    import csv
    import os

    def processData(self):
        try:
            if self.global_file_path is None:
                QMessageBox.warning(self, "Error", "No file selected.")
                return

            start_date = self.startDateEdit.date().toPyDate()
            end_date = self.endDateEdit.date().toPyDate()
            selected_plots = [i for i, checkBox in enumerate(self.plotCheckBoxes, start=1) if checkBox.isChecked()]

            print("Selected Start Date:", start_date)
            print("Selected End Date:", end_date)
            print("Selected Plots:", selected_plots)

            self.table.setRowCount(0)
            processed_data = []

            with open(self.global_file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split(',')
                    if len(parts) < 7:
                        print("Skipping line due to insufficient parts:", line)
                        continue

                    plot_identifier = parts[5].strip()
                    plot_index = self.plot_id_to_index_mapping.get(plot_identifier)

                    if plot_index is None:
                        print(f"No mapping found for plot identifier: {plot_identifier}")
                        continue

                    if plot_index not in selected_plots:
                        print(f"Plot index {plot_index} not in selected plots.")
                        continue

                    temperature = parts[3].strip().replace('+', '')
                    date_str = ' '.join(parts[6:]).strip()
                    date_time = datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y")

                    if not (start_date <= date_time.date() <= end_date):
                        print(f"Date {date_time} out of range.")
                        continue

                    processed_data.append([plot_index, temperature, date_time.strftime("%Y-%m-%d")])

                    row_position = self.table.rowCount()
                    self.table.insertRow(row_position)
                    self.table.setItem(row_position, 0, QTableWidgetItem(str(plot_index)))
                    self.table.setItem(row_position, 1, QTableWidgetItem(temperature))
                    self.table.setItem(row_position, 2, QTableWidgetItem(date_time.strftime("%Y-%m-%d")))

            self.processed_data = processed_data  # Store processed data for later use

            QMessageBox.information(self, "Info", "Data processing complete.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
            print(f"An error occurred: {traceback.format_exc()}")

    def saveData(self):
        try:
            if not hasattr(self, 'processed_data'):
                QMessageBox.warning(self, "Error", "No data to save.")
                return

            output_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Files")
            self.outpudir = output_dir

            if not output_dir:
                QMessageBox.warning(self, "Error", "No directory selected.")
                return

            for plot_index in set(row[0] for row in self.processed_data):
                for date in set(row[2] for row in self.processed_data):
                    file_name = os.path.join(output_dir, f"{date}_plot{plot_index}.csv")
                    data_for_file = [[row[1], row[2]] for row in self.processed_data if
                                     row[0] == plot_index and row[2] == date]

                    with open(file_name, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['Temperature', 'Date'])
                        writer.writerows(data_for_file)

            QMessageBox.information(self, "Info", "Data saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
            print(f"An error occurred: {traceback.format_exc()}")
        with open('savepath.txt', 'w') as file:
            file.write(self.outpudir)

    def reset(self):
        # Implement reset logic
        QMessageBox.information(self, "Info", "Reset clicked")

    def calculateMean(self):
        # Implement mean calculation logic
        QMessageBox.information(self, "Info", "Calculate Mean clicked")

    def saveMean(self):
        # Implement save mean logic
        QMessageBox.information(self, "Info", "Save Mean clicked")


