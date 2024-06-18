import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextBrowser,
    QHBoxLayout, QDesktopWidget, QAction, QMenu
)
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QBrush, QPen, QFont
from PyQt5.QtCore import Qt, QRectF, QPoint


class ShadowButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.color = QColor("#c0392b")
        self.shadow_color = QColor("cc0000")
        self.hover_color = QColor("#cc0000")
        self.shadow_offset = 3
        self.setMinimumHeight(40)  # Setting a minimum height for the buttons
        self.setStyleSheet("color: white")  # Setting text color to white

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw shadow
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, self.shadow_offset, self.width(), self.height()), 10, 10)
        painter.fillPath(path, QBrush(self.shadow_color))

        # Draw button
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.width(), self.height() - self.shadow_offset), 10, 10)
        painter.fillPath(path, QBrush(self.color))

        painter.drawText(QPoint(int(self.width() / 2 - painter.fontMetrics().width(self.text()) / 2),
                                int(self.height() / 2 + painter.fontMetrics().height() / 3)), self.text())

    def enterEvent(self, event):
        self.color = self.hover_color
        self.update()

    def leaveEvent(self, event):
        self.color = QColor("#2ECC71")
        self.update()


class Analysis(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Analylsis")
        self.setGeometry(100, 100, 700, 500)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet("background-color: 	#003366;")  # background

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.menu_buttons_container = QWidget()
        self.menu_buttons_layout = QHBoxLayout()
        self.menu_buttons_container.setLayout(self.menu_buttons_layout)
        self.menu_buttons_container.setStyleSheet("background-color: #00264d;")  # background for buttons

        self.layout.addWidget(self.menu_buttons_container)

        self.display_widget = QTextBrowser()
        self.display_widget.setStyleSheet(
            """
            QTextBrowser {
                font-size: 14pt;
                padding: 10px;
                line-height: 1.5;
                background-color: #ffffff; /* White background for text display */
                color: #333333; /* Dark text color */
            }
            """
        )
        self.layout.addWidget(self.display_widget)

        self.create_menu_bar()

        self.move(QDesktopWidget().availableGeometry().center())
        self.center()
        self.showMaximized()

    def create_menu_bar(self):
        menubar = self.menuBar()

        dropdown_styles = [
            ("Statistics", ['Ground', 'Thermal', 'Correlation', 'Correlation Graph']),
            ("Bland-Altman", ['Graph1', 'Graph2']),
            ("Error Metrics", ['MAE', 'MAE Graph', 'RMSE', 'RMSE Graph']),
            ("Visualize", ['Time Series', 'Scatter Plot']),
            ("Hypothesis", ['ANOVA', 'ANOVA Graph']),
            ("Other", ['1', '2']),
        ]

        for name, buttons in dropdown_styles:
            dropdown_container = self.create_dropdown_container(name, buttons)
            self.menu_buttons_layout.addWidget(dropdown_container)

    def create_dropdown_container(self, menu_name, buttons):
        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)

        dropdown_menu = QMenu(menu_name)

        for button_name in buttons:
            action = QAction(button_name, self)
            if button_name == 'Ground':
                action.triggered.connect(self.show_Gnd_statistics)
            # Connect other buttons to respective functions
            # For now, leaving other buttons disconnected
            if button_name == 'Thermal':
                action.triggered.connect(self.show_Ther_statistics)
            if button_name == 'Correlation':
                action.triggered.connect(self.correlation)
            if button_name == 'MAE':
                action.triggered.connect(self.MAE)
            if button_name == 'RMSE':
                action.triggered.connect(self.RMSE)
            if button_name == 'ANOVA':
                action.triggered.connect(self.ANOVA)
            if button_name == 'Graph1':
                action.triggered.connect(self.Graph1)
            if button_name == 'Graph2':
                action.triggered.connect(self.Graph2)
            if button_name == 'Correlation Graph':
                action.triggered.connect(self.Graph2)
            if button_name == 'MAE Graph':
                action.triggered.connect(self.MAEGraph)
            if button_name == 'RMSE Graph':
                action.triggered.connect(self.RMSEGraph)
            if button_name == 'Time Series':
                action.triggered.connect(self.TimeSeriesGraph)
            if button_name == 'Scatter Plot':
                action.triggered.connect(self.scatter_plots)
            if button_name == 'ANOVA Graph':
                action.triggered.connect(self.anovaGraph)

            dropdown_menu.addAction(action)

        dropdown_button = ShadowButton(menu_name)
        dropdown_button.setMenu(dropdown_menu)
        dropdown_button.setCursor(Qt.PointingHandCursor)

        container_layout.addWidget(dropdown_button)

        return container

    def show_Gnd_statistics(self):
        self.display_widget.clear()

        # Code for calculating summary statistics and correlation matrix
        columns = ['day', 'year'] + [f'plot{i}' for i in range(1, 25)]

        def generate_temperature_data():
            data = {
                'day': np.arange(1, 101),  # Sequential days from 1 to 100
                'year': np.random.randint(2000, 2023, 100),  # Years from 2000 to 2022
            }
            for i in range(1, 25):
                data[f'plot{i}'] = np.random.uniform(20, 30, 100)  # Random data for plot columns within the range
            return data

        g_truth_data = pd.DataFrame(generate_temperature_data(), columns=columns)
        thermal_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        g_truth_stats = g_truth_data.describe().T
        g_truth_summary = g_truth_stats[['mean', '50%', 'std', 'min', 'max']]

        stats_str = "Summary Statistics for g_truth_data:\n"
        stats_str += g_truth_summary.to_string()
        stats_str += "\n\n"

        self.display_widget.setPlainText(stats_str)

    def show_Ther_statistics(self):
        self.display_widget.clear()

        # Creating column headers
        columns = ['day', 'year'] + [f'plot{i}' for i in range(1, 25)]

        # Function to generate temperature data with sequential days
        def generate_temperature_data():
            data = {
                'day': np.arange(1, 101),  # Sequential days from 1 to 100
                'year': np.random.randint(2000, 2023, 100),  # Assuming years from 2000 to 2022
            }
            for i in range(1, 25):
                data[f'plot{i}'] = np.random.uniform(20, 30, 100)  # Random data for plot columns within the range
            return data

        # Generating ground truth temperature data with sequential days
        g_truth_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Generating thermal image temperature data with sequential days
        thermal_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Calculating summary statistics for each plot in thermal_data
        thermal_stats = thermal_data.describe().T  # Transpose for easier plotting

        # Extracting relevant statistics (mean, median, std, min, max) for plot columns
        thermal_summary = thermal_stats[['mean', '50%', 'std', 'min', 'max']]

        # Displaying summary statistics for thermal_data
        stats_str = "Summary Statistics for thermal_data:\n"
        stats_str += thermal_summary.to_string()

        self.display_widget.setPlainText(stats_str)

    def correlation(self):
        self.display_widget.clear()

        # Creating column headers
        columns = ['day', 'year'] + [f'plot{i}' for i in range(1, 25)]

        # Function to generate temperature data with sequential days
        def generate_temperature_data():
            data = {
                'day': np.arange(1, 101),  # Sequential days from 1 to 100
                'year': np.random.randint(2000, 2023, 100),  # Assuming years from 2000 to 2022
            }
            for i in range(1, 25):
                data[f'plot{i}'] = np.random.uniform(20, 30, 100)  # Random data for plot columns within the range
            return data

        # Generating ground truth temperature data with sequential days
        g_truth_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Generating thermal image temperature data with sequential days
        thermal_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Calculating summary statistics for each plot in thermal_data
        thermal_stats = thermal_data.describe().T  # Transpose for easier plotting
        # Calculating the correlation matrix between g_truth_data and thermal_data
        correlation_matrix = g_truth_data.corrwith(thermal_data, axis=0)

        # Displaying the correlation matrix
        stats_str = "Correlation Matrix between g_truth_data and thermal_data:\n"
        stats_str += correlation_matrix.to_string()

        self.display_widget.setPlainText(stats_str)

    def MAE(self):
        self.display_widget.clear()

        # Creating column headers
        columns = ['day', 'year'] + [f'plot{i}' for i in range(1, 25)]

        # Function to generate temperature data with sequential days
        def generate_temperature_data():
            data = {
                'day': np.arange(1, 101),  # Sequential days from 1 to 100
                'year': np.random.randint(2000, 2023, 100),  # Assuming years from 2000 to 2022
            }
            for i in range(1, 25):
                data[f'plot{i}'] = np.random.uniform(20, 30, 100)  # Random data for plot columns within the range
            return data

        # Generating ground truth temperature data with sequential days
        g_truth_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Generating thermal image temperature data with sequential days
        thermal_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Calculating absolute differences between measurements for each plot
        absolute_differences = abs(g_truth_data.iloc[:, 2:] - thermal_data.iloc[:, 2:])

        # Calculating Mean Absolute Error (MAE) for each plot
        mae_per_plot = absolute_differences.mean()

        # Displaying the Mean Absolute Error (MAE) for each plot
        stats_str = "Mean Absolute Error (MAE) for each plot:\n"
        stats_str += mae_per_plot.to_string()

        self.display_widget.setPlainText(stats_str)

    def RMSE(self):
        self.display_widget.clear()

        # Creating column headers
        columns = ['day', 'year'] + [f'plot{i}' for i in range(1, 25)]

        # Function to generate temperature data with sequential days
        def generate_temperature_data():
            data = {
                'day': np.arange(1, 101),  # Sequential days from 1 to 100
                'year': np.random.randint(2000, 2023, 100),  # Assuming years from 2000 to 2022
            }
            for i in range(1, 25):
                data[f'plot{i}'] = np.random.uniform(20, 30, 100)  # Random data for plot columns within the range
            return data

        # Generating ground truth temperature data with sequential days
        g_truth_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Generating thermal image temperature data with sequential days
        thermal_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Calculating squared differences between measurements
        squared_differences = (g_truth_data.iloc[:, 2:] - thermal_data.iloc[:, 2:]) ** 2

        # Calculating mean of squared differences for each plot
        mean_squared_differences = squared_differences.mean()

        # Calculating RMSE for each plot
        rmse_per_plot = np.sqrt(mean_squared_differences)

        # Displaying the RMSE for each plot
        stats_str = "Root Mean Squared Error (RMSE) for each plot:\n"
        stats_str += rmse_per_plot.to_string()

        self.display_widget.setPlainText(stats_str)

    def ANOVA(self):
        self.display_widget.clear()

        # Creating column headers
        columns = ['day', 'year'] + [f'plot{i}' for i in range(1, 25)]

        # Function to generate temperature data with sequential days
        def generate_temperature_data():
            data = {
                'day': np.arange(1, 101),  # Sequential days from 1 to 100
                'year': np.random.randint(2000, 2023, 100),  # Assuming years from 2000 to 2022
            }
            for i in range(1, 25):
                data[f'plot{i}'] = np.random.uniform(20, 30, 100)  # Random data for plot columns within the range
            return data

        # Generating ground truth temperature data with sequential days
        g_truth_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Generating thermal image temperature data with sequential days
        thermal_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Extracting temperature data for ANOVA test
        temperature_columns = g_truth_data.columns[2:]  # Columns containing temperature data

        # Performing ANOVA for each plot temperature between g_truth_data and thermal_data
        p_values = []
        for column in temperature_columns:
            p_value = f_oneway(g_truth_data[column], thermal_data[column]).pvalue
            p_values.append(p_value)

        # Displaying p-values for ANOVA tests
        stats_str = "ANOVA p-values for each plot temperature:\n"
        for i, column in enumerate(temperature_columns):
            stats_str += f'ANOVA p-value for {column}: {p_values[i]}\n'

        self.display_widget.setPlainText(stats_str)

    # visualization
    def Graph1(self):
        self.display_widget.clear()
        # Code for Bland-Altman analysis
        columns = ['day', 'year'] + [f'plot{i}' for i in range(1, 25)]

        def generate_temperature_data():
            data = {
                'day': np.arange(1, 101),
                'year': np.random.randint(2000, 2023, 100),
            }
            for i in range(1, 25):
                data[f'plot{i}'] = np.random.uniform(20, 30, 100)
            return data

        g_truth_data = pd.DataFrame(generate_temperature_data(), columns=columns)
        thermal_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        differences = g_truth_data.iloc[:, 2:] - thermal_data.iloc[:, 2:]
        means = (g_truth_data.iloc[:, 2:] + thermal_data.iloc[:, 2:]) / 2
        mean_difference = differences.mean().mean()

        plt.figure(figsize=(8, 6))
        plt.scatter(means.values.flatten(), differences.values.flatten(), alpha=0.7)
        plt.axhline(mean_difference, color='red', linestyle='--', label=f'Mean Difference: {mean_difference:.2f}')
        plt.xlabel('Mean of Measurements')
        plt.ylabel('Difference between Measurements')
        plt.title('Bland-Altman Plot: g_truth_data vs thermal_data')
        plt.legend()
        plt.grid(True)
        plt.show()

    # graph two for bland-altma
    def Graph2(self):
        self.display_widget.clear()

        # Creating column headers
        columns = ['day', 'year'] + [f'plot{i}' for i in range(1, 25)]

        # Function to generate temperature data with sequential days
        def generate_temperature_data():
            data = {
                'day': np.arange(1, 101),  # Sequential days from 1 to 100
                'year': np.random.randint(2000, 2023, 100),  # Assuming years from 2000 to 2022
            }
            for i in range(1, 25):
                data[f'plot{i}'] = np.random.uniform(20, 30, 100)  # Random data for plot columns within the range
            return data

        # Generating ground truth temperature data with sequential days
        g_truth_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Generating thermal image temperature data with sequential days
        thermal_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Calculating the average and difference between ground truth and thermal image data
        average = (g_truth_data.mean(axis=1) + thermal_data.mean(axis=1)) / 2
        difference = g_truth_data.mean(axis=1) - thermal_data.mean(axis=1)

        # Creating the Difference vs. Average plot
        plt.figure(figsize=(8, 6))
        plt.scatter(average, difference, alpha=0.7)
        plt.axhline(0, color='red', linestyle='--', label='Mean Difference: 0')
        plt.xlabel('Average')
        plt.ylabel('Difference')
        plt.title('Difference vs. Average Plot')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # graph for correlation
    def correlationGraph(self):
        self.display_widget.clear()

        # Creating column headers
        columns = ['day', 'year'] + [f'plot{i}' for i in range(1, 25)]

        # Function to generate temperature data with sequential days
        def generate_temperature_data():
            data = {
                'day': np.arange(1, 101),  # Sequential days from 1 to 100
                'year': np.random.randint(2000, 2023, 100),  # Assuming years from 2000 to 2022
            }
            for i in range(1, 25):
                data[f'plot{i}'] = np.random.uniform(20, 30, 100)  # Random data for plot columns within the range
            return data

        # Generating ground truth temperature data with sequential days
        g_truth_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Generating thermal image temperature data with sequential days
        thermal_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Calculate correlation coefficients for each plot
        correlation_values = []
        temperature_columns = [f'plot{i}' for i in range(1, 25)]
        for column in temperature_columns:
            correlation = g_truth_data[column].corr(thermal_data[column])
            correlation_values.append(correlation)

        # Creating a heatmap for correlation coefficients
        plt.figure(figsize=(8, 6))
        sns.heatmap(np.array(correlation_values).reshape(1, -1), annot=True, cmap='coolwarm', cbar=True)
        plt.xlabel('Plots')
        plt.ylabel('Correlation Coefficient')
        plt.title('Correlation between Ground Truth and Thermal Image Data per Plot')
        plt.xticks(ticks=np.arange(len(temperature_columns)) + 0.5, labels=temperature_columns, rotation=45)
        plt.yticks([0], ['Correlation'])
        plt.tight_layout()
        plt.show()

    # graph for MAE
    def MAEGraph(self):
        self.display_widget.clear()

        # Creating column headers
        columns = ['day', 'year'] + [f'plot{i}' for i in range(1, 25)]

        # Function to generate temperature data with sequential days
        def generate_temperature_data():
            data = {
                'day': np.arange(1, 101),  # Sequential days from 1 to 100
                'year': np.random.randint(2000, 2023, 100),  # Assuming years from 2000 to 2022
            }
            for i in range(1, 25):
                data[f'plot{i}'] = np.random.uniform(20, 30, 100)  # Random data for plot columns within the range
            return data

        # Generating ground truth temperature data with sequential days
        g_truth_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Generating thermal image temperature data with sequential days
        thermal_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Calculate correlation coefficients for each plot
        correlation_values = []
        temperature_columns = [f'plot{i}' for i in range(1, 25)]
        for column in temperature_columns:
            correlation = g_truth_data[column].corr(thermal_data[column])
            correlation_values.append(correlation)

        # Calculate MAE for each plot
        mae_values = []
        for column in temperature_columns:
            mae = np.abs(g_truth_data[column] - thermal_data[column]).mean()
            mae_values.append(mae)

        # Creating box plot for MAE values
        plt.figure(figsize=(10, 6))
        plt.boxplot(mae_values, patch_artist=True, showmeans=True)
        plt.xlabel('MAE')
        plt.title('Distribution of Mean Absolute Error (MAE) across Plots')
        plt.xticks([1], ['MAE'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # graph for rmse
    def RMSEGraph(self):
        self.display_widget.clear()

        # Creating column headers
        columns = ['day', 'year'] + [f'plot{i}' for i in range(1, 25)]

        # Function to generate temperature data with sequential days
        def generate_temperature_data():
            data = {
                'day': np.arange(1, 101),  # Sequential days from 1 to 100
                'year': np.random.randint(2000, 2023, 100),  # Assuming years from 2000 to 2022
            }
            for i in range(1, 25):
                data[f'plot{i}'] = np.random.uniform(20, 30, 100)  # Random data for plot columns within the range
            return data

        # Generating ground truth temperature data with sequential days
        g_truth_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Generating thermal image temperature data with sequential days
        thermal_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Calculate correlation coefficients for each plot
        correlation_values = []
        temperature_columns = [f'plot{i}' for i in range(1, 25)]
        for column in temperature_columns:
            correlation = g_truth_data[column].corr(thermal_data[column])
            correlation_values.append(correlation)

        # Calculate RMSE for each plot
        rmse_values = []
        for column in temperature_columns:
            rmse = np.sqrt(((g_truth_data[column] - thermal_data[column]) ** 2).mean())
            rmse_values.append(rmse)

        # Creating violin plot for RMSE values
        plt.figure(figsize=(10, 6))
        sns.violinplot(y=rmse_values, color='skyblue', inner='quartile')
        plt.ylabel('RMSE')
        plt.title('Distribution of Root Mean Squared Error (RMSE) across Plots')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # graph for time series  visualization
    def TimeSeriesGraph(self):
        self.display_widget.clear()

        # Creating column headers
        columns = ['day', 'year'] + [f'plot{i}' for i in range(1, 25)]

        # Function to generate temperature data with sequential days
        def generate_temperature_data():
            data = {
                'day': np.arange(1, 101),  # Sequential days from 1 to 100
                'year': np.random.randint(2000, 2023, 100),  # Assuming years from 2000 to 2022
            }
            for i in range(1, 25):
                data[f'plot{i}'] = np.random.uniform(20, 30, 100)  # Random data for plot columns within the range
            return data

        # Generating ground truth temperature data with sequential days
        g_truth_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Generating thermal image temperature data with sequential days
        thermal_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Get the number of plots
        num_plots = len(g_truth_data.columns[2:])

        # Plot temperature variations for all plots
        for i in range(num_plots):
            plot_num = i + 1
            plt.figure(figsize=(8, 5))
            plt.plot(g_truth_data['day'], g_truth_data[f'plot{plot_num}'], label=f'Ground Truth Plot {plot_num}')
            plt.plot(thermal_data['day'], thermal_data[f'plot{plot_num}'], label=f'Thermal Image Plot {plot_num}',
                     linestyle='--')
            plt.xlabel('Day')
            plt.ylabel('Temperature')
            plt.title(f'Plot {plot_num} Temperature Variations over Time')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    # Graph for scatter plot
    def scatter_plots(self):
        self.display_widget.clear()

        # Creating column headers
        columns = ['day', 'year'] + [f'plot{i}' for i in range(1, 25)]

        # Function to generate temperature data with sequential days
        def generate_temperature_data():
            data = {
                'day': np.arange(1, 101),  # Sequential days from 1 to 100
                'year': np.random.randint(2000, 2023, 100),  # Assuming years from 2000 to 2022
            }
            for i in range(1, 25):
                data[f'plot{i}'] = np.random.uniform(20, 30, 100)  # Random data for plot columns within the range
            return data

        # Generating ground truth temperature data with sequential days
        g_truth_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Generating thermal image temperature data with sequential days
        thermal_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Get the number of plots
        num_plots = len(g_truth_data.columns[2:])

        # Display scatter plots for each plot
        for i in range(num_plots):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(g_truth_data[f'plot{i + 1}'], thermal_data[f'plot{i + 1}'])
            ax.plot([20, 30], [20, 30], color='red', linestyle='--')  # Adding a y=x line for reference
            ax.set_xlabel(f'Ground Truth Plot {i + 1}')
            ax.set_ylabel(f'Thermal Image Plot {i + 1}')
            ax.set_title(f'Scatter Plot for Plot {i + 1}')
            ax.grid(True)
            ax.legend(['y = x'])
            plt.show()

    # graph for anova
    def anovaGraph(self):
        self.display_widget.clear()

        # Creating column headers
        columns = ['day', 'year'] + [f'plot{i}' for i in range(1, 25)]

        # Function to generate temperature data with sequential days
        def generate_temperature_data():
            data = {
                'day': np.arange(1, 101),  # Sequential days from 1 to 100
                'year': np.random.randint(2000, 2023, 100),  # Assuming years from 2000 to 2022
            }
            for i in range(1, 25):
                data[f'plot{i}'] = np.random.uniform(20, 30, 100)  # Random data for plot columns within the range
            return data

        # Generating ground truth temperature data with sequential days
        g_truth_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Generating thermal image temperature data with sequential days
        thermal_data = pd.DataFrame(generate_temperature_data(), columns=columns)

        # Performing ANOVA for each plot temperature between g_truth_data and thermal_data
        temperature_columns = columns[2:]  # Exclude 'day' and 'year'
        p_values = []
        for column in temperature_columns:
            p_value = f_oneway(g_truth_data[column], thermal_data[column]).pvalue
            p_values.append(p_value)

        # Creating a bar plot to visualize ANOVA results
        plt.figure(figsize=(10, 6))
        plt.bar(temperature_columns, p_values, color='skyblue')
        plt.xlabel('Plot Number')
        plt.ylabel('ANOVA p-value')
        plt.title('ANOVA p-values for Temperature Measurements per Plot')
        plt.xticks(rotation=45, ha='right')
        plt.yscale('log')  # Using a logarithmic scale for better visualization of small p-values
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


